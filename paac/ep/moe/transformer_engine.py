# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import os
import warnings
from importlib.metadata import version
from typing import Callable

import torch
import transformer_engine as te
from pkg_resources import packaging
from torch import Tensor

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_and_expert_parallel_world_size,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""

    def get_te_version_str():
        if hasattr(te, '__version__'):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    return packaging.version.Version(get_te_version_str())


_te_version = get_te_version()

def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {"params_dtype": config.params_dtype}

    if _te_version >= packaging.version.Version("0.12.0"):
        if config.use_cpu_initialization:
            extra_transformer_engine_kwargs["device"] = 'cpu'
        else:
            extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()
    return extra_transformer_engine_kwargs


def condition_init_method(config, init_method):    
    """Condition TE init_method on config.perform_initialization."""
    return init_method if config.perform_initialization else (lambda w: None)

if _te_version >= packaging.version.Version("1.9.0.dev0"):

    class TEGroupedLinear(te.pytorch.GroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer.

        Note that if Megatron's parallel_state has not been initialized
        yet, the tp_group passed to TE will be None and must be set later
        via set_tensor_parallel_group().
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            parallel_mode: str,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
        ):
            self.config = config

            # TE returns a zero length Tensor when bias=False and
            # return_bias=True, but we prefer None.  So in that case we
            # tell TE to not return the bias, and return None
            # ourselves. This way our forward always returns two values
            # and we don't have to deal with the zero length Tensor.
            self.te_return_bias = skip_bias_add and bias
            self.is_first_microbatch = True
            self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache

            extra_kwargs = _get_extra_te_kwargs(config)
            extra_kwargs["ub_name"] = tp_comm_buffer_name

            self.expert_parallel = self.config.expert_model_parallel_size > 1
            if self.expert_parallel:
                extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

            # For MoE models, the comms between TP and EP group is explicitly handled by
            # MoE token dispatcher. So we disable comms by making TE agnostic of model parallel.
            self.explicit_expert_comm = is_expert and (
                config.tensor_model_parallel_size > 1 or self.expert_parallel
            )
            tp_group = get_tensor_model_parallel_group(check_initialized=False)
            if self.explicit_expert_comm and config.moe_extended_tp:
                tp_size = parallel_state.get_tensor_and_expert_parallel_world_size()
            else:
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if self.explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                parallel_mode = None
                tp_size = 1
                tp_group = None

            super().__init__(
                num_gemms=num_gemms,
                in_features=input_size,
                out_features=output_size,
                sequence_parallel=self.config.sequence_parallel,
                fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
                tp_group=tp_group,
                tp_size=tp_size,
                get_rng_state_tracker=(
                    get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
                ),
                init_method=condition_init_method(config, init_method),
                bias=bias,
                return_bias=self.te_return_bias,
                parallel_mode=parallel_mode,
                **extra_kwargs,
            )

            for param in self.parameters():
                setattr(param, 'allreduce', not (is_expert and self.expert_parallel))

        def forward(self, x, m_splits):
            """Forward."""
            _is_first_microbatch = (
                None if self.disable_parameter_transpose_cache else self.is_first_microbatch
            )
            out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
            self.is_first_microbatch = False

            # TE only returns a tuple when return_bias is True, otherwise
            # it returns a single Tensor, we always want to return two
            # values regardless of the arguments.
            if self.te_return_bias:
                return out
            return out, None

        def _sharded_state_dict_grouped(
            self, tp_axis_map, prefix='', sharded_offsets=(), metadata=None
        ):
            """
            prefix should be module_name to make keys identical to sequetial ones.
            """
            sharded_state_dict = {}
            full_state_dict = self.state_dict(prefix='', keep_vars=True)
            num_global_experts = (
                parallel_state.get_expert_model_parallel_world_size() * self.num_gemms
            )
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_gemms
            )
            ep_axis = len(sharded_offsets)
            for gemm_idx in range(self.num_gemms):
                state_dict = {
                    f'{gemm_idx}.weight': full_state_dict[f'weight{gemm_idx}'],
                    f'{gemm_idx}._extra_state': full_state_dict['_extra_state'],
                }
                if self.use_bias:
                    state_dict[f'{gemm_idx}.bias'] = full_state_dict[f'bias{gemm_idx}']
                sub_sd = make_sharded_tensors_for_checkpoint(
                    state_dict,
                    '',
                    tp_axis_map,
                    (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + gemm_idx, num_global_experts),
                    ),
                )
                # Remove expert layers indexing from sharded keys
                replace_prefix_for_sharding(sub_sd, f'{gemm_idx}.', prefix)
                sharded_state_dict.update(
                    {
                        f'{prefix}weight{gemm_idx}': sub_sd[f'{gemm_idx}.weight'],
                        # TODO: TE's GroupedLinear only has one _extra_state for all experts.
                        # We need sharding or build/merge fn to handle _extra_state correctly.
                        f'{prefix}_extra_state{"" if gemm_idx == 0 else gemm_idx}': sub_sd[
                            f'{gemm_idx}._extra_state'
                        ],
                    }
                )
                if self.use_bias:
                    sharded_state_dict[f'{prefix}bias{gemm_idx}'] = sub_sd[f'{gemm_idx}.bias']
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in sharded_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(),
                )
            return sharded_state_dict

    class TEColumnParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to column-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: str = None,
        ):
            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="column",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

        def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 0, bias sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            print("TEColumnParallelGroupedLinear---shareded")
            tp_axis_map = {}
            for gemm_idx in range(self.num_gemms):
                tp_axis_map.update({f'{gemm_idx}.weightxxxz': 0, f'{gemm_idx}.bias': 0})
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

    class TERowParallelGroupedLinear(TEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
        to row-parallel style.
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            tp_comm_buffer_name: str = None,
        ):

            super().__init__(
                num_gemms=num_gemms,
                input_size=input_size,
                output_size=output_size,
                parallel_mode="row",
                config=config,
                init_method=condition_init_method(config, init_method),
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

        def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
            """
            For each gemm, sharding along axis 1, bias not sharded.
            Assume sharded_offsets[-1] is the expert parallel offset.
            """
            tp_axis_map = {f'{gemm_idx}.weight': 1 for gemm_idx in range(self.num_gemms)}
            return super()._sharded_state_dict_grouped(
                tp_axis_map, prefix, sharded_offsets, metadata
            )

else:

    TEGroupedLinear = None
    TEColumnParallelGroupedLinear = None
    TERowParallelGroupedLinear = None
