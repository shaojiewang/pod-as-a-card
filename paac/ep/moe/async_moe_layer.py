# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from .experts import GroupedMLP, SequentialMLP, TEGroupedMLP
# from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from .router import TopKRouter
# from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from .shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.training import get_args

#from .moe_alltoall_overlap import MoELayerOverlapAll2All
from .moe_utils import (
    forward_func, 
    backward_func,
    sort_chunks_by_idxs,
)
from megatron.core.transformer.moe.moe_utils import (
    get_capacity,
    permute,
    unpermute,
)

from .ccl_utils import (
    async_all_to_all,
)

@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_extended_tp:
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
            )

        self.use_shared_expert = None # self.config.moe_shared_expert_intermediate_size is not None

        self.shared_expert_overlap = None # self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayerOverlapAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, moe_layer: BaseMoELayer):
        # args = get_args()
        save_tensors = []
        save_tensors_for_grad = []
        ctx.input_shape = hidden_states.shape
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True

        # router
        with torch.enable_grad():
            scores, indices = moe_layer.router(hidden_states)

        save_tensors.append(scores)
        scores = scores.detach()
        scores.requires_grad = True
        save_tensors.append(scores)

        # print(f"forward scores={scores}")

        save_tensors.append(indices)

        if moe_layer.use_shared_expert:
            ctx.shared_experts = moe_layer.shared_experts
            #TODO: support TP here
        
        def alltoall_token_permutation1(hidden_states, indices):
            tokens_per_expert = moe_layer.token_dispatcher.preprocess(indices)
            # Flatten the input tensor
            # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
            hidden_states = hidden_states.view(-1, ctx.input_shape[-1])
            
            # TODO: support TP here

            # Permutation 1: input to AlltoAll input
            moe_layer.token_dispatcher.local_input_tokens_global_experts_indices = indices
            permutated_local_input_tokens, moe_layer.token_dispatcher.reversed_local_input_permutation_mapping = permute(
                hidden_states, moe_layer.token_dispatcher.local_input_tokens_global_experts_indices, 
            )
            return tokens_per_expert, permutated_local_input_tokens

        (tokens_per_expert, permutated_local_input_tokens), *_ = forward_func(alltoall_token_permutation1,
                                                                          (hidden_states, indices))
        # permute 1
        save_tensors.append(permutated_local_input_tokens)

        #permutated_local_input_tokens = permutated_local_input_tokens.detach()
        #permutated_local_input_tokens.require_grad = True 
        #save_tensors.append(permutated_local_input_tokens)

        # Perform expert parallel AlltoAll communication
        ep_group = parallel_state.get_expert_model_parallel_group()
        global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            ep_group,
            permutated_local_input_tokens,
            moe_layer.token_dispatcher.output_splits,
            moe_layer.token_dispatcher.input_splits,
            True,
        )

        ctx.output_splits = moe_layer.token_dispatcher.output_splits
        ctx.input_splits = moe_layer.token_dispatcher.input_splits
        ctx.router_topk = moe_layer.token_dispatcher.router_topk

        permute1_ep_all_to_all_handle.wait()

        # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
        if moe_layer.num_local_experts > 1:
            def alltoall_token_permutation2(global_input_tokens):
                global_input_tokens, moe_layer.token_dispatcher.reversed_global_input_permutation_mapping = permute(
                    global_input_tokens, moe_layer.token_dispatcher.global_input_tokens_local_experts_indices
                )

                # TODO: add TP support here

                return global_input_tokens

            # token 重排2 input
            (global_input_tokens), global_input_tokens_detach = forward_func(alltoall_token_permutation2,
                                                                             global_input_tokens)
            save_tensors.append(global_input_tokens_detach)
            save_tensors.append(global_input_tokens)
            save_tensors_for_grad.append(global_input_tokens_detach)
            global_input_tokens_detach.untyped_storage().resize_(0)

            
        # routed expert
        (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts, (global_input_tokens, tokens_per_expert))
        global_input_tokens_detach = global_input_tokens.detach()
        global_input_tokens_detach.requires_grad = True
        save_tensors.append(global_input_tokens_detach)
        save_tensors.append(expert_output)

        expert_output = expert_output.detach()

        print(f"permutated_local_input_tokens={permutated_local_input_tokens}")

        print(f"global_input_tokens={global_input_tokens}")

        print(f"global_input_tokens_detach={global_input_tokens_detach}")

        save_tensors.append(hidden_states)
        ctx.save_for_backward(*save_tensors)

        return expert_output, None 

    @staticmethod
    def backward(ctx, *args):
        # global_args = get_args()
        (route_graph, 
         detach_scores,
         indices, 
         permute1_graph,
         permute2_input_detach,
         permute2_graph,
         experts_input_detach,
         experts_graph,
         detach_input,
        ) = ctx.saved_tensors

        ctx.save_for_backward()

        output_splits = ctx.output_splits
        input_splits = ctx.input_splits
        router_topk = ctx.router_topk
       
        print(f"args={args}")

        backward_func(experts_graph, args[0])
       
        print(f"experts_input_detach.grad={experts_input_detach.grad}") 

        backward_func(permute2_graph, experts_input_detach.grad)        
        #backward_func(permute2_graph, args[0])        

        # permute1_graph.backward(args[0])
        ep_group = parallel_state.get_expert_model_parallel_group()
        permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
            ep_group,
            permute2_input_detach.grad,
            input_splits,
            output_splits,
            True,
        )
        
        bw_permute1_ep_all2all_handle.wait()

        backward_func(permute1_graph, permute1_backward_input)


        #print(detach_scores.grad)

        # route_graph.backward(detach_scores.grad)
        route_graph = None
        grad_output = detach_input.grad
        return grad_output, None

class AsyncMoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(AsyncMoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        self.use_shared_expert = False # config.enable_shared_expert

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize experts
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules.experts, MLPSubmodules):
                self.experts = TEGroupedMLP(
                    self.num_local_experts, self.config, self.submodules.experts
                )
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules.experts, MLPSubmodules)
            self.experts = SequentialMLP(
                self.num_local_experts, self.config, self.submodules.experts
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(self.config, self.submodules.shared_experts)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        
        return MoELayerOverlapAll2All.apply(hidden_states, self)

        # process MoE
        def custom_forward(hidden_states):
            probs, indices = self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, indices
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output += self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias


