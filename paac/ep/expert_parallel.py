#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import socket
import fcntl
import struct
import argparse
import warnings

import nvtx

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from test_utilities import Utils

from moe.async_moe_layer import AsyncMoELayer
from moe.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from moe.transformer_config import MoETransformerConfig

class MoEModel():
    def __init__():
        if "OMPI_COMM_WORLD_SIZE" in os.environ:
            # Execution with `mpirun -np N`
            WORLD_RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
            WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
            LOCAL_RANK = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
            LOCAL_SIZE = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
            opts.tcp_init = True
            opts.bind_to_device = True
            opts.bootstrap_backend = "mpi"
        elif "TORCHELASTIC_RUN_ID" in os.environ:
            WORLD_RANK = int(os.getenv("RANK", "0"))
            WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
            LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
            LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        else:
            raise RuntimeError(f"{__file__} must be launched with either `mpirun` or `torchrun`!")

        print(f"WORLD_RANK={WORLD_RANK}, LOCAL_RANK={LOCAL_RANK}, LOCAL_SIZE={LOCAL_SIZE}, WORLD_SIZE={WORLD_SIZE}")

        tp_size = 1
        pp_size = 1
        ep_size = 2
        cp_size = 1

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        num_moe_experts = 32
        enable_shared_expert = True
        num_shared_experts = 2
        hidden_size = 1536
        moe_shared_expert_intermediate_size = num_shared_experts * hidden_size
        moe_token_dispatcher_type = "alltoall"
        grouped_gemm = True

        self.transformer_config = MoETransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            enable_shared_expert=enable_shared_expert,
            num_shared_experts=num_shared_experts,
            moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=6,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=True,
        )
        
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )

        print(f"submodules={transformer_layer_spec.submodules.mlp.submodules}")

        self.moe_layer = AsyncMoELayer(
            self.transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )

        self.moe_layer.cuda()
        
    def run_fwd_bwd():
        # [sequence length, batch size, hidden size]
        seq_len = 4096
        batch_size = 2
        hidden_states = torch.rand(
            (seq_len, batch_size, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        hidden_states.retain_grad()

        moe_layer_ = self.moe_layer.bfloat16()

        for i in range(20):
            with nvtx.annotate(f"iteration{i}", color="red"):
                with nvtx.annotate(f"forward", color="green"):
                    output_smm, _ = moe_layer_(hidden_states)
                with nvtx.annotate(f"backward", color="blue"):
                    output_smm.mean().backward()

        Utils.destroy_model_parallel()

if __name__ == "__main__":
    moe_layer_test = MoEModel()
    moe_layer_test.train()

