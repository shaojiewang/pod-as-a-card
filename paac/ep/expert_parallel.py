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

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from test_utilities import Utils

from moe.async_moe_layer import AsyncMoELayer

def __train():
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
    NUM_NODES = WORLD_SIZE // LOCAL_SIZE

    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)

    num_moe_experts = 64
    hidden_size = 12
    moe_token_dispatcher_type = "alltoall"
    grouped_gemm = True

    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=True,
        moe_token_dispatcher_type=moe_token_dispatcher_type,
        moe_router_topk=2,
        moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=grouped_gemm,
        add_bias_linear=False,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        moe_router_load_balancing_type="sinkhorn",
    )
    
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
    )
    moe_layer = MoELayer(
        transformer_config, transformer_layer_spec.submodules.mlp.submodules
    )


    moe_layer.cuda()
    # [sequence length, batch size, hidden size]
    seq_len = 32
    batch_size = 2
    hidden_states = torch.rand(
        (seq_len, batch_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    hidden_states.retain_grad()

    output_smm, _ = moe_layer(hidden_states)
    output_smm.mean().backward()


    Utils.destroy_model_parallel()

    pass

if __name__ == "__main__":
    __train()

