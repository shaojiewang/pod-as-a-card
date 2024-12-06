import torch
from megatron.training import get_args
from megatron.core import mpu

from pkg_resources import packaging

from megatron.core.transformer.custom_layers.transformer_engine import _te_version
import transformer_engine as te
import logging

logger = logging.getLogger(__name__)

if _te_version >= packaging.version.Version("1.11.0.dev0"):
    fused_permute = te.pytorch.permutation.moe_permute
    fused_unpermute = te.pytorch.permutation.moe_unpermute
else:
    logger.warning(f"Dispatcher permute fusion depends on _te_version > 1.11.0, current version is {_te_version}, \
                   try upgrading your transformer_engine version")
    fused_permute = None
    fused_unpermute = None

def permute(tokens, indices, permute_fusion: bool = False):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens, topk].
        topk (int, optional): The topk value. Defaults to 1.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    if indices.dim() == 1:
        indices = indices.unsqueeze(1)
    if permute_fusion and fused_permute is not None:
        permuted_tokens, sorted_indices = fused_permute(
            tokens, indices, -1
        )
        return permuted_tokens, sorted_indices
    
    topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, 
              sorted_indices, 
              probs: torch.Tensor = None, 
              unpermute_fusion: bool = False
            ):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.
        topk (int, optional): The number of top tokens to consider for merging with probabilities. Defaults to 1.
    """
    assert sorted_indices.numel() == permuted_tokens.size(0)
    
    if unpermute_fusion and fused_unpermute is not None:
        unpermuted_tokens = fused_unpermute(permuted_tokens, sorted_indices, probs)
        return unpermuted_tokens

    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        topk = 1
    
    unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)

    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens

GMM_BWD_TENSORS_NEEDED = None
ALL2ALL_EXPERTS_OUTPUT = None

def set_gemm_backward_need_tensors(_inputs):
    global GEMM_BACKWARD_NEED_TENSORS
    GEMM_BACKWARD_NEED_TENSORS = _inputs


def get_gemm_backward_need_tensors():
    global GEMM_BACKWARD_NEED_TENSORS
    result = GEMM_BACKWARD_NEED_TENSORS
    GEMM_BACKWARD_NEED_TENSORS = None
    return result

def set_all2all_experts_output(_input):
    global ALL2ALL_EXPERTS_OUTPUT
    ALL2ALL_EXPERTS_OUTPUT = _input


def get_all2all_experts_output():
    global ALL2ALL_EXPERTS_OUTPUT
    result = ALL2ALL_EXPERTS_OUTPUT
    ALL2ALL_EXPERTS_OUTPUT = None
    return result


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output

def forward_func(func, inputs):
    def detach_tensor(input_):
        if input_.requires_grad and input_.grad_fn is None:
            return input_
        else:
            new_input = input_.detach()
            new_input.requires_grad = True
        return new_input

    detach_inputs = []
    if isinstance(inputs, tuple):
        for input_ in inputs:
            if isinstance(input_, tuple):
                detach_input = []
                for i in input_:
                    if isinstance(i, torch.Tensor) and torch.is_floating_point(i):
                        detach_input.append(detach_tensor(i))
                    else:
                        detach_input.append(i)
                detach_inputs.append(tuple(detach_input))
            else:
                if isinstance(input_, torch.Tensor) and torch.is_floating_point(input_):
                    detach_input = detach_tensor(input_)
                else:
                    detach_input = input_
                detach_inputs.append(detach_input)
    elif isinstance(inputs, torch.Tensor):
        detach_inputs.append(detach_tensor(inputs))

    with torch.enable_grad():
        output = func(*detach_inputs)

    return output, *detach_inputs


def backward_func(func_tensor, gradinputs):
    if gradinputs is None or func_tensor.grad_fn is None:
        return
    if isinstance(gradinputs, torch.Tensor):
        func_tensor.backward(gradinputs)
    elif isinstance(gradinputs, tuple):
        func_tensor.backward(*gradinputs)


