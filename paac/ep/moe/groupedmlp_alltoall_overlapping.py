import torch

from .moe_utils import (
    forward_func, 
    backward_func,
)

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


def grouped_gemm_is_available():
    """Check if grouped_gemm is available."""
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    """Assert that grouped_gemm is available."""
    assert grouped_gemm_is_available(), (
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`."
    )


ops = grouped_gemm.ops if grouped_gemm_is_available() else None

backend = grouped_gemm.backedn if grouped_gemm_is_available() else None

class GroupedMlpAlltoallOverlapping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, args, moe_layer_ctx):
        original_weight1, original_weight2, activation_func, tokens_per_expert, layer_number = args
        fc1_output = ops.gmm(
            inputs, weight1, tokens_per_expert, trans_b=False
        )

        act_out, detached_act_inputs = forward_func(activation_func, fc1_output)

        fc2_output = ops.gmm(act_out, weight2, tokens_per_expert, trans_b=False)

        ctx.save_for_backward(detached_act_inputs, act_out, weights1, weights2, original_weight1, original_weight2, tokens_per_expert)
        
        return fc2_output, _
    
    @staticmethod
    def backward(ctx, *grad_outs):
        act_inputs, mm2_inputs, weights1, weights2, original_weight1, original_weight2, original_weight1, original_weight2, tokens_per_expert = ctx.saved_tensors
        grad_outs = grad_outs[0]

        grad_gmm2_inputs = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=True)

        


def grouped_mlp_all2all_overlapping(inputs, weights1, weights2, args, ctx):
    return GroupedMlpAlltoallOverlapping.apply(inputs, weights1, weights2, args, ctx)
