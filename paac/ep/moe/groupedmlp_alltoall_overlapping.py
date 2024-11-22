import torch

from .moe_utils import (
    forward_func, 
    backward_func,
    get_gemm_backward_need_tensors,
    set_all2all_experts_output,
)

from .ccl_utils import (
    async_all_to_all,
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

backend = grouped_gemm.backend if grouped_gemm_is_available() else None

class GroupedMlpAlltoallOverlapping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights1, weights2, args, moe_layer_ctx):
        original_weight1, original_weight2, activation_func, tokens_per_expert, layer_number = args
        fc1_output = backend.gmm(
            inputs, weights1, tokens_per_expert, trans_a=False, trans_b=False
        )

        act_out, detached_act_inputs = forward_func(activation_func, fc1_output)

        fc2_output = backend.gmm(act_out, weights2, tokens_per_expert, trans_a=False, trans_b=False)

        ctx.save_for_backward(inputs, detached_act_inputs, act_out, weights1, weights2, original_weight1, original_weight2, tokens_per_expert)
        
        return fc2_output, None
    
    @staticmethod
    def backward(ctx, *grad_outs):
        inputs, act_inputs, mm2_inputs, weights1, weights2, original_weight1, original_weight2, tokens_per_expert = ctx.saved_tensors
        grad_outs = grad_outs[0]

        (ep_group, permute2_input_detach, permute2_graph, output_splits, input_splits) = get_gemm_backward_need_tensors()

        grad_gmm2_inputs = backend.gmm(
            grad_outs, weights2, tokens_per_expert, trans_a=False, trans_b=True
        )

        grad_weight2 = backend.gmm(
            mm2_inputs, grad_outs, tokens_per_expert, trans_a=True, trans_b=False
        )


        act_graph = mm2_inputs
        # grad of activation_func
        grad_outs.untyped_storage().resize_(0)
        # mm2_inputs.untyped_storage().resize_(0)

        act_graph.backward(grad_gmm2_inputs)
        grad_gmm2_inputs.untyped_storage().resize_(0)
        act_inputs.untyped_storage().resize_(0)

        mm1_inputs_grad = backend.gmm(
            act_inputs.grad, weights1, tokens_per_expert, trans_a=False, trans_b=True
        )

        backward_func(permute2_graph, mm1_inputs_grad)

        #TODO: add tp

        permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
            ep_group,
            permute2_input_detach.grad,
            input_splits,
            output_splits,
            True,
        )        

        grad_weight1 = backend.gmm(
            inputs, act_inputs.grad, tokens_per_expert, trans_a=True, trans_b=False
        )

        bw_permute1_ep_all2all_handle.wait()

        set_all2all_experts_output((permute1_backward_input))

        return mm1_inputs_grad, grad_weight1, grad_weight2, None, None
        

def grouped_mlp_all2all_overlapping(inputs, weights1, weights2, args, ctx):
    return GroupedMlpAlltoallOverlapping.apply(inputs, weights1, weights2, args, ctx)

