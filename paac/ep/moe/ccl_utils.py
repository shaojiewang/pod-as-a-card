import torch
import torch.distributed
import torch.distributed as dist

class _AllToAllAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, overlap: bool = False):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )

        if overlap:
            handle = torch.distributed.all_to_all_single(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=overlap,
            )

        else:
            handle = torch.distributed.all_to_all_single(
                output,
                input,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=False,
            )
        return output, handle

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAllAsync.apply(ctx.group, grad_output[0], ctx.input_split_sizes, ctx.output_split_sizes)[0],
            None,
            None,
            None,
        )

def async_all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None, async_op: bool = False):
    """Wrapper for autograd function"""
    return _AllToAllAsync.apply(group, input_, output_split_sizes_, input_split_sizes, async_op)




