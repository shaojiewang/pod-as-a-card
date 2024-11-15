import torch
from megatron.training import get_args
from megatron.core import mpu


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


