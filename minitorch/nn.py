from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # New Height and New Width is equal to the folding of the kernel height and width.
    new_height = height // kh
    new_width = width // kw

    # Makes the input contiguous and then reshapes it to the new shape through view.
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # Using the permute function to rearrange the memory of the tensor,
    # so that we can view into the shape we want.
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    # View the tensor into the new shape.
    input = input.view(batch, channel, new_height, new_width, kh * kw)
    return input, new_height, new_width


def avgpool2d(t: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling 2D over a tensor t using a given kernel size

    Args:
    ----
        t: Input tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width that has been pooled over the original tensor using the tile function.

    """
    batch, channel, _, _ = t.shape
    # Reshape the input tensor using the tile function
    tiled, new_height, new_width = tile(t, kernel)
    # Once data is returned from Tile, each element in the last dimension is one element for the kernel.
    # We just need to the the average of the last dimension to calculate the average pooling.
    # We then reshape the tensor to the new shape.
    output = tiled.mean(dim=-1).view(batch, channel, new_height, new_width)
    return output


# Implemented for Task 4.4

# Defined the max reduction function
max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(t: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        t: Input tensor of size batch x channel x height x width
        dim: Dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width that has been reduced over the specified dimension.

    """
    out = max_reduce(t, dim)
    # Create a 1-hot tensor with the same shape as the input tensor.
    # The 1-hot tensor is created by setting the index of the maximum value to 1 and the rest to 0.
    return out == t


class Max(Function):
    """Max operator for maxpool2d function"""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operator

        Args:
        ----
            ctx: Context object
            input: Input tensor batch x channel x height x width
            dim: Dimension to reduce over

        Returns:
        -------
            Tensor of size batch x channel x height x width that has been reduced over the specified dimension.

        """
        dim_int = int(dim.item())  # converts the dim to an integer
        ctx.save_for_backward(input, dim_int)
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max operator.

        Args:
        ----
            ctx: Context object
            grad_output: Gradient of the output tensor

        Returns:
        -------
            Tuple of the gradient of the input tensor.

        """
        input, dim_int = ctx.saved_tensors

        # only backprogate the values to the max values argmax tensor. 0 otherwise.
        return grad_output * argmax(input, dim_int), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction

    Args:
    ----
        input: Input tensor batch x channel x height x width
        dim: Dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width that has been reduced over the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: Input tensor batch x channel x height x width
        dim: Dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width that has been reduced over the specified dimension.

    """
    # Exp of the input tensor
    exp_input = input.exp()
    # Sum of the exp of the input tensor
    sum_exp_input = exp_input.sum(dim)
    # Divide the exp of the input tensor by the sum of the exp of the input tensor
    return exp_input / sum_exp_input


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: Input tensor batch x channel x height x width
        dim: Dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width that has been reduced over the specified dimension.

    """
    return input - input.exp().sum(dim=dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling 2D over a tensor t using a given kernel size

    Args:
    ----
        input: Input tensor batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width that has been pooled over the original tensor using the tile function.

    """
    batch, channel, _, _ = input.shape
    # Reshape the input tensor using the tile function
    tiled, new_height, new_width = tile(input, kernel)
    # Once data is returned from Tile, each element in the last dimension is one element for the kernel.
    # We just need to the the maximum of the last dimension to calculate the max pooling.
    # We then reshape the tensor to the new shape.
    output = max(tiled, dim=-1).view(batch, channel, new_height, new_width)
    return output


def dropout(input: Tensor, dropout_rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to a tensor

    Args:
    ----
        input: Input tensor batch x channel x height x width
        dropout_rate: Drop out rate
        ignore: if True, not dropout the values

    Returns:
    -------
        Tensor of size batch x channel x height x width that has been dropout over the specified dimension.

    """
    if ignore:
        return input
    else:
        # Dropout the values with a probability of dropout_rate. 1 if the value is greater than dropout_rate, 0 otherwise.
        return input * (rand(input.shape) > dropout_rate)
