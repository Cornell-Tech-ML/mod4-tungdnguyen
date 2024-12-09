from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


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
        t: Input tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
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