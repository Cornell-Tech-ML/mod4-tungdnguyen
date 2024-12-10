"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend, Optional

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negate function for every element in a Tensor $f(t1) = -x$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negate"""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass. Returns $-grad_output$"""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Inverse function $f(t1) = 1/t1$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for inversing a"""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for inverse. Returns $grad_output/map(t1,square)$"""
        (t1,) = ctx.saved_tensors
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Addition function $f(t1, t2) = t1 + t2$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for add"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for Add."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true. Returns true for all elements if Dim is not specified."""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    """Multiplication function $f(t1,t2) = t1*t2$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for tensor multiplication"""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass. Returns $t2*d_output$ for t1 and $t1*d_output$ for t2"""
        (t1, t2) = ctx.saved_tensors
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    """Sigmoid function $f(12) = sigmoid(t1)$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass of sigmoid."""
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass. Returns $sigmoid(t1)*(1-sigmoid(t1))*grad_output$"""
        (t1,) = ctx.saved_tensors
        sigmoid_val = t1.f.mul_zip(
            t1.f.sigmoid_map(t1), (t1._ensure_tensor(1) - t1.f.sigmoid_map(t1))
        )
        return grad_output.f.mul_zip(sigmoid_val, grad_output)


class ReLU(Function):
    """ReLU function $f(t1) = max(0,t1)$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass of ReLU"""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass. Returns grad_output if an element is > 0 and 0 otherwise"""
        (t1,) = ctx.saved_tensors
        return t1.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Log function $f(t1) = log_map(t1)$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for log of float a"""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for Log. Returns $d_output/a$"""
        (t1,) = ctx.saved_tensors
        return grad_output / t1


class Exp(Function):
    """Exp function $f(t1) = exp(t1)$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass of exp"""
        output = t1.f.exp_map(t1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass. Returns $exp(t1)*grad_output$"""
        (output,) = ctx.saved_tensors
        return grad_output.f.mul_zip(output, grad_output)


class Sum(Function):
    """Sum a dimension in the tensor"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass of sum"""
        ctx.save_for_backward(t1.shape, int(dim.item()))
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Sum. Returns grad_output expanded to the original tensor's shape."""
        (t1_shape, dim) = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    """Less than function $f(t1,t2) = x<y"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass of Less Than."""
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass of less than. Returns zeros since the function is non-differentiable."""
        return grad_output._ensure_tensor(0.0), grad_output._ensure_tensor(0.0)


class EQ(Function):
    """Equal function $f(x,y) = x==y"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass of Equal To."""
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass of equal. Returns zeros since the function is non-differentiable."""
        return grad_output._ensure_tensor(0.0), grad_output._ensure_tensor(0.0)


class IsClose(Function):
    """Equal function $f(x,y) = x==y"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass of IsClose: Check if t1 and t2 tensors are close"""
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    """Permute function for tensor"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Forward pass of Permute."""
        ctx.save_for_backward(order)
        return t1._new(t1._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass of permute. Returns the original tensor"""
        # Calculate the inverse permutation
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Returns a new Tensor with a different shape (Add dims to a Tensor)"""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_tensors

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    r"""Computes an approximation to the derivative of a Tensor with respect to all args.

    Args:
    ----
        f : arbitrary function from a Tensor args to one value
        *vals : Tensors from $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant
        ind: the Index of the arg to compute the derivative

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
