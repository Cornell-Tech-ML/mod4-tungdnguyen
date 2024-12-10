from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Unpacks vals
    mutable_vals = list(vals)

    # Creates x+epsilon and x-epsilon.
    positive_vals = mutable_vals.copy()
    positive_vals[arg] += epsilon

    negative_vals = mutable_vals.copy()
    negative_vals[arg] -= epsilon
    a = (f(*positive_vals) - f(*negative_vals)) / (2 * epsilon)
    print(
        "central difference is: ",
        a,
        positive_vals,
        negative_vals,
        mutable_vals,
        epsilon,
        f,
    )
    return a


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """Is the scalar a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the input variables of this Variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for input variables."""
        ...


def visit(
    variable: Variable, visited_variables: set[int], topo_list: List[Variable]
) -> None:
    """Performs DFS on a Variable.

    Args:
    ----
        variable: Current visiting Variable.
        visited_variables: Set of variables' unique_id visited by the function. Being in the set means the Variable is marked.
        topo_list: List of topologically sorted list of Variables. The right-most Variables is the final output of the network.

    """
    if variable.unique_id in visited_variables:
        return
    for parent in variable.parents:
        if parent.unique_id not in visited_variables and not parent.is_constant():
            visit(parent, visited_variables, topo_list)
    visited_variables.add(variable.unique_id)
    topo_list.append(variable)


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited_variables = set()
    topo_list = []
    visit(variable, visited_variables, topo_list)
    return topo_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    topo_list = list(topological_sort(variable))

    derivatives = {}
    rightest = topo_list[-1]
    if rightest.is_leaf():
        variable.accumulate_derivative(0)
        return

    # if rightmost node is not leaf, starts propagating
    derivatives[rightest.unique_id] = deriv
    for variable in reversed(topo_list):
        if variable.is_leaf():
            derivatives.get(variable.unique_id, 0.0)
            variable.accumulate_derivative(derivatives[variable.unique_id])
            continue

        chain_rule_outputs = variable.chain_rule(derivatives[variable.unique_id])
        # Update gradient for each of the parents.
        for parent, output in chain_rule_outputs:
            if parent.unique_id not in derivatives:
                derivatives[parent.unique_id] = 0.0
            derivatives[parent.unique_id] += output


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns saved values from the forward pass during backpropagation."""
        return self.saved_values
