"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A float representing the product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the input.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A float representing the product of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        A float representing the negation of x.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean representing whether x is less than y.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal
      Args:
        x: A float.
        y: A float.

    Returns
    -------
        A boolean representing whether x is equal to y.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers

    Args:
    ----
        x: A float.
        y: A float.

    """
    return x if x >= y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        A boolean representing whether x is close to y.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x: A float.

    Returns:
    -------
        Sigmoid value of x.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculates the relu function.

    Args:
    ----
        x: A float.

    Returns:
    -------
        ReLU value of x.

    """
    return x if x >= 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm function.

    Args:
    ----
        x: A positive float.

    Returns:
    -------
        Natural logarithm value of x.

    """
    if x <= 0:
        raise ValueError("Input must be a positive number")
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x: A float.

    Returns:
    -------
        Exponential value of x.

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Calculates the derivative of the logarithm function,  and times it with a scalar d.

    Args:
    ----
        x: A positive float.
        d: A float.

    Returns:
    -------
        Derivative of the logarithm function times d.

    """
    if x <= 0:
        raise ValueError("Input must be a positive number")
    return d / x


def inv(x: float) -> float:
    """Calculates the inverse of a number.

    Args:
    ----
        x: A non-zero float.

    Returns:
    -------
        Inverse value of x.

    """
    if x == 0:
        raise ValueError("Input must be a non-zero number")
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """Calculates the derivative of the inverse function,  and times it with a scalar d.

    Args:
    ----
        x: A non-zero float.
        d: A float.

    Returns:
    -------
        Derivative of the inverse function times d.

    """
    return -d / x**2


def relu_back(x: float, d: float) -> float:
    """Calculates the derivative of the relu function,  and times it with a scalar d.

    Args:
    ----
        x: A float.
        d: A float.

    Returns:
    -------
        Derivative of the relu function times d.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
# - zipWith
# - reduce


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Takes a function F and returns a function which would apply f to every element of an
    Iterable.

    Args:
    ----
        f: A function.

    Returns:
    -------
        A function which takes an Iterable and applies f to every element of the Iterable.

    """

    def apply_function(iter: Iterable[float]) -> Iterable[float]:
        return [f(x) for x in iter]

    return apply_function


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Takes a function F and returns a function which would apply f to every element of two
    Iterables.

    Args:
    ----
        f: A function.

    Returns:
    -------
        A function which takes two Iterables and applies f to every element of the Iterables.

    """

    def zip_function(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
        iterator1 = iter(iter1)
        iterator2 = iter(iter2)
        while True:
            try:
                item1 = next(iterator1)
                item2 = next(iterator2)
                yield f(item1, item2)
            except StopIteration:
                break

    return zip_function


def reduce(f: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """Takes a function f and returns a function which would apply f on all
    elements of an Iterable and reduces all of them into a single value.

    Args:
    ----
        f: A function.

    Returns:
    -------
        A function which takes an Iterable and applies f to reduce an Iterable to a single value.

    """

    def reduce_function(it: Iterable[float]) -> float:
        iterator = iter(it)
        initial = next(iterator, None)
        if initial is None:
            return 0.0
        start = initial
        for item in iterator:
            start = f(start, item)
        return start

    return reduce_function


#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def negList(it: Iterable[float]) -> Iterable[float]:
    """Negates a list of numbers

    Args:
    ----
        it: An Iterable of floats.

    Returns:
    -------
        An Iterable of floats representing the negation of it.

    """
    return map(neg)(it)


def addLists(it1: Iterable[float], it2: Iterable[float]) -> Iterable[float]:
    """Adds two lists together

    Args:
    ----
        it1: An Iterable of floats.
        it2: An Iterable of floats.

    Returns:
    -------
        An Iterable of floats representing the sum of it1 and it2.

    """
    return zipWith(add)(it1, it2)


def sum(it: Iterable[float]) -> float:
    """Sums a list of numbers

    Args:
    ----
        it: An Iterable of floats.

    Returns:
    -------
        A float representing the sum of the Iterable.

    """
    return reduce(add)(it)


def prod(it: Iterable[float]) -> float:
    """Takes the product of a list of numbers

    Args:
    ----
        it: An Iterable of floats.

    Returns:
    -------
        A float representing the product of the Iterable.

    """
    return reduce(mul)(it)
