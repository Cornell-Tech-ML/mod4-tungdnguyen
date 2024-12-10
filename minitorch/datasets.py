import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Create a list of N random points in a unit square."""
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Create a simple dataset with 2 classes which are split
    vertically on the middle (line x = 0.5)

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points following the simple vertical split shape.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Create a dataset with 2 classes which are split
    diagonally on line x + y = 0.5.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points following the diagonal shape.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Create a dataset with 2 classes:
    - Class 0 are points between 2 horizontal lines x=0.2 and x=0.8
    - Class 1 are the remaning points.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points following the split shape.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Create a dataset with XOR properties:
    - Class 0 are points in either top right or bottom left quadrant.
    - Class 1 are points in either top left or bottom right quadrant,

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points following the XOR shape.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Create a dataset with circle properties:
    - Class 0 are points inside a circle of radius ~0.3, and center (0.5, 0.5).
    - Class 1 are points outside the circle above.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points following the circle shape.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Create a dataset with 2 spiral shapes for both classes:
    - Class 0 are points following a spiral coordinations
    - Class 1 are points following the reverse spiral coordinations of Class 0.
    Both classes have the same number of points.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points following the spiral shape.

    """

    def x(t: float) -> float:
        """Compute the x-coordinate of a point on the spiral.

        Args:
        ----
            t: The radius of the spiral in the polar coordinate system.

        Returns:
        -------
            The x-coordinate of the point on the spiral.

        """
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Compute the y-coordinate of a point on the spiral.

        Args:
        ----
            t: The radius of the spiral in the polar coordinate system.

        Returns:
        -------
            The y-coordinate of the point on the spiral.

        """
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
