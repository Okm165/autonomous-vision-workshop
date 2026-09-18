"""Partial type stubs for :mod:`scipy.interpolate`.

SciPy ships no type stubs, so type checkers analyse the Python sources.  Several
constructors leave ``fill_value`` unannotated, e.g. ::

    def __init__(self, x, y, kind='linear', axis=-1,
                 copy=True, bounds_error=None, fill_value=np.nan,
                 assume_sorted=False):

Inference therefore narrows ``fill_value`` to ``float`` from the ``np.nan``
default, even though the documented contract is ``array-like or
(array-like, array-like) or "extrapolate"`` (see the ``interp1d`` docstring in
``scipy/interpolate/_interpolate.py``).  Passing the literal ``"extrapolate"``
is the canonical way to request extrapolation and is what this workspace uses.

Declaring the real contract here keeps the diagnostic honest instead of
silencing it.
"""

from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt

__all__ = ["interp1d"]

_ArrayLike = npt.ArrayLike
_FillValue = float | _ArrayLike | tuple[_ArrayLike, _ArrayLike] | Literal["extrapolate"]

class interp1d:
    @overload
    def __init__(
        self,
        x: _ArrayLike,
        y: _ArrayLike,
        kind: Literal[
            "linear",
            "nearest",
            "nearest-up",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "previous",
            "next",
        ] = ...,
        axis: int = ...,
        copy: bool = ...,
        bounds_error: bool | None = ...,
        fill_value: _FillValue = ...,
        assume_sorted: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        x: _ArrayLike,
        y: _ArrayLike,
        kind: int,
        axis: int = ...,
        copy: bool = ...,
        bounds_error: bool | None = ...,
        fill_value: _FillValue = ...,
        assume_sorted: bool = ...,
    ) -> None: ...
    def __call__(self, x: _ArrayLike, /) -> np.ndarray[Any, np.dtype[np.float64]]: ...
