"""Partial type stubs for :mod:`mpl_toolkits.mplot3d.axes3d`.

``mpl_toolkits`` ships no ``py.typed`` marker, so type checkers fall back to
analysing its source.  The source declares numeric defaults such as
``zs=0`` and ``s=20``; inference then narrows those parameters to ``int`` even
though the documented contract is ``float or array-like`` (see the ``scatter``
docstring).  Every 3D call site in this workspace passes arrays, so the naive
inference reports hundreds of false ``reportArgumentType`` errors.

Declaring the true contract here keeps the diagnostic honest instead of
silencing it.  Only the members exercised by this workspace are declared;
``Axes3D`` inherits the remaining 2D API from ``matplotlib.axes.Axes``.
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver

__all__ = ["Axes3D"]

_ArrayLike = npt.ArrayLike
_ZDir = Literal["x", "y", "z", "-x", "-y", "-z"]

class Axes3D(Axes):
    def __init__(
        self,
        fig: Any,
        rect: Any = ...,
        *,
        elev: float = ...,
        azim: float = ...,
        roll: float = ...,
        sharez: Axes3D | None = ...,
        proj_type: Literal["persp", "ortho"] = ...,
        focal_length: float | None = ...,
        box_aspect: _ArrayLike | None = ...,
        computed_zorder: bool = ...,
        auto_add_to_figure: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    # --- 3D-only artists -------------------------------------------------
    def scatter(
        self,
        xs: _ArrayLike,
        ys: _ArrayLike,
        zs: float | _ArrayLike = ...,
        zdir: _ZDir = ...,
        s: float | _ArrayLike = ...,
        c: Any = ...,
        depthshade: bool | None = ...,
        *args: Any,
        depthshade_minalpha: float | None = ...,
        axlim_clip: bool = ...,
        data: Any = ...,
        **kwargs: Any,
    ) -> PathCollection: ...
    def plot(
        self,
        xs: _ArrayLike,
        ys: _ArrayLike,
        *args: Any,
        zdir: _ZDir = ...,
        axlim_clip: bool = ...,
        **kwargs: Any,
    ) -> list[Line2D]: ...
    # `plot3D` is a straight alias of `plot` in matplotlib's source.
    plot3D = plot
    def plot_surface(
        self,
        X: _ArrayLike,
        Y: _ArrayLike,
        Z: _ArrayLike,
        *,
        norm: str | Normalize | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        lightsource: Any = ...,
        axlim_clip: bool = ...,
        **kwargs: Any,
    ) -> Poly3DCollection: ...
    def plot_wireframe(
        self,
        X: _ArrayLike,
        Y: _ArrayLike,
        Z: _ArrayLike,
        *,
        axlim_clip: bool = ...,
        **kwargs: Any,
    ) -> Line3DCollection: ...
    def plot_trisurf(
        self,
        *args: Any,
        color: Any = ...,
        norm: str | Normalize | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        lightsource: Any = ...,
        axlim_clip: bool = ...,
        **kwargs: Any,
    ) -> Poly3DCollection: ...
    def bar3d(
        self,
        x: _ArrayLike,
        y: _ArrayLike,
        z: _ArrayLike,
        dx: _ArrayLike,
        dy: _ArrayLike,
        dz: _ArrayLike,
        color: Any = ...,
        zsort: Literal["average", "min", "max"] = ...,
        shade: bool = ...,
        lightsource: Any = ...,
        *args: Any,
        axlim_clip: bool = ...,
        data: Any = ...,
        **kwargs: Any,
    ) -> Poly3DCollection: ...
    def contour(
        self,
        X: _ArrayLike,
        Y: _ArrayLike,
        Z: _ArrayLike,
        *args: Any,
        extend3d: bool = ...,
        stride: int = ...,
        zdir: _ZDir = ...,
        offset: float | _ArrayLike | None = ...,
        axlim_clip: bool = ...,
        data: Any = ...,
        **kwargs: Any,
    ) -> Any: ...
    def contourf(
        self,
        X: _ArrayLike,
        Y: _ArrayLike,
        Z: _ArrayLike,
        *args: Any,
        zdir: _ZDir = ...,
        offset: float | _ArrayLike | None = ...,
        axlim_clip: bool = ...,
        data: Any = ...,
        **kwargs: Any,
    ) -> Any: ...
    def quiver(
        self,
        X: _ArrayLike,
        Y: _ArrayLike,
        Z: _ArrayLike,
        U: _ArrayLike,
        V: _ArrayLike,
        W: _ArrayLike,
        *,
        length: float = ...,
        arrow_length_ratio: float = ...,
        pivot: Literal["tail", "middle", "tip"] = ...,
        normalize: bool = ...,
        axlim_clip: bool = ...,
        data: Any = ...,
        **kwargs: Any,
    ) -> Quiver: ...
    def text(
        self,
        x: float,
        y: float,
        z: float,
        s: str,
        zdir: _ZDir | None = ...,
        *,
        axlim_clip: bool = ...,
        **kwargs: Any,
    ) -> Any: ...
    def add_collection3d(
        self,
        col: Any,
        zs: float | _ArrayLike = ...,
        zdir: _ZDir = ...,
    ) -> None: ...
    # --- view / limits ---------------------------------------------------
    def view_init(
        self,
        elev: float | None = ...,
        azim: float | None = ...,
        roll: float | None = ...,
        vertical_axis: Literal["x", "y", "z"] = ...,
        share: bool = ...,
    ) -> None: ...
    def set_proj_type(
        self,
        proj_type: Literal["persp", "ortho"] = ...,
        focal_length: float = ...,
    ) -> None: ...
    def set_box_aspect(
        self,
        aspect: _ArrayLike,
        *,
        zoom: float = ...,
    ) -> None: ...
    def set_zlim(
        self,
        bottom: float | None = ...,
        top: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool = ...,
        view_margin: float | None = ...,
        zmin: float | None = ...,
        zmax: float | None = ...,
    ) -> tuple[float, float]: ...
    def get_zlim(self) -> tuple[float, float]: ...
    def set_zticks(
        self,
        ticks: _ArrayLike | None,
        labels: Sequence[str] | None = ...,
        *,
        minor: bool = ...,
        **kwargs: Any,
    ) -> list[Any]: ...
    def set_zticklabels(
        self,
        labels: Sequence[str],
        *,
        minor: bool = ...,
        fontdict: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> list[Any]: ...
    def set_zlabel(
        self,
        zlabel: str,
        fontdict: dict[str, Any] | None = ...,
        labelpad: float | None = ...,
        **kwargs: Any,
    ) -> Any: ...
    def get_zlabel(self) -> str: ...
    def invert_zaxis(self) -> None: ...
    def add_contour_set(
        self,
        cset: Any,
        extend3d: bool = ...,
        zdir: _ZDir = ...,
        offset: Any = ...,
    ) -> Any: ...
    def add_contourf_set(
        self,
        cset: Any,
        zdir: _ZDir = ...,
        offset: Any = ...,
    ) -> Any: ...
    @property
    def w_xaxis(self) -> Any: ...
    @property
    def w_yaxis(self) -> Any: ...
    @property
    def w_zaxis(self) -> Any: ...
    @property
    def zaxis(self) -> Any: ...
    @property
    def zaxis_inverted(self) -> bool: ...
    @property
    def M(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]]: ...

class Line3DCollection(LineCollection):
    def do_3d_projection(self) -> float: ...
