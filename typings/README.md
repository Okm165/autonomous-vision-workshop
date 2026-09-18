# Local type stubs

Partial stubs that **correct upstream type information**, not work around it.
They are picked up automatically via `stubPath = "typings"` in
`[tool.basedpyright]`.

The rule for adding a stub here: a dependency publishes wrong or missing inline
types, and the stub can state the *documented, real* contract. Stating the true
contract keeps the diagnostic live at every call site, so a genuinely wrong call
is still reported. This is deliberately different from ignoring a diagnostic:
nothing is silenced.

## `mpl_toolkits.mplot3d.axes3d`

`mpl_toolkits` ships no `py.typed` marker, so type checkers fall back to
analysing its Python source. The source defaults are

```python
def scatter(self, xs, ys, zs=0, zdir='z', s=20, ...)
```

Type inference therefore narrows `zs` and `s` to `int`, while the documented
contract is `float or array-like` (see the `Axes3D.scatter` docstring). Every 3D
plot in this workspace passes arrays, so the inferred signature produces
hundreds of false `reportArgumentType` errors.

`mplot3d/axes3d.pyi` declares the real contract for the members this workspace
uses. `Axes3D` inherits the remaining 2D API from `matplotlib.axes.Axes`, which
*is* typed, so only the 3D-specific surface needs restating.

## `scipy.interpolate`

SciPy ships no stubs, so type checkers analyse its sources. `interp1d.__init__`
leaves `fill_value` unannotated:

```python
def __init__(self, x, y, kind='linear', axis=-1,
             copy=True, bounds_error=None, fill_value=np.nan,
             assume_sorted=False):
```

`fill_value` is therefore inferred as `float` from the `np.nan` default, while
the documented contract is `array-like or (array-like, array-like) or
"extrapolate"`. `mplot3d`-style stubbing applies: `interpolate/__init__.pyi`
restates the real contract.

## `open3d`

Open3D ships neither a `py.typed` marker nor stubs, and it assembles its public
API at import time by re-exporting names from the compiled extension
`open3d.cpu.pybind`. A type checker therefore sees every `open3d.*` symbol as
`Unknown`, which decays to `object` at unannotated boundaries — `mesh.vertices`
is reported as unknown even though `TriangleMesh` really does have it.

The stub restates the members this workspace uses, following the same policy as
the others: nothing is muted, so a genuinely wrong call is still reported.

## Maintenance

These stubs track the installed versions of their packages. If a stub drifts
from reality, the symptom is a spurious diagnostic at a correct call site — fix
the stub rather than annotating around it. Re-verify after any dependency
upgrade with

```sh
.venv/bin/basedpyright
```

A directory stub under `typings/` fully **replaces** the package's own inline
types, so each subtree needs an `__init__.pyi` at every level; without one,
basedpyright silently ignores the whole directory. This is why `cv2` is *not*
stubbed here: a partial `cv2` stub would hide every symbol it does not declare.
OpenCV's mismatches are instead handled at the call sites by the owned facade
`src/_cv.py`.