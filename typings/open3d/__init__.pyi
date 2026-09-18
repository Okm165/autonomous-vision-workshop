# Local stub for `open3d`.
#
# Open3D ships neither a `py.typed` marker nor stubs, and its public API is
# assembled at import time by `open3d/__init__.py` re-exporting names from the
# compiled extension `open3d.cpu.pybind`.  A type checker therefore sees every
# `open3d.*` symbol as `Unknown`, which decays to `object` at unannotated
# boundaries: `mesh.vertices` is then reported as unknown even though the real
# `TriangleMesh` has that attribute.
#
# This stub states the real contract for the members this workspace uses, the
# same policy as `typings/mpl_toolkits` and `typings/scipy`: nothing is muted,
# so a genuinely wrong call is still reported.  See `typings/README.md`.

from open3d.cpu.pybind import (
    camera as camera,
)
from open3d.cpu.pybind import (
    core as core,
)
from open3d.cpu.pybind import (
    data as data,
)
from open3d.cpu.pybind import (
    geometry as geometry,
)
from open3d.cpu.pybind import (
    io as io,
)
from open3d.cpu.pybind import (
    pipelines as pipelines,
)
from open3d.cpu.pybind import (
    t as t,
)
from open3d.cpu.pybind import (
    utility as utility,
)

__version__: str
