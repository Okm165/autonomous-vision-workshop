# Stub for `open3d.cpu.pybind`, the compiled pybind11 extension that holds the
# real Open3D API.  Without it, every symbol below is `Unknown` to a type
# checker and decays to `object`, hiding real attribute errors (e.g.
# `mesh.vertices`) behind false ones.  See `typings/open3d/__init__.pyi`.

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
