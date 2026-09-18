# Stub for `open3d.cpu.pybind.utility`.
#
# Open3D's pybind converters.  `Vector3dVector` / `Vector3iVector` are the
# attribute containers used by `geometry.PointCloud` and `geometry.TriangleMesh`;
# they are the *same* classes imported there, so this module re-exports the
# definitions rather than duplicating them.

from open3d.cpu.pybind.geometry import (
    Vector3dVector as Vector3dVector,
)
from open3d.cpu.pybind.geometry import (
    Vector3iVector as Vector3iVector,
)

class Vector2dVector:
    def __len__(self) -> int: ...

class Vector4iVector:
    def __len__(self) -> int: ...
