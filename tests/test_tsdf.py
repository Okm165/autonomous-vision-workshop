"""TSDF fusion + mesh extraction tests against an analytic plane."""

import numpy as np

from src.tsdf import TSDFVolume

K = np.array([[200.0, 0.0, 64.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]])
BOUNDS = np.array([[-2.0, 2.0], [-2.0, 2.0], [0.4, 1.6]])
VOXEL = 0.02


def _fuse_plane(z_true=1.0):
    vol = TSDFVolume(BOUNDS, voxel_size=VOXEL, trunc_dist=0.06)
    vol.integrate(np.full((128, 128), z_true), K, np.eye(4))
    return vol


class TestPointcloud:
    def test_plane(self):
        vol = _fuse_plane(1.0)
        pc = np.asarray(vol.get_point_cloud())
        assert pc.shape[0] > 0
        assert np.abs(pc[:, 2] - 1.0).max() <= VOXEL + 1e-9

    def test_unobserved_empty(self):
        vol = TSDFVolume(BOUNDS, voxel_size=VOXEL)
        assert np.asarray(vol.get_point_cloud()).shape[0] == 0


class TestMesh:
    def test_plane_no_boundary_artefacts(self):
        """Mesh vertices must lie on the fused plane (no fill-value sheets)."""
        vol = _fuse_plane(1.0)
        verts, faces, _norms, _colors = vol.extract_mesh()
        assert len(verts) > 0
        assert len(faces) > 0
        assert np.abs(verts[:, 2] - 1.0).max() <= VOXEL + 1e-9

    def test_tracks_other_depths(self):
        for z_true in (0.6, 0.8, 1.2, 1.4):
            vol = TSDFVolume(BOUNDS, voxel_size=VOXEL, trunc_dist=0.06)
            vol.integrate(np.full((128, 128), z_true), K, np.eye(4))
            verts, _, _, _ = vol.extract_mesh()
            assert np.abs(verts[:, 2] - z_true).max() <= VOXEL + 1e-9, z_true

    def test_normals_unit(self):
        vol = _fuse_plane(1.0)
        _, _, norms, _ = vol.extract_mesh()
        assert np.allclose(np.linalg.norm(norms, axis=1), 1.0, atol=1e-5)

    def test_face_indices_valid(self):
        vol = _fuse_plane(1.0)
        verts, faces, _, _ = vol.extract_mesh()
        assert faces.min() >= 0
        assert faces.max() < len(verts)


class TestFusionSemantics:
    def test_idempotent_integration(self):
        vol = _fuse_plane(1.0)
        before = vol._tsdf.copy()
        vol.integrate(np.full((128, 128), 1.0), K, np.eye(4))
        assert np.allclose(vol._tsdf, before, atol=1e-12)

    def test_weights_monotonic(self):
        vol = _fuse_plane(1.0)
        w1 = vol._weight.copy()
        vol.integrate(np.full((128, 128), 1.0), K, np.eye(4))
        assert (vol._weight >= w1).all()
