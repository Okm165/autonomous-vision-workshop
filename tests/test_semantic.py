"""Tests for src/semantic.py — semantic segmentation and semantic TSDF fusion."""

import numpy as np
import pytest

from src.semantic import SemanticSegmenter, SemanticTSDF, semantic_pointcloud

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(scope="module")
def seg():
    return SemanticSegmenter(model_name="simple")


def _street_scene() -> np.ndarray:
    """Synthetic BGR street scene: sky / vegetation / road / red car stripes."""
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:40] = [255, 255, 255]  # white sky   -> HSV s=0, v=255 (upper half)
    img[40:80] = [0, 255, 0]  # green vegetation -> hue 60, s=255
    img[80:] = [128, 128, 128]  # gray road   -> s=0, v=128 (lower half)
    img[85:95, 20:60] = [0, 0, 255]  # bright red -> "car" slot
    return img


# ======================================================================
# SemanticSegmenter (simple HSV backend — no torch required)
# ======================================================================


class TestSemanticSegmenter:
    def test_labels_shape_dtype(self, seg):
        labels, overlay = seg.segment(_street_scene())
        assert labels.shape == (120, 160)
        assert labels.dtype == np.int32
        assert overlay.shape == (120, 160, 3)
        assert overlay.dtype == np.uint8

    def test_sky_is_background_in_upper_half(self, seg):
        labels, _ = seg.segment(_street_scene())
        assert set(np.unique(labels[:30, :])) == {0}

    def test_vegetation_detected_by_green_hue(self, seg):
        labels, _ = seg.segment(_street_scene())
        region = labels[50:70, 70:150]  # pure green block
        assert np.all(region == 10)

    def test_road_is_ground_class_in_lower_half(self, seg):
        labels, _ = seg.segment(_street_scene())
        region = labels[100:, 80:]  # gray block, lower half
        assert np.all(region == 9)

    def test_bright_red_maps_to_vehicle(self, seg):
        labels, _ = seg.segment(_street_scene())
        assert np.all(labels[85:95, 20:60] == 7)

    def test_overlay_uses_palette(self, seg):
        labels, overlay = seg.segment(_street_scene())
        veg = labels == 10
        assert veg.any()
        assert np.all(overlay[veg] == seg._palette[10])

    def test_segment_everything_and_prompt_fall_back_to_segment(self, seg):
        img = _street_scene()
        base_labels, base_overlay = seg.segment(img)
        lb, ob = seg.segment_everything(img)
        lp, op = seg.segment_with_prompt(img, points=np.array([[80.0, 60.0]]))
        assert np.array_equal(lb, base_labels)
        assert np.array_equal(ob, base_overlay)
        assert np.array_equal(lp, base_labels)
        assert np.array_equal(op, base_overlay)

    def test_track_masks_returns_one_map_per_frame(self, seg):
        img = _street_scene()
        labels0, _ = seg.segment(img)
        tracked = seg.track_masks([img, img, img], labels0)
        assert len(tracked) == 3
        assert np.array_equal(tracked[0], labels0)
        assert all(np.array_equal(t, labels0) for t in tracked[1:])

    def test_unknown_model_raises_on_segment(self):
        bad = SemanticSegmenter(model_name="invalid")
        with pytest.raises(ValueError, match="Unknown model"):
            bad.segment(_street_scene())

    def test_class_names(self, seg):
        names = seg.class_names
        assert len(names) == 21
        assert names[0] == "background"
        assert names[7] == "car"

    def test_custom_num_classes_truncates_palette(self):
        seg5 = SemanticSegmenter(model_name="simple", num_classes=5)
        labels, overlay = seg5.segment(_street_scene())
        # labels beyond the palette are left black in the overlay
        veg = labels == 10
        assert np.all(overlay[veg] == 0)
        assert len(seg5.class_names) == 5


# ======================================================================
# semantic_pointcloud
# ======================================================================


class TestSemanticPointcloud:
    def test_groups_points_by_class(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=np.float64)
        colors = np.zeros((4, 3))
        labels = np.array([0, 0, 7, 7], dtype=np.int32)
        out = semantic_pointcloud(pts, colors, labels)
        assert set(out.keys()) == {"background", "car"}
        assert out["background"].shape == (2, 6)
        assert out["car"].shape == (2, 6)
        assert np.allclose(out["car"][:, :3], [[0, 1, 0], [5, 5, 5]])

    def test_unknown_label_uses_fallback_name(self):
        pts = np.array([[1, 2, 3]], dtype=np.float64)
        out = semantic_pointcloud(pts, np.zeros((1, 3)), np.array([25]))
        assert "class_25" in out

    def test_custom_class_names(self):
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        out = semantic_pointcloud(
            pts, np.zeros((2, 3)), np.array([1, 0]), class_names=["a", "b"]
        )
        assert set(out.keys()) == {"a", "b"}

    def test_preserves_point_colors(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (6, 3))
        colors = rng.uniform(0, 1, (6, 3))
        labels = np.zeros(6, dtype=np.int32)
        out = semantic_pointcloud(pts, colors, labels)
        assert np.allclose(out["background"][:, 3:], colors)


# ======================================================================
# SemanticTSDF
# ======================================================================

# Volume: 10x10x10 voxels of 5 cm around a wall at z = 2 m.
# Voxel centres sit at origin + (k + 0.5)·voxel ∈ [-0.225, 0.225] (x, y) and
# [1.775, 2.225] (z).  With the camera at the origin looking down +z, depth
# 2.0 m everywhere, and trunc_dist 0.15, exactly the 8 near z-layers observe
# the wall (sdf >= -trunc fails for z >= 2.15); the 2 far layers stay at the
# uniform prior.
BOUNDS = np.array([[-0.25, 0.25], [-0.25, 0.25], [1.75, 2.25]])
VOXEL = 0.05
N_OBSERVED_LAYERS = 8
K_TSDF = np.array(
    [
        [100.0, 0.0, 64.0],
        [0.0, 100.0, 64.0],
        [0.0, 0.0, 1.0],
    ]
)


def _wall_labels(h=128, w=128, left=3, right=7):
    """Split label image: left half `left`, right half `right`."""
    labels = np.full((h, w), right, dtype=np.int32)
    labels[:, : w // 2] = left
    return labels


class TestSemanticTSDFInit:
    def test_uniform_prior(self):
        vol = SemanticTSDF(BOUNDS, voxel_size=VOXEL, num_classes=21)
        probs = vol.get_class_probabilities()
        assert probs.shape == (10, 10, 10, 21)
        assert np.allclose(probs, 1.0 / 21, atol=1e-6)
        assert np.all(vol.get_voxel_labels() == 0)

    def test_voxel_centers_span_bounds(self):
        vol = SemanticTSDF(BOUNDS, voxel_size=VOXEL, num_classes=4)
        coords = vol._voxel_coords
        assert coords.shape == (1000, 3)
        for ax in range(3):
            lo, hi = BOUNDS[ax]
            assert coords[:, ax].min() >= lo
            assert coords[:, ax].max() <= hi

    def test_empty_mesh_when_unobserved(self):
        vol = SemanticTSDF(BOUNDS, voxel_size=VOXEL)
        verts, faces, normals, vlabels = vol.get_semantic_mesh()
        assert len(verts) == 0
        assert len(faces) == 0
        assert len(normals) == 0
        assert len(vlabels) == 0


class TestSemanticTSDFIntegration:
    def _integrate_wall(self, confidences=None):
        vol = SemanticTSDF(BOUNDS, voxel_size=VOXEL, num_classes=21)
        depth = np.full((128, 128), 2.0)
        labels = _wall_labels()
        vol.integrate(depth, K_TSDF, np.eye(4), labels, confidences=confidences)
        return vol

    def _voxel_index(self, coords):
        """Map world coords to integer voxel indices."""
        return np.clip(np.floor((coords - BOUNDS[:, 0]) / VOXEL).astype(int), 0, 9)

    def test_observed_voxels_take_their_image_label(self):
        vol = self._integrate_wall()
        coords = vol._voxel_coords
        labels_img = _wall_labels()
        # contract: each observed voxel fuses the label of the pixel it
        # projects to (camera at origin, looking down +z)
        pix_x = np.round(
            K_TSDF[0, 0] * coords[:, 0] / coords[:, 2] + K_TSDF[0, 2]
        ).astype(int)
        pix_y = np.round(
            K_TSDF[1, 1] * coords[:, 1] / coords[:, 2] + K_TSDF[1, 2]
        ).astype(int)
        observed = vol._weight > 0
        # voxels with sdf < -trunc (the two far z-layers) stay unobserved
        assert observed.sum() == 100 * N_OBSERVED_LAYERS
        idx = self._voxel_index(coords)
        expected = labels_img[pix_y, pix_x]
        assert np.array_equal(
            vol.get_voxel_labels()[observed], expected[observed.ravel()]
        )
        # sanity: the projection indeed hits the split boundary correctly
        assert expected[coords[:, 0] < 0].min() == 3
        assert expected[coords[:, 0] > 0].max() == 7
        assert np.all(idx >= 0)
        assert np.all(idx <= 9)

    def test_left_right_split_fusion(self):
        vol = self._integrate_wall()
        got = vol.get_voxel_labels()
        observed = vol._weight > 0
        # x is volume axis 0 (meshgrid with indexing='ij')
        x_centers = BOUNDS[0, 0] + (np.arange(10) + 0.5) * VOXEL
        left = observed & (x_centers < 0.0)[:, None, None]
        right = observed & (x_centers >= 0.0)[:, None, None]
        assert left.any()
        assert right.any()
        assert np.all(got[left] == 3)
        assert np.all(got[right] == 7)

    def test_probability_mass_concentrates_on_observed_class(self):
        vol = self._integrate_wall()
        probs = vol.get_class_probabilities()
        observed = vol._weight > 0
        # every observed voxel was seen once with conf 0.8:
        # p(observed class) must exceed the uniform value 1/21
        assert probs[observed].max(axis=-1).min() > 1.0 / 21
        # unobserved voxels keep the uniform prior
        assert np.allclose(probs[~observed], 1.0 / 21, atol=1e-6)

    def test_repeated_observations_sharpen_belief(self):
        vol = SemanticTSDF(BOUNDS, voxel_size=VOXEL, num_classes=21)
        depth = np.full((128, 128), 2.0)
        labels = _wall_labels()
        vol.integrate(depth, K_TSDF, np.eye(4), labels)
        p1 = vol.get_class_probabilities()
        vol.integrate(depth, K_TSDF, np.eye(4), labels)
        p2 = vol.get_class_probabilities()
        # second observation increases confidence in the same class
        mask = vol._weight > 0
        assert p2.max(axis=-1)[mask].min() > p1.max(axis=-1)[mask].min()

    def test_higher_confidence_sharpens_more(self):
        low = np.full((128, 128), 0.6, dtype=np.float32)
        high = np.full((128, 128), 0.95, dtype=np.float32)
        p_low = self._integrate_wall(low).get_class_probabilities()
        p_high = self._integrate_wall(high).get_class_probabilities()
        mask = self._integrate_wall()._weight > 0
        assert p_high.max(axis=-1)[mask].min() > p_low.max(axis=-1)[mask].min()

    def test_tsdf_and_weight_updated(self):
        vol = self._integrate_wall()
        # voxels beyond sdf < -trunc (z >= 2.15) remain unobserved
        observed = vol._weight > 0
        assert observed.sum() == 100 * N_OBSERVED_LAYERS
        assert not observed[:, :, -2:].any()
        # wall at z=2 -> zero crossing inside the volume
        assert vol._tsdf.min() < 0.0 < vol._tsdf.max()
        assert np.all(np.abs(vol._tsdf[observed]) <= 1.0 + 1e-6)

    def test_mesh_extraction_after_integration(self):
        vol = self._integrate_wall()
        verts, faces, normals, vlabels = vol.get_semantic_mesh()
        assert len(verts) > 0
        assert len(faces) > 0
        assert len(vlabels) == len(verts)
        assert len(normals) == len(verts)
        # surface sits at the observed wall depth
        assert np.abs(verts[:, 2] - 2.0).max() < 3 * VOXEL
        # every vertex label is one of the two observed classes
        assert set(np.unique(vlabels)) <= {3, 7}

    def test_zero_depth_pixels_are_ignored(self):
        vol = SemanticTSDF(BOUNDS, voxel_size=VOXEL, num_classes=21)
        depth = np.zeros((128, 128))
        labels = _wall_labels()
        vol.integrate(depth, K_TSDF, np.eye(4), labels)
        assert np.all(vol._weight == 0)
        verts, *_ = vol.get_semantic_mesh()
        assert len(verts) == 0
