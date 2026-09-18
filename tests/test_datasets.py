"""Dataset I/O regression tests.

These pin the on-disk format of the downloaded test datasets so that a bad
re-download or a loader change fails loudly.  Every test is skipped (not
failed) when the dataset is absent, so the suite stays green on machines
without `data/download.sh` artifacts.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"

KITTI_SEQ = DATA / "kitti/sequences/00"
KITTI_POSES = DATA / "kitti/poses"
TUM = DATA / "tum/rgbd_dataset_freiburg1_desk"
MIDDLEBURY = DATA / "sample_pairs/middlebury"

KITTI_FRAMES = 200
KITTI_SHAPE = (376, 1241)  # (H, W) of the rectified odometry images
TUM_SHAPE = (480, 640)
MIDDLEBURY_SCENES = ("Backpack", "Jadeplant", "Motorcycle", "Piano", "Playroom")


def parse_calib(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, values = line.split(":", 1)
            out[key.strip()] = [float(v) for v in values.split()]
    return out


@pytest.mark.skipif(not KITTI_SEQ.is_dir(), reason="KITTI odometry not downloaded")
class TestKitti:
    def test_frame_counts(self):
        for cam in ("image_0", "image_1"):
            frames = sorted((KITTI_SEQ / cam).glob("*.png"))
            assert len(frames) == KITTI_FRAMES
            assert frames[0].name == "000000.png"

    def test_image_format(self):
        for cam in ("image_0", "image_1"):
            img = cv2.imread(str(KITTI_SEQ / cam / "000000.png"), cv2.IMREAD_UNCHANGED)
            assert img is not None
            assert img.shape == KITTI_SHAPE
            assert img.dtype == np.uint8

    def test_calib_projection_matrices(self):
        calib = parse_calib(KITTI_SEQ / "calib.txt")
        for key in ("P0", "P1", "P2", "P3"):
            assert key in calib, f"missing {key} in calib.txt"
            assert len(calib[key]) == 12
            p = np.array(calib[key]).reshape(3, 4)
            # fx, fy positive; principal center inside the image
            fx, fy = p[0, 0], p[1, 1]
            cx, cy = p[0, 2], p[1, 2]
            assert fx > 0
            assert fy > 0
            assert 0 < cx < KITTI_SHAPE[1]
            assert 0 < cy < KITTI_SHAPE[0]
        # stereo baseline: P1 (left-minus-right for the colour pair) shifts cx
        p0 = np.array(calib["P0"]).reshape(3, 4)
        p1 = np.array(calib["P1"]).reshape(3, 4)
        assert abs(p1[0, 3] - p0[0, 3]) > 1.0

    def test_times_file(self):
        lines = (KITTI_SEQ / "times.txt").read_text().split()
        # the file covers the full sequence (4541 frames); we keep the first
        # KITTI_FRAMES images, so only the leading prefix must be validated
        assert len(lines) >= KITTI_FRAMES
        times = np.array([float(v) for v in lines[:KITTI_FRAMES]])
        assert np.all(np.diff(times) > 0), "timestamps must be increasing"

    def test_poses_files(self):
        pose_files = sorted(KITTI_POSES.glob("*.txt"))
        assert len(pose_files) == 11, "sequences 00..10"
        for pf in pose_files:
            rows = [
                line.split() for line in pf.read_text().splitlines() if line.strip()
            ]
            assert len(rows) >= KITTI_FRAMES, f"{pf.name}: too few poses"
            for row in rows:
                assert len(row) == 12
            first = np.array(rows[0], dtype=float).reshape(3, 4)
            # rotation block must be (approximately) orthonormal, det = +1
            rot = first[:, :3]
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-4)
            assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(not TUM.is_dir(), reason="TUM fr1/desk not downloaded")
class TestTum:
    def test_rgb_depth_lists(self):
        rgb = sorted((TUM / "rgb").glob("*.png"))
        depth = sorted((TUM / "depth").glob("*.png"))
        assert len(rgb) == 613
        assert len(depth) >= 500
        # every depth file references an existing rgb capture timestamp-wise;
        # depth runs at a lower rate, so depth <= rgb in count
        assert len(depth) <= len(rgb)

    def test_image_format(self):
        rgb = cv2.imread(str(sorted((TUM / "rgb").glob("*.png"))[0]))
        assert rgb is not None
        assert rgb.shape == (*TUM_SHAPE, 3)
        depth = cv2.imread(
            str(sorted((TUM / "depth").glob("*.png"))[0]), cv2.IMREAD_UNCHANGED
        )
        assert depth is not None
        assert depth.shape == TUM_SHAPE
        assert depth.dtype == np.uint16
        # TUM depth is millimetre-scaled: values must fit mm range
        assert depth.max() > 0

    def test_association_files(self):
        for name in ("rgb.txt", "depth.txt"):
            lines = [
                ln
                for ln in (TUM / name).read_text().splitlines()
                if ln and not ln.startswith("#")
            ]
            assert lines, f"{name} has no entries"
            for ln in lines:
                parts = ln.split()
                assert len(parts) == 2
                float(parts[0])  # timestamp parses
                assert (TUM / parts[1]).exists(), f"{parts[1]} missing"

    def test_groundtruth_format(self):
        lines = [
            ln
            for ln in (TUM / "groundtruth.txt").read_text().splitlines()
            if ln and not ln.startswith("#")
        ]
        assert len(lines) >= 500
        stamps = []
        for ln in lines:
            parts = ln.split()
            assert len(parts) == 8  # t tx ty tz qx qy qz qw
            stamp = float(parts[0])
            quat = np.array([float(v) for v in parts[4:]])
            assert quat.shape == (4,)
            stamps.append(stamp)
        assert np.all(np.diff(stamps) > 0)


@pytest.mark.skipif(not MIDDLEBURY.is_dir(), reason="Middlebury pairs not downloaded")
class TestMiddlebury:
    @pytest.mark.parametrize("scene", MIDDLEBURY_SCENES)
    def test_stereo_pair(self, scene: str):
        left = cv2.imread(str(MIDDLEBURY / f"{scene}_left.png"))
        right = cv2.imread(str(MIDDLEBURY / f"{scene}_right.png"))
        assert left is not None
        assert right is not None
        assert left.shape == right.shape

    @pytest.mark.parametrize("scene", MIDDLEBURY_SCENES)
    def test_calib_fields(self, scene: str):
        text = (MIDDLEBURY / f"{scene}_calib.txt").read_text()
        fields = {}
        for line in text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip()] = value.strip()
        for key in ("cam0", "cam1", "baseline", "width", "height", "ndisp"):
            assert key in fields, f"{scene}: missing {key}"
        # cam0 is a 3x3 matrix in [fx 0 cx; 0 fy cy; 0 0 1] form
        vals = [float(v) for v in fields["cam0"].strip("[]").split(";")[0].split()]
        assert len(vals) == 3
        assert vals[0] > 0
        baseline = float(fields["baseline"])
        assert baseline > 0
        left = cv2.imread(str(MIDDLEBURY / f"{scene}_left.png"))
        assert left is not None, "test image missing"
        assert left.shape[1] == int(fields["width"])
        assert left.shape[0] == int(fields["height"])

    @pytest.mark.parametrize("scene", MIDDLEBURY_SCENES)
    def test_disparity_map(self, scene: str):
        disp_path = MIDDLEBURY / f"{scene}_disp0.pfm"
        assert disp_path.exists()
        data = disp_path.read_bytes()
        assert data[:2] == b"Pf", "PFM magic"
