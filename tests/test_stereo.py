"""Stereo geometry tests: rectification, SGBM, depth; Middlebury when data present."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src import stereo

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_pairs" / "middlebury"


class TestDepthEquation:
    def test_z_equals_fB_over_d(self):
        d = np.array([[12.0, 6.0], [3.0, 0.0]])
        z = stereo.disparity_to_depth(d, 500.0, 0.12)
        assert np.isclose(z[0, 0], 5.0)
        assert np.isclose(z[0, 1], 10.0)
        assert z[1, 0] == pytest.approx(500 * 0.12 / 3)
        assert z[1, 1] == 0.0

    def test_depth_precision_relationship(self):
        """dZ = Z^2 * dd / (f B) is the derivative of Z = fB/d."""
        f, B, dd = 500.0, 0.12, 0.5
        # numeric derivative of Z wrt d at Z=10 (d = fB/Z = 6 px)
        d0 = f * B / 10.0
        eps = 1e-4
        dZ_numeric = (f * B / d0 - f * B / (d0 + eps)) / eps * dd
        assert dZ_numeric == pytest.approx(10.0**2 * dd / (f * B), rel=1e-4)


class TestSGBMSynthetic:
    def test_recovers_shift(self):
        """Random texture with a constant positive shift -> SGBM recovers d.

        Rectified convention: left[x] = right[x - d], so the right view is
        the LEFT-shifted crop (right = img[:, d:]) and the left view drops
        the last d columns.
        """
        rng = np.random.default_rng(1)
        img = cv2.GaussianBlur(
            rng.integers(0, 256, (120, 260), dtype=np.uint8).astype(np.float32),
            (5, 5),
            1.0,
        ).astype(np.uint8)
        d_true = 16
        left = np.ascontiguousarray(img[:, :-d_true])
        right = np.ascontiguousarray(img[:, d_true:])
        disp = stereo.compute_disparity_sgbm(
            left, right, num_disparities=32, block_size=5
        )
        interior = disp[:, 40:-40]
        valid = interior > 0
        assert valid.mean() > 0.8
        assert np.median(interior[valid]) == d_true


@pytest.mark.skipif(
    not (DATA / "Backpack_left.png").is_file(), reason="Middlebury data not downloaded"
)
class TestMiddlebury:
    """Validate the stereo pipeline against the real Middlebury 2014 pair.

    The dataset ships per-scene calib.txt and disp0.pfm ground truth; the GT
    disparity uses the same x_left - x_right convention as OpenCV SGBM.
    """

    SCENE: str = "Backpack"
    f: float
    cx: float
    cy: float
    doffs: float
    baseline: float
    ndisp: int
    gt_full: NDArray[np.float64]
    left: NDArray[np.uint8]
    right: NDArray[np.uint8]

    @classmethod
    def setup_class(cls):
        import re

        calib = (DATA / f"{cls.SCENE}_calib.txt").read_text()
        m_cam = re.search(r"cam0=\[([^\]]+)\]", calib)
        assert m_cam is not None, "calib.txt missing cam0"
        row = m_cam.group(1)
        vals = [float(v) for v in row.replace(";", " ").split()]
        cls.f = vals[0]
        cls.cx, cls.cy = vals[2], vals[5]
        m_doffs = re.search(r"doffs=([\d.eE+-]+)", calib)
        assert m_doffs is not None, "calib.txt missing doffs"
        cls.doffs = float(m_doffs.group(1))
        m_baseline = re.search(r"baseline=([\d.eE+-]+)", calib)
        assert m_baseline is not None, "calib.txt missing baseline"
        cls.baseline = float(m_baseline.group(1)) / 1000.0
        m_ndisp = re.search(r"ndisp=(\d+)", calib)
        assert m_ndisp is not None, "calib.txt missing ndisp"
        cls.ndisp = int(m_ndisp.group(1))

        def read_pfm(path) -> NDArray[np.float64]:
            with open(path, "rb") as fh:
                fh.readline()
                hdr = re.match(rb"^(\d+)\s(\d+)\s$", fh.readline())
                assert hdr is not None, "bad PFM size header"
                w, h = map(int, hdr.groups())
                scale = float(fh.readline().rstrip())
                data = np.fromfile(fh, "<f4" if scale < 0 else ">f4")
            return np.flipud(np.reshape(data, (h, w))).copy()

        cls.gt_full = read_pfm(DATA / f"{cls.SCENE}_disp0.pfm")
        left = cv2.imread(str(DATA / f"{cls.SCENE}_left.png"), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(DATA / f"{cls.SCENE}_right.png"), cv2.IMREAD_GRAYSCALE)
        assert left is not None, "test image missing"
        assert right is not None, "test image missing"
        cls.left = np.ascontiguousarray(left, dtype=np.uint8)
        cls.right = np.ascontiguousarray(right, dtype=np.uint8)

    def test_disparity_accuracy(self):
        h, w = self.left.shape
        s = min(1.0, 960 / w)
        size = (int(w * s), int(h * s))
        left = cv2.resize(self.left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(self.right, size, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(self.gt_full, size, interpolation=cv2.INTER_NEAREST) * s

        disp = stereo.compute_disparity_sgbm(
            left, right, num_disparities=256, block_size=5
        ).astype(np.float64)

        valid = (gt > 0) & np.isfinite(gt)
        err = np.abs(disp[valid] - gt[valid])
        # median error must be sub-pixel; bad-pixel rate must beat a chance level
        assert np.median(err) < 1.0
        assert (err > 2.0).mean() < 0.45

    def test_depth_matches_gt_scale(self):
        h, w = self.left.shape
        s = min(1.0, 960 / w)
        size = (int(w * s), int(h * s))
        left = cv2.resize(self.left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(self.right, size, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(self.gt_full, size, interpolation=cv2.INTER_NEAREST) * s

        disp = stereo.compute_disparity_sgbm(
            left, right, num_disparities=256, block_size=5
        ).astype(np.float64)

        f = self.f * s
        z_gt = np.median(
            f * self.baseline / (gt[(gt > 0) & np.isfinite(gt)] + self.doffs * s)
        )
        v = disp > 0
        z_est = np.median(f * self.baseline / (disp[v] + self.doffs * s))
        assert z_est == pytest.approx(z_gt, rel=0.25)
