"""Feature detection + epipolar geometry + optical flow tests."""

import numpy as np

from src import features as feat
from src import transforms as tr


def _synthetic_matched_points(rng, n=100):
    """Exact normalized-coordinate correspondences of a planar scene.

    3-D points X = (Z·x1, Z) with Z = 3 live in camera 1; camera 2 has pose
    (R, t) relative to camera 1, so x2 = (R X + t)_xy / (R X + t)_z.  The
    returned x1, x2 are normalized (unit-plane) image coordinates.
    """
    R = tr.so3_exp(np.array([0.05, -0.08, 0.12]))
    t = np.array([0.3, -0.1, 0.05])
    z = 3.0
    x1 = rng.uniform(-1, 1, (n, 2))
    X = np.column_stack([z * x1, np.full(n, z)])
    xc = (R @ X.T).T + t
    x2 = xc[:, :2] / xc[:, 2:3]
    return x1, x2, R, t


def _triangulate_all(x1, x2, P1, P2):
    n = len(x1)
    X = np.zeros((n, 3))
    for j in range(n):
        h1 = np.array([*x1[j], 1.0])
        h2 = np.array([*x2[j], 1.0])
        A = np.vstack(
            [
                h1[0] * P1[2] - h1[2] * P1[0],
                h1[1] * P1[2] - h1[2] * P1[1],
                h2[0] * P2[2] - h2[2] * P2[0],
                h2[1] * P2[2] - h2[2] * P2[1],
            ]
        )
        _, _, V = np.linalg.svd(A)
        X[j] = V[-1][:3] / V[-1][3]
    return X


class TestEpipolar:
    def test_epipolar_constraint(self, rng):
        x1, x2, _R, _t = _synthetic_matched_points(rng, 50)
        F = feat.compute_fundamental_8point(x1, x2)
        h1 = np.column_stack([x1, np.ones(len(x1))])
        h2 = np.column_stack([x2, np.ones(len(x2))])
        res = np.abs(np.einsum("ni,ij,nj->n", h2, F, h1))
        assert np.median(res) < 1e-6

    def test_essential_decomposition_cheirality(self, rng):
        """Of the 4 decompositions of E, exactly one puts all points in front
        of both cameras."""
        x1, x2, R, t = _synthetic_matched_points(rng, 40)
        E = tr.skew(t) @ R
        U, _, Vt = np.linalg.svd(E)
        if np.linalg.det(U) < 0:
            U = -U
        if np.linalg.det(Vt) < 0:
            Vt = -Vt
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        t_hat = U[:, 2]
        sols = [
            (U @ W @ Vt, t_hat),
            (U @ W @ Vt, -t_hat),
            (U @ W.T @ Vt, t_hat),
            (U @ W.T @ Vt, -t_hat),
        ]
        P1 = np.column_stack([np.eye(3), np.zeros(3)])
        n_good = 0
        for Ri, ti in sols:
            P2 = np.column_stack([Ri, ti])
            X = _triangulate_all(x1, x2, P1, P2)
            z1 = X[:, 2]
            z2 = (P2 @ np.column_stack([X, np.ones(len(X))]).T)[2]
            if (z1 > 0).all() and (z2 > 0).all():
                n_good += 1
        assert n_good == 1
