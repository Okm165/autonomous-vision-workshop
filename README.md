# Autonomus — Vision, 3D Mapping & Autonomous Flight Workshop

A comprehensive Python workshop covering the full stack from computer vision and 3D mapping
to autonomous drone flight. 24 notebooks with deep mathematical rigor,
state-of-the-art algorithms, and hands-on implementations.

## Overview

| Block | Notebooks | Topics |
|-------|-----------|--------|
| 1. Mathematical Foundations | 01–04 | Image processing, projective geometry, camera models, Lie groups |
| 2. Motion & Correspondence | 05–07 | Feature detection, optical flow, visual odometry |
| 3. Multi-View Geometry & Depth | 08–10 | Stereo vision, neural depth, Structure from Motion |
| 4. 3D Mapping | 11–12 | Point clouds, TSDF, occupancy grids |
| 5. Full Systems | 13–15 | VIO/EKF, Visual SLAM, 3D Gaussian Splatting |
| 6. Semantic Understanding | 16 | Semantic 3D mapping |
| 7. Evaluation & Capstone | 17–18 | Benchmarking, full pipeline |
| **8. Autonomous Flight** | **19–24** | **Quadrotor dynamics, sensor fusion, path planning, RL, visual navigation** |

## Quick Start

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd autonomus
uv sync

# For GPU-accelerated notebooks (depth estimation, learned features)
uv sync --extra gpu

# Launch Jupyter
uv run jupyter notebook notebooks/
```

## Project Structure

```
autonomus/
  notebooks/           # 24 Jupyter notebooks (the workshop)
  src/                 # Reusable Python library
    imgproc.py         # Convolution, filtering, edge detection, pyramids
    projective.py      # Homographies, vanishing points, cross-ratio
    camera.py          # Pinhole model, calibration, PnP, triangulation
    transforms.py      # Rotations, quaternions, SE(3) Lie groups
    features.py        # Harris, ORB, SIFT, SuperPoint, LightGlue
    flow.py            # Lucas-Kanade, Farneback, RAFT, scene flow
    odometry.py        # Monocular + stereo visual odometry
    stereo.py          # Rectification, SGBM, disparity-to-depth
    depth.py           # Neural depth estimation (Depth Anything v2)
    sfm.py             # Incremental Structure from Motion
    pointcloud.py      # Back-projection, ICP, normal estimation
    representations.py # Voxel grids, octrees, representation comparison
    tsdf.py            # TSDF volume (Curless & Levoy 1996)
    occupancy.py       # Log-odds probabilistic occupancy grid
    ekf.py             # Extended Kalman Filter for VIO
    slam.py            # Pose graph optimization, bundle adjustment
    gaussian_splatting.py  # 3DGS forward model & rendering
    semantic.py        # Semantic segmentation + semantic TSDF
    pipeline.py        # Full mapping pipeline orchestrator
    eval.py            # ATE, RPE, depth metrics, Umeyama alignment
    viz.py             # 3D visualization helpers
    drone.py           # Quadrotor dynamics, PID & SE(3) control
    path_planning.py   # A*, RRT/RRT*, APF for 3D navigation
    rl_agents.py       # PPO, SAC reinforcement learning agents
  data/                # Sample data and download scripts
  pyproject.toml       # Dependencies (managed by uv)
```

## Mathematical Philosophy

Every derivation is shown step-by-step, not just stated. Key mathematical tools:

- **SVD**: The universal solver for Ah=0 (homography, essential matrix, triangulation)
- **Nonlinear Least Squares**: Gauss-Newton, Levenberg-Marquardt for BA and pose optimization
- **Lie Groups**: SO(3) and SE(3) with exp/log maps for pose optimization
- **Bayesian Estimation**: EKF for sensor fusion, log-odds for occupancy
- **Projective Geometry**: Homogeneous coordinates, cross-ratio, the foundation of multi-view geometry

## Prerequisites

- Python 3.10+
- Linear algebra (matrices, eigenvalues, SVD)
- Basic calculus (gradients, Taylor expansions)
- NumPy fluency

No prior computer vision experience required — the workshop builds from first principles.

## Estimated Time

~160–190 hours of focused work (equivalent to a two-semester university sequence).

| Block | Hours | Focus |
|-------|-------|-------|
| 1. Foundations (NB01–04) | 10–12 | Projective geometry, Lie groups |
| 2. Motion (NB05–07) | 12–14 | Features, flow, visual odometry |
| 3. Multi-View (NB08–10) | 14–16 | Stereo, neural depth, SfM |
| 4. Mapping (NB11–12) | 8–10 | Point clouds, TSDF, occupancy |
| 5. Systems (NB13–15) | 18–22 | VIO, SLAM, 3DGS, NeRF |
| 6. Semantic (NB16) | 4–5 | Semantic 3D mapping |
| 7. Capstone (NB17–18) | 8–10 | Evaluation, full pipeline |
| **8. Autonomous Flight (NB19–24)** | **40–50** | **Drone dynamics, control, sensor fusion, path planning, RL, visual navigation** |

## Academic Alignment

Cross-referenced against 6 university courses and 2 standard textbooks:

- **Stanford CS231A** — Camera models, epipolar geometry, SfM, NeRF, 3DGS
- **ETH Zurich 3D Vision** — Features, SfM, BA+SLAM, MVS, depth sensors
- **CMU 16-385** — Image processing, homographies, two-view geometry, stereo, flow
- **CMU 16-889 Learning for 3D Vision** — 3D representations, differentiable rendering
- **MIT 6.8300 Advances in CV** — Multi-view geometry, point tracking, scene flow
- **Tübingen Self-Driving Cars** — Stereo matching, 3D object detection, scene flow

Production systems studied: DJI, Apple ARKit, Meta Quest 3, Skydio, Tesla, Waymo, ORB-SLAM3, PX4 Autopilot.

## References

Based on 100+ seminal papers and standard textbooks:

- Hartley & Zisserman, "Multiple View Geometry in Computer Vision" (2nd ed.)
- Szeliski, "Computer Vision: Algorithms and Applications" (2nd ed.)
- Eade, "Lie Groups for 2D and 3D Transformations"
- Kerbl et al., "3D Gaussian Splatting" (SIGGRAPH 2023)
- Mildenhall et al., "NeRF" (ECCV 2020)
- Teed & Deng, "RAFT" (ECCV 2020)
- Murai et al., "MASt3R-SLAM" (CVPR 2025)
- Yang et al., "Depth Anything v2" (2024)
- Lin et al., "Depth Anything v3" (ICLR 2026)
- Forster et al., "On-Manifold Preintegration" (TRO 2017)
- Yang et al., "GNC for Robust Spatial Perception" (2020)
- Mur-Artal et al., "ORB-SLAM3" (TRO 2021)
- Wang et al., "DUSt3R" (CVPR 2024)
- Lee, Leok, McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)" (CDC 2010)
- Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors" (ICRA 2011)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Haarnoja et al., "Soft Actor-Critic" (ICML 2018)
- Karaman & Frazzoli, "Sampling-based algorithms for optimal motion planning" (IJRR 2011)
- Kaufmann et al., "Champion-level drone racing using deep reinforcement learning" (Nature 2023)
- Loquercio et al., "Learning High-Speed Flight in the Wild" (Science Robotics 2021)

See the plan file for the complete 100+ reference list grouped by topic.

## License

Educational use. See individual paper citations for algorithm references.
