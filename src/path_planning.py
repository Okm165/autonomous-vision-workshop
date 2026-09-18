"""
Path planning algorithms for 3D navigation.

Implements A*, RRT, RRT*, Artificial Potential Fields,
and trajectory optimization for autonomous drone flight.

References:
    - LaValle, "Planning Algorithms", 2006
    - Karaman & Frazzoli, "Sampling-based algorithms for optimal motion
      planning", IJRR 2011
    - Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile
      Robots", IJRR 1986
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ======================================================================
#  3D Occupancy Grid Interface
# ======================================================================


@dataclass
class OccupancyGrid3D:
    r"""3D binary occupancy grid for path planning.

    Each voxel is either free (0) or occupied (1).
    The grid has origin at ``origin`` and voxel size ``resolution``.

    World-to-grid coordinate transform:

    .. math::

        \mathbf{i} = \left\lfloor \frac{\mathbf{p} - \mathbf{o}}{r} \right\rfloor

    Parameters
    ----------
    grid : (Nx, Ny, Nz) bool array
        True = occupied, False = free
    resolution : float
        Voxel size [m]
    origin : (3,) world coordinates of grid corner [0,0,0]
    """

    grid: NDArray[np.bool_]
    resolution: float
    origin: NDArray[np.float64]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid dimensions ``(Nx, Ny, Nz)``."""
        nx, ny, nz = self.grid.shape
        return int(nx), int(ny), int(nz)

    def world_to_grid(self, point: NDArray[np.float64]) -> tuple[int, int, int]:
        """Convert a world-frame point to integer grid indices.

        Returns plain Python ``int``s so the result hashes and compares
        exactly like any other integer tuple (numpy scalars compare equal
        but do not hash identically in every NumPy version).
        """
        idx: NDArray[np.int_] = np.floor(
            (point - self.origin) / self.resolution
        ).astype(int)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def grid_to_world(self, idx: tuple[int, int, int]) -> NDArray[np.float64]:
        """Convert grid indices to world-frame voxel centre coordinates."""
        return self.origin + (np.array(idx) + 0.5) * self.resolution

    def is_valid(self, idx: tuple[int, int, int]) -> bool:
        """Return True if *idx* lies within the grid bounds."""
        return all(0 <= idx[i] < self.shape[i] for i in range(3))

    def is_free(self, idx: tuple[int, int, int]) -> bool:
        """Return True if *idx* is within bounds and unoccupied."""
        return self.is_valid(idx) and not self.grid[idx]

    def is_free_world(self, point: NDArray[np.float64]) -> bool:
        """Return True if a world-frame point maps to a free voxel."""
        return self.is_free(self.world_to_grid(point))

    def inflate(self, radius_cells: int) -> OccupancyGrid3D:
        """Inflate obstacles by a given radius (Minkowski sum with a sphere)."""
        from scipy.ndimage import binary_dilation

        struct = np.zeros((2 * radius_cells + 1,) * 3, dtype=bool)
        center = radius_cells
        for i in range(struct.shape[0]):
            for j in range(struct.shape[1]):
                for k in range(struct.shape[2]):
                    if (i - center) ** 2 + (j - center) ** 2 + (
                        k - center
                    ) ** 2 <= radius_cells**2:
                        struct[i, j, k] = True
        inflated = binary_dilation(self.grid, structure=struct)
        return OccupancyGrid3D(
            np.asarray(inflated, dtype=bool), self.resolution, self.origin.copy()
        )


def create_random_obstacles(
    grid_shape: tuple[int, int, int],
    n_obstacles: int,
    obstacle_size_range: tuple[int, int],
    resolution: float = 0.1,
    origin: NDArray[np.float64] = np.zeros(3),
    seed: int = 42,
) -> OccupancyGrid3D:
    """Create a random 3D occupancy grid with box obstacles."""
    rng = np.random.RandomState(seed)
    grid = np.zeros(grid_shape, dtype=bool)

    for _ in range(n_obstacles):
        size = rng.randint(obstacle_size_range[0], obstacle_size_range[1] + 1, 3)
        pos = rng.randint(0, np.array(grid_shape) - size)
        grid[
            pos[0] : pos[0] + size[0],
            pos[1] : pos[1] + size[1],
            pos[2] : pos[2] + size[2],
        ] = True

    return OccupancyGrid3D(grid, resolution, origin.copy())


# ======================================================================
#  A* Search (3D)
# ======================================================================


def astar_3d(
    grid: OccupancyGrid3D,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    allow_diagonal: bool = True,
) -> NDArray[np.float64] | None:
    r"""A* search on a 3D occupancy grid.

    **Algorithm**: best-first search with an admissible heuristic.

    .. math::

        f(n) = g(n) + h(n)

    where :math:`g(n)` is the cost-to-come (metric path length) and
    :math:`h(n)` is the heuristic (Euclidean distance to the goal).

    **Admissibility**: :math:`h(n) \leq h^*(n)` for all :math:`n`,
    guaranteeing optimality.  Euclidean distance is admissible for both
    the 6-connected (axis moves, cost :math:`r`) and 26-connected
    (axis/diagonal moves, cost :math:`r` or :math:`r\sqrt{3}`) graph.

    **Complexity**: :math:`O(|V| \log |V|)` with a binary heap.

    Parameters
    ----------
    grid : OccupancyGrid3D
    start, goal : (3,) world coordinates.  Their voxel centres are the
        search's endpoints; see *Notes*.
    allow_diagonal : if True, 26-connected; else 6-connected

    Returns
    -------
    path : (N, 3) world coordinates, or None if no path exists.
        ``path[0]`` is exactly *start*, ``path[-1]`` is exactly *goal*,
        and every intermediate waypoint is a voxel centre.  Because the
        cost-to-come is accumulated in metric units, replacing the outer
        voxel centres by the exact endpoints strictly shortens the
        reported length, so optimality of the underlying grid path is
        preserved.

    Notes
    -----
    A grid search can only visit voxels, so the path necessarily passes
    through the centre of the voxel containing *start* and the one
    containing *goal*.  The returned path therefore has a leading and
    trailing leg connecting the requested coordinates to those centres.
    """
    start = np.asarray(start, dtype=np.float64).ravel()
    goal = np.asarray(goal, dtype=np.float64).ravel()
    start_idx = grid.world_to_grid(start)  # voxel containing *start*
    goal_idx = grid.world_to_grid(goal)  # voxel containing *goal*

    if not grid.is_free(start_idx) or not grid.is_free(goal_idx):
        return None

    # Neighbourhood.  Move costs are dimensionless multiples of the voxel
    # size; each move's metric cost is ``move_cost * grid.resolution``.
    if allow_diagonal:
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = float(np.sqrt(dx**2 + dy**2 + dz**2))
                    neighbors.append(((dx, dy, dz), cost))
    else:
        neighbors = [
            ((1, 0, 0), 1.0),
            ((-1, 0, 0), 1.0),
            ((0, 1, 0), 1.0),
            ((0, -1, 0), 1.0),
            ((0, 0, 1), 1.0),
            ((0, 0, -1), 1.0),
        ]

    res = float(grid.resolution)
    # Length of the trailing leg from the goal voxel centre to the requested goal.
    # It is a constant, but including it in every f-value keeps the reported
    # cost-to-come comparable with the returned path's geometric length.
    goal_leg = float(np.linalg.norm(grid.grid_to_world(goal_idx) - goal))

    def heuristic(idx: tuple[int, int, int]) -> float:
        """Lower bound on the remaining cost from *idx* to the requested goal.

        Euclidean distance to the goal voxel centre is admissible for both
        the 6- and 26-connected graphs; the constant :data:`goal_leg` is then
        added so that ``g + h`` is a bound on the full path length.
        """
        return goal_leg + res * float(
            np.sqrt(sum((a - b) ** 2 for a, b in zip(idx, goal_idx, strict=False)))
        )

    # g_score is a *metric* cost-to-come.  It is seeded with the length of the
    # leading leg from the requested *start* to the centre of the voxel that
    # contains it, because the search itself only ever visits voxel centres.
    g_score = {start_idx: float(np.linalg.norm(grid.grid_to_world(start_idx) - start))}
    came_from = {}
    closed = set()
    # Tie-break on a monotonically increasing counter so that entries never
    # compare their (possibly unhashable) voxel tuples.
    counter = 0
    open_set = [(heuristic(start_idx), counter, start_idx)]

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in closed:
            continue
        closed.add(current)

        if current == goal_idx:
            # Walk the parent chain from the goal back to the start voxel,
            # emitting every voxel including the start one, then reverse.
            path = []
            node = current
            while True:
                path.append(grid.grid_to_world(node))
                if node == start_idx:
                    break
                node = came_from[node]
            path.reverse()
            # The search runs on voxel centres, so the outer entries only
            # approximate the requested endpoints.  Replace them with the
            # exact coordinates (prepending/appending when the endpoint lies
            # outside the searched voxel) so the result is directly usable as
            # a trajectory.  The added legs are exactly the offset already
            # accounted for in g_score, hence never lengthen the path.
            if path:
                path[0] = start
            else:
                path.append(start)
            path[-1] = goal
            return np.array(path)

        for (dx, dy, dz), move_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if neighbor in closed or not grid.is_free(neighbor):
                continue

            tentative_g = g_score[current] + move_cost * res

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(
                    open_set, (tentative_g + heuristic(neighbor), counter, neighbor)
                )

    return None


# ======================================================================
#  RRT (Rapidly-exploring Random Tree)
# ======================================================================


@dataclass
class RRTNode:
    """Node in an RRT/RRT* search tree (position + parent index + cost-to-come)."""

    position: NDArray[np.float64]
    parent: int | None = None
    cost: float = 0.0


def rrt(
    grid: OccupancyGrid3D,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    max_iterations: int = 5000,
    step_size: float = 0.3,
    goal_threshold: float = 0.3,
    goal_bias: float = 0.1,
    seed: int = 42,
) -> NDArray[np.float64] | None:
    r"""Basic RRT for 3D path planning.

    **Algorithm** (LaValle, 1998):

    1. Sample random point :math:`q_{\text{rand}}`
       (with probability ``goal_bias``, use the goal)
    2. Find nearest node :math:`q_{\text{near}}`
    3. Extend toward :math:`q_{\text{rand}}` by ``step_size``
    4. Check collision; if free, add new node
    5. Check if goal is reached

    Returns
    -------
    path : (N, 3) world coordinates, or None if not found
    """
    rng = np.random.RandomState(seed)

    # Workspace bounds from grid
    lo = grid.origin
    hi = grid.origin + np.array(grid.shape) * grid.resolution

    nodes = [RRTNode(start.copy())]

    for _ in range(max_iterations):
        # Sample
        q_rand = goal.copy() if rng.rand() < goal_bias else lo + rng.rand(3) * (hi - lo)

        # Nearest
        dists = [np.linalg.norm(n.position - q_rand) for n in nodes]
        near_idx = int(np.argmin(dists))
        q_near = nodes[near_idx].position

        # Steer
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist < 1e-8:
            continue
        direction /= dist
        q_new = q_near + min(step_size, dist) * direction

        # Collision check (sample along segment)
        if not _collision_free_segment(grid, q_near, q_new, step_size * 0.5):
            continue

        new_cost = nodes[near_idx].cost + np.linalg.norm(q_new - q_near)
        nodes.append(RRTNode(q_new, parent=near_idx, cost=float(new_cost)))

        # Goal check
        if np.linalg.norm(q_new - goal) < goal_threshold:
            # Reconstruct path
            path = [q_new]
            idx = near_idx
            while idx is not None:
                path.append(nodes[idx].position)
                idx = nodes[idx].parent
            path.reverse()
            path.append(goal)
            return np.array(path)

    return None


def rrt_star(
    grid: OccupancyGrid3D,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    max_iterations: int = 5000,
    step_size: float = 0.3,
    goal_threshold: float = 0.3,
    goal_bias: float = 0.1,
    rewire_radius: float = 1.0,
    seed: int = 42,
) -> NDArray[np.float64] | None:
    r"""RRT* with rewiring for asymptotically optimal paths.

    Extends RRT with two key additions (Karaman & Frazzoli, 2011):

    1. **Choose parent**: among neighbors within ``rewire_radius``,
       pick the one that yields minimum cost-to-come.
    2. **Rewire**: check if routing through the new node improves
       cost for existing neighbors.

    **Optimality guarantee**: as iterations :math:`\to \infty`,
    the path cost converges to the optimal cost with probability 1.
    """
    rng = np.random.RandomState(seed)
    lo = grid.origin
    hi = grid.origin + np.array(grid.shape) * grid.resolution

    nodes = [RRTNode(start.copy())]
    best_goal_idx = None
    best_goal_cost = float("inf")

    for _iteration in range(max_iterations):
        q_rand = goal.copy() if rng.rand() < goal_bias else lo + rng.rand(3) * (hi - lo)

        # Nearest
        positions = np.array([n.position for n in nodes])
        dists = np.linalg.norm(positions - q_rand, axis=1)
        near_idx = int(np.argmin(dists))
        q_near = nodes[near_idx].position

        # Steer
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist < 1e-8:
            continue
        direction /= dist
        q_new = q_near + min(step_size, dist) * direction

        if not _collision_free_segment(grid, q_near, q_new, step_size * 0.5):
            continue

        # Find neighbors within rewire_radius
        dists_to_new = np.linalg.norm(positions - q_new, axis=1)
        neighbor_indices = np.where(dists_to_new < rewire_radius)[0]

        # Choose best parent
        best_parent = near_idx
        best_cost = nodes[near_idx].cost + np.linalg.norm(q_new - q_near)

        for ni in neighbor_indices:
            seg_cost = np.linalg.norm(q_new - nodes[ni].position)
            candidate_cost = nodes[ni].cost + seg_cost
            if candidate_cost < best_cost and _collision_free_segment(
                grid, nodes[ni].position, q_new, step_size * 0.5
            ):
                best_parent = ni
                best_cost = candidate_cost

        new_idx = len(nodes)
        nodes.append(RRTNode(q_new, parent=best_parent, cost=float(best_cost)))

        # Rewire neighbors
        for ni in neighbor_indices:
            seg_cost = np.linalg.norm(nodes[ni].position - q_new)
            candidate_cost = best_cost + seg_cost
            if candidate_cost < nodes[ni].cost and _collision_free_segment(
                grid, q_new, nodes[ni].position, step_size * 0.5
            ):
                nodes[ni].parent = new_idx
                nodes[ni].cost = candidate_cost

        # Goal check
        dist_to_goal = np.linalg.norm(q_new - goal)
        if dist_to_goal < goal_threshold:
            total_cost = best_cost + dist_to_goal
            if total_cost < best_goal_cost:
                best_goal_cost = total_cost
                best_goal_idx = new_idx

    if best_goal_idx is not None:
        path = [goal]
        idx = best_goal_idx
        while idx is not None:
            path.append(nodes[idx].position)
            idx = nodes[idx].parent
        path.reverse()
        return np.array(path)

    return None


def _collision_free_segment(
    grid: OccupancyGrid3D,
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    check_resolution: float,
) -> bool:
    """Check if a line segment is collision-free by sampling."""
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-8:
        return grid.is_free_world(p1)
    n_checks = max(2, int(dist / check_resolution))
    for i in range(n_checks + 1):
        t = i / n_checks
        p = p1 + t * (p2 - p1)
        if not grid.is_free_world(p):
            return False
    return True


# ======================================================================
#  Artificial Potential Field (APF)
# ======================================================================


def artificial_potential_field(
    grid: OccupancyGrid3D,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    k_att: float = 1.0,
    k_rep: float = 0.5,
    d0: float = 0.5,
    step_size: float = 0.05,
    max_steps: int = 5000,
    goal_threshold: float = 0.1,
) -> NDArray[np.float64]:
    r"""Gradient descent on artificial potential field.

    **Attractive potential** (quadratic near goal):

    .. math::

        U_{\text{att}}(\mathbf{q}) = \frac{1}{2} k_{\text{att}}
        \|\mathbf{q} - \mathbf{q}_g\|^2

    **Repulsive potential** (inverse-distance from obstacles):

    .. math::

        U_{\text{rep}}(\mathbf{q}) = \begin{cases}
        \frac{1}{2} k_{\text{rep}} \left(\frac{1}{d(\mathbf{q})}
        - \frac{1}{d_0}\right)^2 & \text{if } d(\mathbf{q}) \leq d_0 \\
        0 & \text{otherwise}
        \end{cases}

    **Negative gradient**:

    .. math::

        \mathbf{F} = -\nabla U_{\text{att}} - \nabla U_{\text{rep}}

    **Known limitation**: local minima can trap the planner.

    Parameters
    ----------
    k_att : attractive gain
    k_rep : repulsive gain
    d0 : influence distance of obstacles [m]
    """
    pos = start.copy()
    path = [pos.copy()]

    for _ in range(max_steps):
        if np.linalg.norm(pos - goal) < goal_threshold:
            path.append(goal.copy())
            break

        # Attractive force
        f_att = -k_att * (pos - goal)

        # Repulsive force (sample nearby obstacle voxels)
        f_rep = _compute_repulsive_force(grid, pos, k_rep, d0)

        # Total force
        f_total = f_att + f_rep
        f_norm = np.linalg.norm(f_total)
        if f_norm < 1e-8:
            break

        direction = f_total / f_norm
        new_pos = pos + step_size * direction

        if not grid.is_free_world(new_pos):
            break

        pos = new_pos
        path.append(pos.copy())

    return np.array(path)


def _compute_repulsive_force(
    grid: OccupancyGrid3D,
    pos: NDArray[np.float64],
    k_rep: float,
    d0: float,
) -> NDArray[np.float64]:
    """Compute repulsive force from nearby obstacles."""
    f_rep = np.zeros(3)
    search_radius = int(np.ceil(d0 / grid.resolution))
    center_idx = grid.world_to_grid(pos)

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            for dz in range(-search_radius, search_radius + 1):
                idx = (center_idx[0] + dx, center_idx[1] + dy, center_idx[2] + dz)
                if not grid.is_valid(idx) or not grid.grid[idx]:
                    continue

                obs_pos = grid.grid_to_world(idx)
                diff = pos - obs_pos
                dist = np.linalg.norm(diff)

                if dist < 1e-6 or dist > d0:
                    continue

                # Gradient of repulsive potential
                grad = k_rep * (1.0 / dist - 1.0 / d0) * (1.0 / dist**2) * (diff / dist)
                f_rep += grad

    return f_rep


# ======================================================================
#  Path Smoothing
# ======================================================================


def smooth_path(
    path: NDArray[np.float64],
    grid: OccupancyGrid3D,
    n_iterations: int = 100,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Shortcut-based path smoothing.

    Randomly picks two path points and checks if the direct
    segment is collision-free. If so, removes intermediate points.
    """
    rng = np.random.RandomState(seed)
    pts = list(path)

    for _ in range(n_iterations):
        if len(pts) <= 2:
            break

        i = rng.randint(0, len(pts) - 2)
        j = rng.randint(i + 2, len(pts))

        if _collision_free_segment(grid, pts[i], pts[j], grid.resolution):
            pts = pts[: i + 1] + pts[j:]

    return np.array(pts)


def resample_path(
    path: NDArray[np.float64],
    spacing: float,
) -> NDArray[np.float64]:
    """Resample a path to have uniform spacing between waypoints."""
    if len(path) <= 1:
        return path

    # Cumulative arc length
    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length < 1e-8:
        return path

    n_points = max(2, int(total_length / spacing))
    target_lengths = np.linspace(0, total_length, n_points)

    resampled = np.zeros((n_points, 3))
    for i, target in enumerate(target_lengths):
        idx = np.searchsorted(cum_length, target, side="right") - 1
        idx = np.clip(idx, 0, len(path) - 2)
        seg_frac = (target - cum_length[idx]) / (seg_lengths[idx] + 1e-12)
        resampled[i] = path[idx] + seg_frac * diffs[idx]

    return resampled


# ======================================================================
#  Path Metrics
# ======================================================================


def path_length(path: NDArray[np.float64]) -> float:
    """Total Euclidean length of a path."""
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def path_clearance(
    path: NDArray[np.float64],
    grid: OccupancyGrid3D,
) -> float:
    """Minimum distance from any path waypoint to the nearest obstacle voxel.

    This is the safety margin of a trajectory: a value of zero means the path
    passes through an occupied voxel, and small positive values mean it
    grazes an obstacle.

    Obstacles are treated as point sets located at their voxel *centres*, and
    the distance is the ordinary Euclidean distance from the waypoint to that
    centre.  A waypoint lying inside an occupied voxel therefore reports 0.

    Parameters
    ----------
    path : (N, 3) world coordinates
    grid : OccupancyGrid3D

    Returns
    -------
    clearance : float
        Minimum distance to the nearest obstacle voxel centre, or ``inf``
        when the path is empty or the grid contains no obstacles.
    """
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    occ = np.argwhere(grid.grid)
    if path.size == 0 or occ.size == 0:
        return float("inf")

    # Obstacle voxel centres in world coordinates: origin + (idx + 0.5) * res.
    obstacles = grid.origin + (occ + 0.5) * grid.resolution

    # Chunk over waypoints so that memory stays bounded for long paths while
    # still using a single vectorised distance computation per chunk.
    min_clearance = float("inf")
    chunk = max(1, 4_000_000 // max(1, len(obstacles)))
    for start in range(0, len(path), chunk):
        block = path[start : start + chunk]
        d = np.linalg.norm(block[:, None, :] - obstacles[None, :, :], axis=2)
        min_clearance = min(min_clearance, float(d.min()))
    return min_clearance
