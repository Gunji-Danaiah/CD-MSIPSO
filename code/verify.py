"""
verify.py -- independent post-processing safety verification.

The optimiser evaluates safety only at N_SAMPLES = 25 time-synchronised points.
This module checks a final solution WITHOUT using that grid.

Timing model (identical to the fitness function, now stated explicitly):
  * every UAV flies its polyline at constant speed and arrives at the goal at
    the common normalised time t = 1 (so speeds differ between UAVs);
  * the k-th waypoint of UAV u is reached at t_k^u = (cumulative arc length up
    to waypoint k) / (total path length);
  * a moving obstacle moves linearly: c(t) = c0 + v * t,  t in [0, 1].

Under this model the position of every UAV, and of every obstacle, is a
piecewise-linear function of t. The relative position of any two such objects
is therefore also piecewise linear, and the minimum of its norm over each linear
piece is available in closed form. `exact_min_dist` merges the breakpoints of
the two trajectories and minimises over every piece, giving the EXACT
continuous-time minimum separation (no sampling, no tolerance on the grid).
"""
import numpy as np
from environment import N_WP


def uav_trajectory(env, x):
    """-> list of (T, P): breakpoints (K,) and points (K,3) for each UAV."""
    paths = env._decode(x)
    out = []
    for u in range(env.n_uav):
        pts = paths[u]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1] if cum[-1] > 1e-9 else 1e-9
        out.append((cum / total, pts))
    return out


def _pos_at(T, P, t):
    return np.stack([np.interp(t, T, P[:, d]) for d in range(3)], axis=-1)


def exact_min_dist(TA, PA, TB, PB):
    """Exact min over t in [0,1] of |A(t) - B(t)| for piecewise-linear A, B."""
    ts = np.unique(np.concatenate([TA, TB, [0.0, 1.0]]))
    R = _pos_at(TA, PA, ts) - _pos_at(TB, PB, ts)          # (n, 3) at breakpoints
    R0, R1 = R[:-1], R[1:]
    D = R1 - R0
    dd = np.sum(D * D, axis=1)
    s = np.where(dd > 1e-18, -np.sum(R0 * D, axis=1) / np.maximum(dd, 1e-18), 0.0)
    s = np.clip(s, 0.0, 1.0)
    closest = R0 + D * s[:, None]
    return float(np.min(np.linalg.norm(closest, axis=1)))


def exact_report(env, x):
    """Continuous-time clearances of a solution (exact under the timing model)."""
    traj = uav_trajectory(env, x)
    still = np.array([0.0, 1.0])
    min_static = np.inf      # min over (uav, static obstacle) of dist - (r + safety)
    for (cx, cy, cz, r) in env.static_obs:
        c = np.array([[cx, cy, cz], [cx, cy, cz]])
        for (T, P) in traj:
            min_static = min(min_static, exact_min_dist(T, P, still, c) - (r + env.safety_static))
    min_dyn = np.inf
    for (cx, cy, cz, r, vx, vy, vz) in env.dynamic_obs:
        c = np.array([[cx, cy, cz], [cx + vx, cy + vy, cz + vz]])
        for (T, P) in traj:
            min_dyn = min(min_dyn, exact_min_dist(T, P, still, c) - (r + env.safety_dynamic))
    min_sep = np.inf         # min over UAV pairs of centre distance
    for i in range(env.n_uav):
        for j in range(i + 1, env.n_uav):
            min_sep = min(min_sep, exact_min_dist(traj[i][0], traj[i][1], traj[j][0], traj[j][1]))
    tol = 1e-6
    feasible = (min_static >= -tol) and (min_dyn >= -tol) and (min_sep >= env.safety_uav - tol)
    return dict(min_static_margin=min_static, min_dyn_margin=min_dyn,
                min_uav_sep=min_sep, feasible_exact=bool(feasible))


def sampled_feasible(env, x, M):
    """Feasibility checked on M uniformly spaced normalised times (incl. 0 and 1)."""
    traj = uav_trajectory(env, x)
    t = np.linspace(0.0, 1.0, M)
    pos = np.stack([_pos_at(T, P, t) for (T, P) in traj])   # (n_uav, M, 3)
    tol = 1e-6
    for (cx, cy, cz, r) in env.static_obs:
        d = np.linalg.norm(pos - np.array([cx, cy, cz]), axis=2)
        if np.any(d < r + env.safety_static - tol):
            return False
    for (cx, cy, cz, r, vx, vy, vz) in env.dynamic_obs:
        c = np.stack([cx + vx * t, cy + vy * t, cz + vz * t], axis=1)
        d = np.linalg.norm(pos - c[None], axis=2)
        if np.any(d < r + env.safety_dynamic - tol):
            return False
    for i in range(env.n_uav):
        for j in range(i + 1, env.n_uav):
            d = np.linalg.norm(pos[i] - pos[j], axis=1)
            if np.any(d < env.safety_uav - tol):
                return False
    return True


def fixed_endpoint_conflicts(env):
    """Safety margins of the FIXED start (t=0) and goal (t=1) positions.

    Start/goal positions are not decision variables, so if any of them is already
    inside a safety envelope at t=0 or t=1 NO trajectory can be feasible, whatever
    the optimiser does. Returns a list of (what, margin) with margin < 0 = conflict,
    for every conflict found (empty list => endpoints are not a source of infeasibility)."""
    out = []
    for t, pts, tag in ((0.0, env.starts, 'start'), (1.0, env.goals, 'goal')):
        for u, p in enumerate(pts):
            for k, (cx, cy, cz, r) in enumerate(env.static_obs):
                m = np.linalg.norm(p - np.array([cx, cy, cz])) - (r + env.safety_static)
                if m < 0: out.append((f'UAV{u} {tag} vs static{k}', float(m)))
            for k, (cx, cy, cz, r, vx, vy, vz) in enumerate(env.dynamic_obs):
                c = np.array([cx + vx * t, cy + vy * t, cz + vz * t])
                m = np.linalg.norm(p - c) - (r + env.safety_dynamic)
                if m < 0: out.append((f'UAV{u} {tag} vs dynamic{k}', float(m)))
        for i in range(env.n_uav):
            for j in range(i + 1, env.n_uav):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < env.safety_uav: out.append((f'UAV{i}-UAV{j} {tag}s', float(d - env.safety_uav)))
    return out
