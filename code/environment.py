"""
environment.py
Defines 3D multi-UAV cooperative path planning environments with static
and dynamic (moving) obstacles, and the shared fitness function used to
evaluate every optimizer (PSO, GWO, WOA, CD-MSIPSO) fairly.

Path encoding:
  Each UAV path = fixed start -> N_WP interior waypoints -> fixed goal.
  A particle encodes ALL UAVs' interior waypoints concatenated:
    dim = N_UAV * N_WP * 3   (x, y, z per waypoint)

Fitness (to MINIMIZE) = weighted sum of:
  J_len   : total path length (efficiency)
  J_stat  : static obstacle intrusion penalty
  J_dyn   : dynamic (moving) obstacle intrusion penalty (time-aware)
  J_coop  : inter-UAV collision penalty (time-synchronized)
  J_smooth: turning-angle penalty (flyability)
  J_bound : out-of-bound / altitude penalty
"""
import time
import numpy as np

BOUNDS_XY = (0.0, 100.0)
BOUNDS_Z = (5.0, 45.0)
N_WP = 6          # interior waypoints per UAV
N_SAMPLES = 25    # points sampled along each path for cost evaluation
SAFETY_STATIC = 6.0
SAFETY_DYNAMIC = 5.5
SAFETY_UAV = 5.0

WEIGHTS = dict(len=1.0, stat=60.0, dyn=80.0, coop=100.0, smooth=0.4, bound=50.0)


class Environment:
    def __init__(self, name, n_uav, starts, goals, static_obs, dynamic_obs, seed=0):
        """
        starts, goals: (n_uav, 3) arrays
        static_obs:  list of (cx, cy, cz, r)
        dynamic_obs: list of (cx, cy, cz, r, vx, vy, vz)  velocity per unit "time" in [0,1]
        """
        self.name = name
        self.n_uav = n_uav
        self.starts = np.array(starts, dtype=float)
        self.goals = np.array(goals, dtype=float)
        self.static_obs = np.array(static_obs, dtype=float) if static_obs else np.zeros((0, 4))
        self.dynamic_obs = np.array(dynamic_obs, dtype=float) if dynamic_obs else np.zeros((0, 7))
        self.dim = n_uav * N_WP * 3
        self.rng = np.random.default_rng(seed)
        self.n_samples = N_SAMPLES
        self.weights = dict(WEIGHTS)
        self.safety_static = SAFETY_STATIC
        self.safety_dynamic = SAFETY_DYNAMIC
        self.safety_uav = SAFETY_UAV
        self._arch_on = False
        self._arch_x = None
        self._arch_fit = np.inf
        self.n_evals = 0
        self.arch_time = 0.0

    def lower_bounds(self):
        lb = np.tile([BOUNDS_XY[0], BOUNDS_XY[0], BOUNDS_Z[0]], self.n_uav * N_WP)
        return lb

    def upper_bounds(self):
        ub = np.tile([BOUNDS_XY[1], BOUNDS_XY[1], BOUNDS_Z[1]], self.n_uav * N_WP)
        return ub

    def init_population(self, n_particles, rng=None):
        """If rng is provided, use it (makes population generation controlled
        by the caller's seed, independent of call order). Otherwise falls
        back to the environment's own persistent RNG stream."""
        lb, ub = self.lower_bounds(), self.upper_bounds()
        source = rng if rng is not None else self.rng
        return source.uniform(lb, ub, size=(n_particles, self.dim))

    def _decode(self, x):
        """x: (dim,) -> paths: (n_uav, N_WP+2, 3) including fixed start/goal"""
        wp = x.reshape(self.n_uav, N_WP, 3)
        paths = np.zeros((self.n_uav, N_WP + 2, 3))
        paths[:, 0, :] = self.starts
        paths[:, 1:-1, :] = wp
        paths[:, -1, :] = self.goals
        return paths

    def _sample_paths(self, paths):
        """Uniformly resample each UAV's polyline into N_SAMPLES points -> (n_uav, N_SAMPLES, 3)"""
        n_uav = paths.shape[0]
        out = np.zeros((n_uav, self.n_samples, 3))
        t_query = np.linspace(0, 1, self.n_samples)
        for u in range(n_uav):
            pts = paths[u]
            seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg_len)])
            total = cum[-1] if cum[-1] > 1e-9 else 1e-9
            t_pts = cum / total
            for d in range(3):
                out[u, :, d] = np.interp(t_query, t_pts, pts[:, d])
        return out

    def components(self, x):
        """Unweighted cost components of a solution (dict)."""
        paths = self._decode(x)
        sampled = self._sample_paths(paths)  # (n_uav, N_SAMPLES, 3)

        seg = np.diff(paths, axis=1)
        J_len = np.linalg.norm(seg, axis=2).sum()

        J_stat = 0.0
        for (cx, cy, cz, r) in self.static_obs:
            d = np.linalg.norm(sampled - np.array([cx, cy, cz]), axis=2)
            viol = np.maximum(0.0, (r + self.safety_static) - d)
            J_stat += np.sum(viol ** 2)

        J_dyn = 0.0
        if len(self.dynamic_obs):
            t = np.linspace(0, 1, self.n_samples).reshape(1, -1)
            for (cx, cy, cz, r, vx, vy, vz) in self.dynamic_obs:
                center_t = np.stack([cx + vx * t[0], cy + vy * t[0], cz + vz * t[0]], axis=1)
                d = np.linalg.norm(sampled - center_t[None, :, :], axis=2)
                viol = np.maximum(0.0, (r + self.safety_dynamic) - d)
                J_dyn += np.sum(viol ** 2)

        # Each UNORDERED UAV pair is counted exactly once (i < j).
        J_coop = 0.0
        n_uav = sampled.shape[0]
        for i in range(n_uav):
            for j in range(i + 1, n_uav):
                d = np.linalg.norm(sampled[i] - sampled[j], axis=1)
                viol = np.maximum(0.0, self.safety_uav - d)
                J_coop += np.sum(viol ** 2)

        J_smooth = 0.0
        for u in range(n_uav):
            v = np.diff(paths[u], axis=0)
            norms = np.linalg.norm(v, axis=1) + 1e-9
            vn = v / norms[:, None]
            cos_ang = np.clip(np.sum(vn[:-1] * vn[1:], axis=1), -1, 1)
            ang = np.arccos(cos_ang)
            J_smooth += np.sum(ang ** 2)

        lb, ub = self.lower_bounds(), self.upper_bounds()
        below = np.maximum(0.0, lb - x)
        above = np.maximum(0.0, x - ub)
        J_bound = np.sum(below ** 2) + np.sum(above ** 2)
        return dict(len=J_len, stat=J_stat, dyn=J_dyn, coop=J_coop, smooth=J_smooth, bound=J_bound)

    def evaluate(self, x):
        """Single-particle weighted fitness (to minimise)."""
        c = self.components(x)
        f = sum(self.weights[k] * c[k] for k in ('len', 'stat', 'dyn', 'coop', 'smooth', 'bound'))
        if getattr(self, 'exact_fitness', False):
            # optional: add quadratic penalties on the EXACT continuous-time clearance violations
            from verify import exact_report
            r = exact_report(self, x)
            f += (self.weights['stat'] * max(0.0, -r['min_static_margin']) ** 2
                  + self.weights['dyn'] * max(0.0, -r['min_dyn_margin']) ** 2
                  + self.weights['coop'] * max(0.0, self.safety_uav - r['min_uav_sep']) ** 2)
        return f

    def evaluate_batch(self, X):
        fits = np.array([self.evaluate(x) for x in X])
        self.n_evals += len(X)
        if self._arch_on:
            _t0 = time.perf_counter()
            # Best-feasible archive: records the lowest-fitness candidate that is
            # feasible (at the optimisation sample points) among ALL candidates
            # ever evaluated. Does not influence any optimiser (no RNG use).
            for x, f in zip(X, fits):
                if f < self._arch_fit and self.is_feasible(x):
                    self._arch_fit = float(f)
                    self._arch_x = x.copy()
            self.arch_time += time.perf_counter() - _t0
        return fits

    def start_archive(self):
        self._arch_on, self._arch_x, self._arch_fit, self.n_evals, self.arch_time = True, None, np.inf, 0, 0.0

    def get_archive(self):
        return self._arch_x, self._arch_fit

    def is_feasible(self, x, tol=1e-6):
        """Check zero-violation feasibility (no static/dynamic/UAV-UAV collision)."""
        paths = self._decode(x)
        sampled = self._sample_paths(paths)
        for (cx, cy, cz, r) in self.static_obs:
            d = np.linalg.norm(sampled - np.array([cx, cy, cz]), axis=2)
            if np.any(d < (r + self.safety_static) - tol):
                return False
        if len(self.dynamic_obs):
            t = np.linspace(0, 1, self.n_samples)
            for (cx, cy, cz, r, vx, vy, vz) in self.dynamic_obs:
                center_t = np.stack([cx + vx * t, cy + vy * t, cz + vz * t], axis=1)
                d = np.linalg.norm(sampled - center_t[None, :, :], axis=2)
                if np.any(d < (r + self.safety_dynamic) - tol):
                    return False
        n_uav = sampled.shape[0]
        for i in range(n_uav):
            for j in range(i + 1, n_uav):
                d = np.linalg.norm(sampled[i] - sampled[j], axis=1)
                if np.any(d < self.safety_uav - tol):
                    return False
        return True

    def path_length(self, x):
        paths = self._decode(x)
        seg = np.diff(paths, axis=1)
        return np.linalg.norm(seg, axis=2).sum()

    def decode_for_plot(self, x):
        return self._decode(x)


def make_environments():
    envs = []

    # ENV-1: 3 UAVs crossing paths, 4 static obstacles, 2 dynamic obstacles
    envs.append(Environment(
        name="E1_CrossingTraffic",
        n_uav=3,
        starts=[[5, 5, 20], [5, 95, 20], [95, 5, 25]],
        goals=[[95, 95, 25], [95, 5, 20], [5, 95, 20]],
        static_obs=[(50, 50, 22, 10), (30, 70, 20, 7), (70, 30, 25, 7), (50, 20, 18, 6)],
        dynamic_obs=[(20, 20, 20, 4, 40, 40, 0), (80, 80, 22, 4, -35, -35, 0)],
        seed=1,
    ))

    # ENV-2: 4 UAVs dense static field, 3 dynamic obstacles
    envs.append(Environment(
        name="E2_DenseField",
        n_uav=4,
        starts=[[5, 5, 15], [5, 50, 20], [5, 95, 25], [50, 5, 30]],
        goals=[[95, 95, 30], [95, 50, 25], [95, 5, 20], [50, 95, 15]],
        static_obs=[(50, 50, 22, 9), (25, 75, 20, 6), (75, 25, 24, 6),
                    (35, 35, 18, 5), (65, 65, 26, 5), (50, 15, 20, 5)],
        dynamic_obs=[(10, 60, 20, 4, 50, -20, 0), (90, 40, 24, 4, -45, 15, 0), (40, 90, 22, 4, 10, -50, 0)],
        seed=2,
    ))

    # ENV-3: 3 UAVs sparse obstacles but fast dynamic threats (stress test)
    envs.append(Environment(
        name="E3_FastDynamicThreats",
        n_uav=3,
        starts=[[5, 50, 20], [50, 5, 22], [95, 50, 18]],
        goals=[[95, 50, 18], [50, 95, 22], [5, 50, 20]],
        static_obs=[(50, 50, 20, 8)],
        dynamic_obs=[(50, 10, 20, 4, 0, 70, 0), (10, 50, 20, 4, 70, 0, 0), (90, 50, 20, 4, -70, 0, 0)],
        seed=3,
    ))

    # ENV-3R: revised fast-threat stress test. Same layout and UAV start/goal
    # positions as E3, but the three fast obstacles start >= 15 units away from the
    # UAV start positions, so that the fixed endpoints are themselves safe
    # (E3 itself is infeasible by construction, see verify.fixed_endpoint_conflicts).
    envs.append(Environment(
        name="E3R_FastDynamicThreats_Revised",
        n_uav=3,
        starts=[[5, 50, 20], [50, 5, 22], [95, 50, 18]],
        goals=[[95, 50, 18], [50, 95, 22], [5, 50, 20]],
        static_obs=[(50, 50, 20, 8)],
        dynamic_obs=[(50, 20, 20, 4, 0, 55, 0), (25, 50, 20, 4, 55, 0, 0), (75, 50, 20, 4, -55, 0, 0)],
        seed=4,
    ))

    return envs
