"""
cdmsipso.py -- CD-MSIPSO (Cooperative Dynamic-obstacle Multi-Strategy Improved PSO)
and its ablation variants, all produced by run_variant():

    rep=False, mut=False   MSIPSO (multi-strategy core + adaptive inertia, deadlock
                           escape, generic diversifying mutation, elitism)
    rep=True               + cooperative repulsion operator
    mut=True               + dynamic-obstacle-aware localised mutation
    tc=True                time-consistent versions of the enabled operators

Operators
  (1) Cooperative repulsion (applied after the position update, as an additive
      correction of the waypoint coordinates): waypoint k of UAV u is pushed away
      from the corresponding waypoint (index-matched, or time-matched when tc=True)
      of every other UAV v OF THE SAME PARTICLE when they are closer than `margin`.
      It does not use the global best (the `gbest` argument is kept only for a
      stable function signature).
  (2) Dynamic-obstacle-aware localised mutation: the position of each moving
      obstacle is predicted (constant velocity) at each waypoint's assumed (or,
      when tc=True, actual arc-length) time; only waypoints inside the danger
      radius are perturbed.
"""
import numpy as np
from environment import N_WP


OP_STATS = dict(rep_wp=0, rep_active=0, rep_disp=0.0, mut_wp=0, mut_active=0, mut_disp=0.0)

def clamp(X, lb, ub):
    return np.clip(X, lb, ub)


def _adaptive_inertia(fit, w_max=0.9, w_min=0.4):
    f_min, f_max = fit.min(), fit.max()
    if f_max - f_min < 1e-12:
        return np.full_like(fit, (w_max + w_min) / 2)
    return w_min + (w_max - w_min) * (fit - f_min) / (f_max - f_min)


def _cooperative_repulsion(X, gbest, env, strength=0.6, margin=9.0):
    """INDEX-MATCHED repulsion (original formulation): waypoint k of UAV u is
    pushed away from waypoint k of every other UAV v of the SAME particle when
    their distance is below `margin`. Degenerate case: if two waypoints coincide
    (distance < 1e-9) the push direction is the fixed unit vector +x for the
    lower-indexed UAV and -x for the higher-indexed UAV (antisymmetric)."""
    n_particles = X.shape[0]
    n_uav = env.n_uav
    if n_uav < 2:
        return np.zeros_like(X)
    Xr = X.reshape(n_particles, n_uav, N_WP, 3)
    rep = np.zeros_like(Xr)
    ex = np.array([1.0, 0.0, 0.0])
    for u in range(n_uav):
        for v in range(n_uav):
            if u == v:
                continue
            diff = Xr[:, u, :, :] - Xr[:, v, :, :]
            raw = np.linalg.norm(diff, axis=2, keepdims=True)
            dist = raw + 1e-6
            violation = np.maximum(0.0, margin - dist[:, :, 0])
            direction = np.where(raw < 1e-9, (1.0 if u < v else -1.0) * ex, diff / dist)
            push = direction * (strength * violation[:, :, None] / margin)
            rep[:, u, :, :] += push
    return rep.reshape(n_particles, -1)


def _paths_and_times(X, env):
    """paths (P,U,N_WP+2,3) and normalised arrival times T (P,U,N_WP+2) under the
    constant-speed / common-arrival-time model used by the fitness function."""
    P = X.shape[0]
    U = env.n_uav
    wp = X.reshape(P, U, N_WP, 3)
    paths = np.empty((P, U, N_WP + 2, 3))
    paths[:, :, 0, :] = env.starts[None]
    paths[:, :, -1, :] = env.goals[None]
    paths[:, :, 1:-1, :] = wp
    seg = np.linalg.norm(np.diff(paths, axis=2), axis=3)
    cum = np.concatenate([np.zeros((P, U, 1)), np.cumsum(seg, axis=2)], axis=2)
    tot = np.maximum(cum[:, :, -1:], 1e-9)
    return paths, cum / tot


def _cooperative_repulsion_tc(X, gbest, env, strength=0.6, margin=9.0):
    """TIME-CONSISTENT repulsion: waypoint k of UAV u (reached at its own arrival
    time t_k^u) is pushed away from the POSITION OF EVERY OTHER UAV v AT THAT SAME
    TIME t_k^u, obtained by interpolating v's polyline at t_k^u. The corrected
    waypoint is a decision variable, so the correction is transferred to the
    trajectory representation directly."""
    P = X.shape[0]
    U = env.n_uav
    if U < 2:
        return np.zeros_like(X)
    paths, T = _paths_and_times(X, env)
    wp = paths[:, :, 1:-1, :]
    rep = np.zeros_like(wp)
    ex = np.array([1.0, 0.0, 0.0])
    for p in range(P):
        for u in range(U):
            tu = T[p, u, 1:-1]
            for v in range(U):
                if v == u:
                    continue
                pos_v = np.stack([np.interp(tu, T[p, v], paths[p, v, :, d]) for d in range(3)], axis=1)
                diff = wp[p, u] - pos_v
                raw = np.linalg.norm(diff, axis=1, keepdims=True)
                dist = raw + 1e-6
                violation = np.maximum(0.0, margin - dist[:, 0])
                direction = np.where(raw < 1e-9, (1.0 if u < v else -1.0) * ex, diff / dist)
                rep[p, u] += direction * (strength * violation[:, None] / margin)
    return rep.reshape(P, -1)


def _dynamic_aware_mutation(X, env, rng, mutation_rate=0.12, strength=0.12, eps=2.0):
    """
    Predict each dynamic obstacle's position at each waypoint's normalized
    time index; if a UAV's waypoint block is within the danger radius,
    mutate that block only (localized escape), scaled by 'strength'.
    Rate/strength are meant to be passed in already annealed by the caller.
    """
    n_particles = X.shape[0]
    n_uav = env.n_uav
    lb, ub = env.lower_bounds(), env.upper_bounds()
    lb_r = lb.reshape(n_uav, N_WP, 3)
    ub_r = ub.reshape(n_uav, N_WP, 3)
    Xr = X.reshape(n_particles, n_uav, N_WP, 3).copy()
    if len(env.dynamic_obs) == 0:
        return Xr.reshape(n_particles, -1)

    t_wp = np.linspace(0, 1, N_WP + 2)[1:-1]  # time for interior waypoints
    for (cx, cy, cz, r, vx, vy, vz) in env.dynamic_obs:
        center_t = np.stack([cx + vx * t_wp, cy + vy * t_wp, cz + vz * t_wp], axis=1)  # (N_WP,3)
        for u in range(n_uav):
            d = np.linalg.norm(Xr[:, u, :, :] - center_t[None, :, :], axis=2)  # (n_particles, N_WP)
            danger = d < (r + env.safety_dynamic + eps)
            if not danger.any():
                continue
            do_mut = danger & (rng.random(danger.shape) < mutation_rate)
            noise = rng.uniform(-1, 1, Xr[:, u, :, :].shape) * strength * (ub_r[u] - lb_r[u])
            Xr[:, u, :, :] = np.where(do_mut[:, :, None], Xr[:, u, :, :] + noise, Xr[:, u, :, :])
    Xr = clamp(Xr.reshape(n_particles, -1), lb, ub)
    return Xr


def _dynamic_aware_mutation_tc(X, env, rng, mutation_rate=0.12, strength=0.12):
    """TIME-CONSISTENT mutation: obstacle position is predicted at each waypoint's
    own arc-length arrival time t_k^u (not at uniformly spaced times)."""
    P = X.shape[0]
    U = env.n_uav
    lb, ub = env.lower_bounds(), env.upper_bounds()
    lb_r, ub_r = lb.reshape(U, N_WP, 3), ub.reshape(U, N_WP, 3)
    if len(env.dynamic_obs) == 0:
        return X.copy()
    paths, T = _paths_and_times(X, env)
    Xr = paths[:, :, 1:-1, :].copy()
    t_wp = T[:, :, 1:-1]                                     # (P,U,N_WP)
    for (cx, cy, cz, r, vx, vy, vz) in env.dynamic_obs:
        center = np.stack([cx + vx * t_wp, cy + vy * t_wp, cz + vz * t_wp], axis=3)
        d = np.linalg.norm(Xr - center, axis=3)
        danger = d < (r + env.safety_dynamic + 2.0)
        if not danger.any():
            continue
        do_mut = danger & (rng.random(danger.shape) < mutation_rate)
        noise = rng.uniform(-1, 1, Xr.shape) * strength * (ub_r - lb_r)[None]
        Xr = np.where(do_mut[..., None], Xr + noise, Xr)
    return clamp(Xr.reshape(P, -1), lb, ub)


def _multi_strategy_update(X, V, pbest, gbest, fit, rng, lb, ub, dim, gwo_prob=0.85):
    """
    Multi-strategy core update: each particle stochastically follows either
    (a) a GWO-style pull toward the top-3 leaders (strong exploitation), or
    (b) a PSO-style velocity update toward pbest/gbest (balanced exploration).
    This is the 'multi-strategy' component of MSIPSO. Returns new X, V.
    """
    n_particles = X.shape[0]
    order = np.argsort(fit)
    leader1, leader2, leader3 = X[order[0]], X[order[1]], X[order[2]]

    use_gwo = rng.random(n_particles) < gwo_prob
    X_new = X.copy()
    V_new = V.copy()

    # GWO-style branch
    if use_gwo.any():
        idx = np.where(use_gwo)[0]
        a = 1.3  # exploitation strength (fixed; annealed externally via caller if desired)
        A1 = 2 * a * rng.random((len(idx), dim)) - a
        C1 = 2 * rng.random((len(idx), dim))
        X1 = leader1 - A1 * np.abs(C1 * leader1 - X[idx])
        A2 = 2 * a * rng.random((len(idx), dim)) - a
        C2 = 2 * rng.random((len(idx), dim))
        X2 = leader2 - A2 * np.abs(C2 * leader2 - X[idx])
        A3 = 2 * a * rng.random((len(idx), dim)) - a
        C3 = 2 * rng.random((len(idx), dim))
        X3 = leader3 - A3 * np.abs(C3 * leader3 - X[idx])
        X_new[idx] = (X1 + X2 + X3) / 3.0
        V_new[idx] = X_new[idx] - X[idx]

    # PSO-style branch
    if (~use_gwo).any():
        idx = np.where(~use_gwo)[0]
        w = _adaptive_inertia(fit[idx])
        r1, r2 = rng.random((len(idx), dim)), rng.random((len(idx), dim))
        V_new[idx] = w[:, None] * V[idx] + 1.49 * r1 * (pbest[idx] - X[idx]) + 1.49 * r2 * (gbest - X[idx])
        X_new[idx] = X[idx] + V_new[idx]

    return clamp(X_new, lb, ub), V_new


def run_variant(env, n_particles=30, iters=120, seed=0, stagnation_limit=8,
                gwo_prob=0.85, repulsion_margin=9.0, rep=True, mut=True, tc=False,
                rep_strength=0.6, mut_rate=0.12, mut_strength=0.12, mut_eps=2.0):
    """Unified runner for the factorial ablation.
        rep=False, mut=False        -> MSIPSO (no proposed operator)
        rep=True,  mut=False        -> cooperative repulsion only
        rep=False, mut=True         -> dynamic-aware mutation only
        rep=True,  mut=True         -> CD-MSIPSO
        tc=True                     -> time-consistent versions of the enabled operators
    The random-number stream is consumed identically by every variant up to the
    point where an operator is applied, so variants that share a seed share the
    same initial population."""
    OP_STATS.update(rep_wp=0, rep_active=0, rep_disp=0.0, mut_wp=0, mut_active=0, mut_disp=0.0)   # diagnostics only (no RNG)
    rng = np.random.default_rng(seed)
    lb, ub = env.lower_bounds(), env.upper_bounds()
    X = env.init_population(n_particles, rng=rng)
    V = rng.uniform(-1, 1, X.shape) * (ub - lb) * 0.05
    fit = env.evaluate_batch(X)
    pbest, pbest_fit = X.copy(), fit.copy()
    gbest_idx = np.argmin(fit)
    gbest, gbest_fit = X[gbest_idx].copy(), fit[gbest_idx]
    history = [gbest_fit]
    stagnant = 0
    rep_fn = _cooperative_repulsion_tc if tc else _cooperative_repulsion
    mut_fn = _dynamic_aware_mutation_tc if tc else _dynamic_aware_mutation

    for it in range(iters):
        decay = 1.0 - 0.7 * (it / max(1, iters - 1))      # anneal 1.0 -> 0.3

        X, V = _multi_strategy_update(X, V, pbest, gbest, fit, rng, lb, ub, env.dim, gwo_prob=gwo_prob)

        if rep and env.n_uav > 1:
            R = rep_fn(X, gbest, env, strength=rep_strength * decay, margin=repulsion_margin)
            Rw = np.linalg.norm(R.reshape(n_particles, -1, 3), axis=2)          # per-waypoint displacement
            OP_STATS['rep_wp'] += Rw.size; OP_STATS['rep_active'] += int((Rw > 0).sum()); OP_STATS['rep_disp'] += float(Rw.sum())
            X = clamp(X + R, lb, ub)

        if rng.random() < 0.2:                             # generic diversified mutation
            idx = rng.random(X.shape) < 0.1
            X = np.where(idx, X + rng.normal(0, 5, X.shape), X)
            X = clamp(X, lb, ub)

        if mut:
            X_before = X
            X = mut_fn(X, env, rng, mutation_rate=mut_rate * decay, strength=mut_strength * decay, **({} if tc else {'eps': mut_eps}))
            Mw = np.linalg.norm((X - X_before).reshape(n_particles, -1, 3), axis=2)
            OP_STATS['mut_wp'] += Mw.size; OP_STATS['mut_active'] += int((Mw > 0).sum()); OP_STATS['mut_disp'] += float(Mw.sum())

        fit = env.evaluate_batch(X)

        worst = np.argmax(fit)                             # elitism
        X[worst] = gbest.copy()
        fit[worst] = gbest_fit

        improve = fit < pbest_fit
        pbest[improve], pbest_fit[improve] = X[improve], fit[improve]
        if fit.min() < gbest_fit - 1e-9:
            gbest_fit = fit.min()
            gbest = X[np.argmin(fit)].copy()
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= stagnation_limit:                   # deadlock escape
            worst_idx = np.argsort(fit)[-max(1, n_particles // 4):]
            X[worst_idx] = env.init_population(len(worst_idx), rng=rng)
            fit[worst_idx] = env.evaluate_batch(X[worst_idx])
            stagnant = 0

        history.append(gbest_fit)
    return gbest, gbest_fit, history


def run_msipso_plain(env, n_particles=30, iters=120, seed=0, stagnation_limit=8):
    return run_variant(env, n_particles, iters, seed, stagnation_limit, rep=False, mut=False)


def run_cdmsipso(env, n_particles=30, iters=120, seed=0, stagnation_limit=8, gwo_prob=0.85, repulsion_margin=9.0):
    return run_variant(env, n_particles, iters, seed, stagnation_limit, gwo_prob, repulsion_margin, rep=True, mut=True)
