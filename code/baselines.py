"""
baselines.py — standard PSO, GWO, WOA implementations operating on a
shared Environment.evaluate fitness function. Used as comparison methods.
"""
import numpy as np


def clamp(X, lb, ub):
    return np.clip(X, lb, ub)


def run_pso(env, n_particles=30, iters=120, seed=0, w=0.72, c1=1.49, c2=1.49):
    rng = np.random.default_rng(seed)
    lb, ub = env.lower_bounds(), env.upper_bounds()
    X = env.init_population(n_particles, rng=rng)
    V = rng.uniform(-1, 1, X.shape) * (ub - lb) * 0.05
    fit = env.evaluate_batch(X)
    pbest = X.copy()
    pbest_fit = fit.copy()
    gbest_idx = np.argmin(fit)
    gbest = X[gbest_idx].copy()
    gbest_fit = fit[gbest_idx]
    history = [gbest_fit]

    for _ in range(iters):
        r1 = rng.random(X.shape)
        r2 = rng.random(X.shape)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = clamp(X + V, lb, ub)
        fit = env.evaluate_batch(X)
        worst = np.argmax(fit)
        X[worst] = gbest.copy()
        fit[worst] = gbest_fit
        improve = fit < pbest_fit
        pbest[improve] = X[improve]
        pbest_fit[improve] = fit[improve]
        if fit.min() < gbest_fit:
            gbest_fit = fit.min()
            gbest = X[np.argmin(fit)].copy()
        history.append(gbest_fit)
    return gbest, gbest_fit, history


def run_gwo(env, n_particles=30, iters=120, seed=0):
    rng = np.random.default_rng(seed)
    lb, ub = env.lower_bounds(), env.upper_bounds()
    X = env.init_population(n_particles, rng=rng)
    fit = env.evaluate_batch(X)
    history = []

    def top3(X, fit):
        idx = np.argsort(fit)[:3]
        return X[idx[0]].copy(), X[idx[1]].copy(), X[idx[2]].copy(), fit[idx[0]]

    alpha, beta, delta, best_fit = top3(X, fit)
    history.append(best_fit)

    for t in range(iters):
        a = 2 - 2 * t / iters
        for i in range(n_particles):
            for leader in (alpha, beta, delta):
                pass
            A1 = 2 * a * rng.random(env.dim) - a
            C1 = 2 * rng.random(env.dim)
            D_alpha = np.abs(C1 * alpha - X[i])
            X1 = alpha - A1 * D_alpha

            A2 = 2 * a * rng.random(env.dim) - a
            C2 = 2 * rng.random(env.dim)
            D_beta = np.abs(C2 * beta - X[i])
            X2 = beta - A2 * D_beta

            A3 = 2 * a * rng.random(env.dim) - a
            C3 = 2 * rng.random(env.dim)
            D_delta = np.abs(C3 * delta - X[i])
            X3 = delta - A3 * D_delta

            X[i] = clamp((X1 + X2 + X3) / 3.0, lb, ub)
        fit = env.evaluate_batch(X)
        alpha, beta, delta, best_fit = top3(X, fit)
        history.append(best_fit)
    return alpha, best_fit, history


def run_woa(env, n_particles=30, iters=120, seed=0, b=1.0):
    rng = np.random.default_rng(seed)
    lb, ub = env.lower_bounds(), env.upper_bounds()
    X = env.init_population(n_particles, rng=rng)
    fit = env.evaluate_batch(X)
    best_idx = np.argmin(fit)
    best = X[best_idx].copy()
    best_fit = fit[best_idx]
    history = [best_fit]

    for t in range(iters):
        a = 2 - 2 * t / iters
        for i in range(n_particles):
            r = rng.random()
            A = 2 * a * r - a
            C = 2 * rng.random()
            p = rng.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * best - X[i])
                    X[i] = best - A * D
                else:
                    rand_idx = rng.integers(0, n_particles)
                    D = np.abs(C * X[rand_idx] - X[i])
                    X[i] = X[rand_idx] - A * D
            else:
                l = rng.uniform(-1, 1, env.dim)
                D = np.abs(best - X[i])
                X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best
        X = clamp(X, lb, ub)
        fit = env.evaluate_batch(X)
        if fit.min() < best_fit:
            best_fit = fit.min()
            best = X[np.argmin(fit)].copy()
        history.append(best_fit)
    return best, best_fit, history
