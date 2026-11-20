"""
robustness_experiments.py -- additional experiments requested in review.
  group finegrid : optimise with 100 (instead of 25) time samples
  group robust   : E1 with different obstacle speed / UAV separation / penalty weights
  group budget   : E2 with different swarm size and iteration budget
usage: python robustness_experiments.py GROUP [N_RUNS]
"""
import sys, os, time, copy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from functools import partial
from environment import make_environments
from baselines import run_gwo
from cdmsipso import run_variant
from run_revision_experiments import evaluate_solution

ARMS = {'GWO': run_gwo,
        'MSIPSO': partial(run_variant, rep=False, mut=False),
        'CD-MSIPSO': partial(run_variant, rep=True, mut=True)}

def variant(base, speed=1.0, uav_safety=None, wscale=1.0, n_samples=None):
    e = copy.deepcopy(base)
    e.dynamic_obs = e.dynamic_obs.copy(); e.dynamic_obs[:, 4:7] *= speed
    if uav_safety is not None: e.safety_uav = uav_safety
    for k in ('stat', 'dyn', 'coop'): e.weights[k] = e.weights[k] * wscale
    if n_samples: e.n_samples = n_samples
    return e

def densify(base, extra, seed=7):
    """Add `extra` random static obstacles (r in [5,7]) to a scenario, rejecting positions that conflict
    with fixed start/goal positions, with moving-obstacle start/end positions, or with existing obstacles."""
    rng = np.random.default_rng(seed); e = copy.deepcopy(base); obs = [tuple(o) for o in e.static_obs]
    pts = list(e.starts) + list(e.goals) + [d[:3] for d in e.dynamic_obs] + [d[:3] + d[4:7] for d in e.dynamic_obs]
    while len(obs) < len(base.static_obs) + extra:
        c = (rng.uniform(15, 85), rng.uniform(15, 85), rng.uniform(15, 30), rng.uniform(5, 7))
        if all(np.linalg.norm(np.array(c[:3]) - np.array(p)) > c[3] + e.safety_static + 6 for p in pts) and \
           all(np.linalg.norm(np.array(c[:3]) - np.array(o[:3])) > c[3] + o[3] + 6 for o in obs):
            obs.append(c)
    e.static_obs = np.array(obs, float); return e

def configs(group):
    E = make_environments(); E1, E2, E3R = E[0], E[1], E[3]
    c = []   # (label, base_index, env, n_particles, iters)
    if group == 'finegrid':
        for gi, b in ((0, E1), (1, E2), (3, E3R)):
            c.append((f'samples=100', gi, variant(b, n_samples=100), 30, 100))
    elif group == 'robust':
        c += [('speed x0.5', 0, variant(E1, speed=0.5), 30, 100), ('speed x1.5', 0, variant(E1, speed=1.5), 30, 100),
              ('UAV sep 3', 0, variant(E1, uav_safety=3.0), 30, 100), ('UAV sep 8', 0, variant(E1, uav_safety=8.0), 30, 100),
              ('penalty w x0.5', 0, variant(E1, wscale=0.5), 30, 100), ('penalty w x2', 0, variant(E1, wscale=2.0), 30, 100)]
    elif group == 'density':
        for extra in (4, 8):
            c.append((f'+{extra} static', 0, densify(E1, extra), 30, 100))
    elif group == 'budget':
        c += [('N=30,T=100', 1, variant(E2), 30, 100), ('N=90,T=100', 1, variant(E2), 90, 100), ('N=30,T=300', 1, variant(E2), 30, 300)]
    return c

if __name__ == '__main__':
    group = sys.argv[1]; n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rows = []; t0 = time.time()
    for label, gi, env, npart, iters in configs(group):
        for name, fn in ARMS.items():
            for run in range(n_runs):
                seed = 1000 * gi + run
                st = time.perf_counter()
                x, f, _ = fn(env, n_particles=npart, iters=iters, seed=seed)
                dt = time.perf_counter() - st
                r = dict(group=group, config=label, env=env.name, algo=name, run=run, n_particles=npart, iters=iters,
                         best_fitness=float(f), time_sec=dt)
                # verification always uses the ORIGINAL 25 vs exact; env may differ (speed, safety)
                r.update(evaluate_solution(env, x)); rows.append(r)
            print(f'{label:16s}{name:10s} {time.time()-t0:6.0f}s', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(HERE, '..', 'results', f'robust_{group}.csv'), index=False)
