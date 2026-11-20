"""E2 with larger budgets, MSIPSO vs CD-MSIPSO, 100 paired runs (confirms the 10-run budget table).
usage: python budget_confirm.py NP ITERS [ARM] [ENV_INDEX, default 1 = E2]"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pandas as pd
from functools import partial
from environment import make_environments
from cdmsipso import run_variant
from run_revision_experiments import evaluate_solution
npart, iters = int(sys.argv[1]), int(sys.argv[2]); gi = int(sys.argv[4]) if len(sys.argv) > 4 else 1; env = make_environments()[gi]; rows = []
only = sys.argv[3] if len(sys.argv) > 3 else None
for name, fn in (('MSIPSO', partial(run_variant, rep=False, mut=False)), ('CD-MSIPSO', partial(run_variant, rep=True, mut=True))):
    if only and name != only: continue
    for run in range(100):
        x, f, _ = fn(env, n_particles=npart, iters=iters, seed=1000 * gi + run)
        r = dict(np=npart, iters=iters, algo=name, run=run, best_fitness=float(f)); r.update(evaluate_solution(env, x)); rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, '..', 'results', f'budget_confirm_{npart}_{iters}' + (f'_env{gi}' if gi != 1 else '') + (f'_{only}' if only else '') + '.csv'), index=False)
