"""
followup_experiments.py
  exactloop : optimise with the exact continuous-time clearance penalty inside the fitness (GWO, MSIPSO, CD-MSIPSO)
  opstats   : operator activation statistics (default operators) and a 'strong operator' test
  pgwo      : p_gwo sweep including 0.0 (pure PSO branch) with exact feasibility
usage: python followup_experiments.py GROUP ENV_INDEX N_RUNS
"""
import sys, os, time, copy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from functools import partial
from environment import make_environments
from baselines import run_gwo
import cdmsipso
from cdmsipso import run_variant, OP_STATS
from run_revision_experiments import evaluate_solution

def main(group, gi, n_runs, only=None):
    env = make_environments()[gi]; rows = []
    if group == 'exactloop':
        env = copy.deepcopy(env); env.exact_fitness = True
        arms = {'GWO': run_gwo, 'MSIPSO': partial(run_variant, rep=False, mut=False),
                'CD-MSIPSO': partial(run_variant, rep=True, mut=True)}
    elif group == 'opstats':
        arms = {'MSIPSO': partial(run_variant, rep=False, mut=False), 'REP': partial(run_variant, rep=True, mut=False),
                'MUT': partial(run_variant, rep=False, mut=True), 'CD-MSIPSO': partial(run_variant, rep=True, mut=True),
                'CD-MSIPSO-strong': partial(run_variant, rep=True, mut=True, rep_strength=1.8, mut_rate=0.5, mut_strength=0.3, mut_eps=6.0)}
    elif group == 'pgwo':
        arms = {f'pgwo={p}': partial(run_variant, rep=True, mut=True, gwo_prob=p) for p in (0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0)}
    for name, fn in arms.items():
        if only and name != only: continue
        for run in range(n_runs):
            seed = 1000 * gi + run
            t0 = time.perf_counter()
            x, f, _ = fn(env, n_particles=30, iters=100, seed=seed)
            r = dict(group=group, env=env.name, algo=name, run=run, best_fitness=float(f), time_sec=time.perf_counter() - t0)
            if name != 'GWO': r.update({k: v for k, v in OP_STATS.items()})
            ev = copy.copy(env); ev.exact_fitness = False
            r.update(evaluate_solution(ev, x)); rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(HERE, '..', 'results', f'followup_{group}_{gi}' + (f'_{only}' if only else '') + '.csv'), index=False)
        print(group, gi, name, flush=True)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
