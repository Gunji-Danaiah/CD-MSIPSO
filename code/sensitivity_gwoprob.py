"""Post-hoc sensitivity of CD-MSIPSO to the GWO-branch probability p_gwo.
NOTE: p_gwo = 0.85 was fixed during development, BEFORE this sweep; the sweep is a
post-hoc robustness check, not the selection procedure.   usage: python sensitivity_gwoprob.py ENV_INDEX"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pandas as pd
from environment import make_environments
from cdmsipso import run_variant
from verify import exact_report
gi = int(sys.argv[1]); env = make_environments()[gi]; rows = []
for p in [0.3, 0.5, 0.7, 0.85, 0.95, 1.0]:
    for run in range(20):
        x, f, _ = run_variant(env, 30, 100, 1000 * gi + run, gwo_prob=p)
        rows.append(dict(env=env.name, p_gwo=p, run=run, fitness=float(f), feas25=bool(env.is_feasible(x)), feas_exact=exact_report(env, x)['feasible_exact']))
    print(env.name, p, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, '..', 'results', f'sensitivity_pgwo_env{gi}.csv'), index=False)
