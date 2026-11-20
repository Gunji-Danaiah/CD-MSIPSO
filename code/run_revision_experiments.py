"""
run_revision_experiments.py -- full comparison + factorial ablation + independent
safety verification.

usage: python run_revision_experiments.py ENV_INDEX [RUN_START RUN_END] [OUT_TAG]

Arms
  PSO, GWO, WOA                 baselines
  MSIPSO                        no proposed operator
  REP                           cooperative repulsion only      (index-matched)
  MUT                           dynamic-aware mutation only     (index-matched)
  CD-MSIPSO                     both operators                  (index-matched)
  TC-REP, TC-MUT, CD-MSIPSO-TC  time-consistent versions
All arms with the same run id use the same seed and therefore the SAME initial
population (the first random draw of every optimiser is env.init_population(rng)),
so results are paired by run.
"""
import sys, time, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from functools import partial
from environment import make_environments
from baselines import run_pso, run_gwo, run_woa
from cdmsipso import run_variant
from verify import exact_report, sampled_feasible

ARMS = {
    'PSO': run_pso, 'GWO': run_gwo, 'WOA': run_woa,
    'MSIPSO':        partial(run_variant, rep=False, mut=False),
    'REP':           partial(run_variant, rep=True,  mut=False),
    'MUT':           partial(run_variant, rep=False, mut=True),
    'CD-MSIPSO':     partial(run_variant, rep=True,  mut=True),
    'TC-REP':        partial(run_variant, rep=True,  mut=False, tc=True),
    'TC-MUT':        partial(run_variant, rep=False, mut=True,  tc=True),
    'CD-MSIPSO-TC':  partial(run_variant, rep=True,  mut=True,  tc=True),
}
N_PARTICLES, ITERS = 30, 100

def evaluate_solution(env, x, prefix=''):
    c = env.components(x)
    rep = exact_report(env, x)
    d = {f'{prefix}J_len': c['len'], f'{prefix}J_stat': c['stat'], f'{prefix}J_dyn': c['dyn'],
         f'{prefix}J_coop': c['coop'], f'{prefix}J_smooth': c['smooth'],
         f'{prefix}path_length': env.path_length(x),
         f'{prefix}feas25': env.is_feasible(x),
         f'{prefix}feas100': sampled_feasible(env, x, 100),
         f'{prefix}feas1000': sampled_feasible(env, x, 1000),
         f'{prefix}feas_exact': rep['feasible_exact'],
         f'{prefix}min_static_margin': rep['min_static_margin'],
         f'{prefix}min_dyn_margin': rep['min_dyn_margin'],
         f'{prefix}min_uav_sep': rep['min_uav_sep']}
    return d

if __name__ == '__main__':
    gi = int(sys.argv[1])
    r0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    r1 = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    tag = sys.argv[4] if len(sys.argv) > 4 else f'env{gi}'
    only = sys.argv[5].split(',') if len(sys.argv) > 5 else list(ARMS)
    env = make_environments()[gi]
    rows, hist = [], {}
    os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
    t_start = time.time()
    for name in only:
        fn = ARMS[name]
        hl = []
        for run in range(r0, r1):
            seed = 1000 * gi + run
            env.start_archive()
            t0 = time.perf_counter()
            x, f, h = fn(env, n_particles=N_PARTICLES, iters=ITERS, seed=seed)
            dt = time.perf_counter() - t0 - env.arch_time      # archive bookkeeping excluded
            row = dict(env=env.name, algo=name, run=run, seed=seed, best_fitness=float(f),
                       time_sec=dt, n_evals=env.n_evals)
            row.update(evaluate_solution(env, x))
            bx, bf = env.get_archive()
            row['bf_found'] = bx is not None
            if bx is not None:
                row['bf_fitness'] = bf
                row.update(evaluate_solution(env, bx, 'bf_'))
            rows.append(row); hl.append([float(v) for v in h])
        hist[f'{env.name}__{name}'] = hl
        print(f'{env.name:24s} {name:13s} elapsed {time.time()-t_start:6.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(HERE, '..', 'results', f'revision_{tag}.csv'), index=False)
    json.dump(hist, open(os.path.join(HERE, '..', 'results', f'revision_hist_{tag}.json'), 'w'))
