import sys, time, json
import os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from environment import Environment
from baselines import run_gwo
from cdmsipso import run_cdmsipso, run_msipso_plain
from verify import exact_report

def make_scalable_env(n_uav, seed=42):
    rng = np.random.default_rng(seed)
    # place starts on a circle, goals diametrically opposite -> forces crossing paths
    R = 42
    center = 50
    angles = np.linspace(0, 2 * np.pi, n_uav, endpoint=False)
    starts = [[center + R * np.cos(a), center + R * np.sin(a), 15 + 5 * (i % 3)] for i, a in enumerate(angles)]
    goals = [[center + R * np.cos(a + np.pi), center + R * np.sin(a + np.pi), 15 + 5 * (i % 3)] for i, a in enumerate(angles)]
    static_obs = [(50, 50, 22, 10)]
    n_extra = min(4, n_uav)
    for i in range(n_extra):
        a = 2 * np.pi * i / n_extra
        static_obs.append((center + 20 * np.cos(a), center + 20 * np.sin(a), 20, 5))
    # obstacle starts at (10,10,20) so that no UAV start lies inside its safety envelope at t=0
    # for any fleet size 2..6 (checked with verify.fixed_endpoint_conflicts)
    dynamic_obs = [(10, 10, 20, 4, 40, 40, 0)]
    return Environment(f"Scalability_N{n_uav}", n_uav, starts, goals, static_obs, dynamic_obs, seed=seed)


if __name__ == "__main__":
    import pandas as pd
    N_SEEDS = 12
    results = []
    for n_uav in [2, 3, 4, 5, 6]:
        env = make_scalable_env(n_uav)
        for algo_name, fn in [("GWO", run_gwo), ("MSIPSO", run_msipso_plain), ("CD-MSIPSO", run_cdmsipso)]:
            for seed in range(N_SEEDS):
                t0 = time.time()
                x, f, _ = fn(env, n_particles=30, iters=80, seed=seed)
                dt = time.time() - t0
                rep = exact_report(env, x)
                results.append(dict(n_uav=n_uav, algo=algo_name, seed=seed, fitness=float(f), time_sec=float(dt),
                                    feas25=bool(env.is_feasible(x)), feas_exact=rep['feasible_exact'],
                                    min_uav_sep=rep['min_uav_sep']))
        print(f"n_uav={n_uav} done", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(HERE, '..', 'results', 'scalability_results_v3.csv'), index=False)
    print(df.groupby(['n_uav', 'algo']).agg(mean_fit=('fitness', 'mean'), feas25=('feas25', 'mean'),
                                             feas_exact=('feas_exact', 'mean')).round(3))
