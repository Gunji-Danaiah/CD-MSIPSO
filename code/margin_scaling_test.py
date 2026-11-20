import sys, time, json
import os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scalability_experiment import make_scalable_env
from cdmsipso import run_cdmsipso
from verify import exact_report

N_SEEDS = 12
BASE_MARGIN = 9.0

results = []
for n_uav in [3, 4, 5, 6]:
    for label, margin_fn in [
        ("fixed_m9", lambda n: BASE_MARGIN),
        ("sqrt_scale", lambda n: BASE_MARGIN * np.sqrt(n / 3.0)),
        ("linear_scale", lambda n: BASE_MARGIN * (n / 3.0)),
    ]:
        env = make_scalable_env(n_uav)  # fresh env each condition: resets env.rng so seeds are truly comparable
        m = margin_fn(n_uav)
        fits, feas, times = [], [], []
        for seed in range(N_SEEDS):
            t0 = time.time()
            x, f, _ = run_cdmsipso(env, n_particles=30, iters=80, seed=seed, repulsion_margin=m)
            dt = time.time() - t0
            fits.append(f)
            feas.append(exact_report(env, x)['feasible_exact'])
            times.append(dt)
        results.append(dict(n_uav=n_uav, condition=label, margin=m,
                             mean_fitness=float(np.mean(fits)), std_fitness=float(np.std(fits)),
                             feasibility_rate=float(np.mean(feas)), mean_time=float(np.mean(times))))
        print(f"n_uav={n_uav} {label:12s} (m={m:.2f}): mean_fit={np.mean(fits):9.2f} "
              f"feas={np.mean(feas)*100:5.1f}%  t={np.mean(times):.2f}s", flush=True)

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(os.path.join(HERE, '..', 'results', 'margin_scaling_test_v3.csv'), index=False)
print("\nSaved margin_scaling_test.csv")
