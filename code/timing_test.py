import time, sys
sys.path.insert(0, '/home/claude/cdmsipso/code')
from environment import make_environments
from baselines import run_pso, run_gwo, run_woa
from cdmsipso import run_msipso_plain, run_cdmsipso

envs = make_environments()
env = envs[0]

t0 = time.time()
run_pso(env, n_particles=30, iters=120, seed=0)
print("PSO:", time.time() - t0)

t0 = time.time()
run_gwo(env, n_particles=30, iters=120, seed=0)
print("GWO:", time.time() - t0)

t0 = time.time()
run_woa(env, n_particles=30, iters=120, seed=0)
print("WOA:", time.time() - t0)

t0 = time.time()
run_msipso_plain(env, n_particles=30, iters=120, seed=0)
print("MSIPSO:", time.time() - t0)

t0 = time.time()
run_cdmsipso(env, n_particles=30, iters=120, seed=0)
print("CD-MSIPSO:", time.time() - t0)
