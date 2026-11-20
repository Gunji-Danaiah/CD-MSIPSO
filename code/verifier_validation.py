"""Validate verify.exact_report against a brute-force dense-sampling computation that does not share its code
(own arc-length interpolation, 200,001 time points). Solutions: random initial candidates (mostly infeasible),
candidates from short optimisation runs, and final solutions of full runs (mix of feasible and infeasible)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from environment import make_environments, N_WP
from verify import exact_report
from cdmsipso import run_variant

def dense(env, x, M=200001):
    t = np.linspace(0, 1, M); wp = x.reshape(env.n_uav, N_WP, 3); pos = []
    for u in range(env.n_uav):
        P = np.vstack([env.starts[u], wp[u], env.goals[u]]); L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        s = t * L[-1]; i = np.clip(np.searchsorted(L, s, side='right') - 1, 0, len(P) - 2)
        w = (s - L[i]) / np.maximum(L[i + 1] - L[i], 1e-12); pos.append(P[i] + (P[i + 1] - P[i]) * w[:, None])
    ms = min(np.min(np.linalg.norm(p - c[:3], axis=1) - (c[3] + env.safety_static)) for p in pos for c in env.static_obs)
    md = min(np.min(np.linalg.norm(p - (c[:3] + c[4:7] * t[:, None]), axis=1) - (c[3] + env.safety_dynamic)) for p in pos for c in env.dynamic_obs)
    su = min(np.min(np.linalg.norm(pos[i] - pos[j], axis=1)) for i in range(env.n_uav) for j in range(i + 1, env.n_uav))
    return ms, md, su

def main():
    rng = np.random.default_rng(123); E = make_environments(); res = []
    for gi in (0, 1, 3):
        env = E[gi]; xs = list(env.init_population(40, rng=rng))
        for s in range(10):
            x, f, _ = run_variant(env, 30, 25, seed=500 + s); xs.append(x)
        for s in range(20):
            x, f, _ = run_variant(env, 30, 100, seed=gi * 1000 + s); xs.append(x)
        for x in xs:
            r = exact_report(env, x); ms, md, su = dense(env, x)
            fd = (ms >= -1e-6) and (md >= -1e-6) and (su >= env.safety_uav - 1e-6)
            res.append(dict(env=gi, exact_feas=bool(r['feasible_exact']), dense_feas=bool(fd), d_static=ms - r['min_static_margin'], d_dyn=md - r['min_dyn_margin'], d_sep=su - r['min_uav_sep'],
                            feas_exact_margin=min(r['min_static_margin'], r['min_dyn_margin'], r['min_uav_sep'] - env.safety_uav)))
    n = len(res); mism = [r for r in res if r['exact_feas'] != r['dense_feas']]
    out = dict(n=n, n_feasible=int(sum(r['exact_feas'] for r in res)), n_infeasible=int(sum(not r['exact_feas'] for r in res)), mismatches=len(mism),
               max_abs_diff=float(max(max(abs(r['d_static']), abs(r['d_dyn']), abs(r['d_sep'])) for r in res)),
               min_signed_diff=float(min(min(r['d_static'], r['d_dyn'], r['d_sep']) for r in res)),
               mismatch_margins=[r['feas_exact_margin'] for r in mism])
    json.dump(out, open(os.path.join(HERE, '..', 'results', 'verifier_validation.json'), 'w'), indent=1); print(out)

if __name__ == '__main__': main()
