"""Tables for the 200-run confirmatory comparison, exact-in-loop optimisation, operator diagnostics and p_gwo sweep."""
import os, glob, json
import numpy as np, pandas as pd
from scipy import stats
from analyze_revision import load, wilson, holm, mcnemar_exact
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results'); T = os.path.join(HERE, '..', 'paper', 'tables')
d = load(); d.to_csv(os.path.join(R, 'revision_all_runs.csv'), index=False)
ENVS = [('E1_CrossingTraffic', 'E1'), ('E2_DenseField', 'E2'), ('E3R_FastDynamicThreats_Revised', 'E3R')]
ARMS = ['GWO', 'MSIPSO', 'REP', 'MUT', 'CD-MSIPSO']; NAME = {'CD-MSIPSO': 'CD-MSIPSO'}
N = 200; rng = np.random.default_rng(0); out = {}

def sub(env, a, n=N): x = d[(d.env == env) & (d.algo == a) & (d.run < n)].sort_values('run'); return x
def pct(k, n): lo, hi = wilson(k, n); return f'{100*k/n:.0f} [{100*lo:.0f},{100*hi:.0f}]'
def rbc(diff):   # matched-pairs rank-biserial (negative: first lower)
    nz = diff[diff != 0]
    if len(nz) == 0: return 0.0
    r = stats.rankdata(np.abs(nz)); wp = r[nz > 0].sum(); wm = r[nz < 0].sum(); return (wp - wm) / (wp + wm)
def boot(diff): b = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]; return np.percentile(b, [2.5, 97.5])
def tost(diff, margin):   # paired TOST, equivalence if |mean diff| < margin
    n = len(diff); m = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((m + margin) / se, n - 1); p2 = stats.t.cdf((m - margin) / se, n - 1); return max(p1, p2)

# ---- Table: 200-run comparison
rows = []
for env, tag in ENVS:
    for a in ARMS:
        x = sub(env, a); n = len(x); assert n == N, (env, a, n)
        rows.append(dict(env=tag, algo=a, n=n, fit=x.best_fitness.mean(), sd=x.best_fitness.std(), f25=100*x.feas25.mean(), fex=pct(int(x.feas_exact.sum()), n),
                         path=x.path_length.mean()))
t1 = pd.DataFrame(rows); t1.to_csv(os.path.join(R, 'table_ext200.csv'), index=False)
L = [r'\begin{tabular}{llrrrr}', r'\toprule', r'Env & Algorithm & Fitness (mean $\pm$ SD) & Path len. & Feas.\ 25 (\%) & Feas.\ exact (\%) [95\% CI]\\', r'\midrule']
for env, tag in ENVS:
    for i, a in enumerate(ARMS):
        r = t1[(t1.env == tag) & (t1.algo == a)].iloc[0]
        L.append(f"{tag if i == 0 else ''} & {a} & {r.fit:.0f} $\\pm$ {r.sd:.0f} & {r.path:.0f} & {r.f25:.0f} & {r.fex}\\\\")
    L.append(r'\midrule')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
open(os.path.join(T, 'tab_ext200.tex'), 'w').write('\n'.join(L))

# ---- Table: paired tests, Holm per environment and metric (family: 7 contrasts)
CON = [('REP', 'MSIPSO'), ('MUT', 'MSIPSO'), ('CD-MSIPSO', 'MSIPSO'), ('CD-MSIPSO', 'REP'), ('CD-MSIPSO', 'MUT'),
       ('GWO', 'MSIPSO'), ('GWO', 'REP'), ('GWO', 'MUT'), ('GWO', 'CD-MSIPSO')]
rows = []
for env, tag in ENVS:
    res = []
    for A, B in CON:
        xa, xb = sub(env, A), sub(env, B); diff = xa.best_fitness.values - xb.best_fitness.values
        pw = stats.wilcoxon(diff).pvalue; lo, hi = boot(diff); pm, n10, n01 = mcnemar_exact(xa.feas_exact.values.astype(bool), xb.feas_exact.values.astype(bool))
        fd = xa.feas_exact.values.astype(float) - xb.feas_exact.values.astype(float)
        marg = 0.05 * xb.best_fitness.mean(); pt = tost(diff, marg)
        res.append(dict(env=tag, A=A, B=B, dmean=diff.mean(), lo=lo, hi=hi, rbc=rbc(diff), p_w=pw, dfeas=100*fd.mean(), flo=100*np.percentile([rng.choice(fd, len(fd)).mean() for _ in range(3000)], 2.5),
                        fhi=100*np.percentile([rng.choice(fd, len(fd)).mean() for _ in range(3000)], 97.5), p_m=pm, tost=pt, margin=marg, n10=n10, n01=n01))
    df = pd.DataFrame(res); df['p_w_adj'] = holm(df.p_w.values); df['p_m_adj'] = holm(df.p_m.values); rows.append(df)
t2 = pd.concat(rows); t2.to_csv(os.path.join(R, 'table_ext200_tests.csv'), index=False)
def pf(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
L = [r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Env & Contrast & $\Delta$fit [95\% CI] & $r_{rb}$ & $p_{\rm adj}$ & $\Delta$feas (pp) & $p_{\rm adj}$ & TOST $p$\\', r'\midrule']
for env, tag in ENVS:
    s = t2[t2.env == tag]
    for i, (_, r) in enumerate(s.iterrows()):
        L.append(f"{tag if i == 0 else ''} & {r.A} $-$ {r.B} & {r.dmean:.0f} [{r.lo:.0f},{r.hi:.0f}] & {r.rbc:.2f} & {pf(r.p_w_adj)} & {r.dfeas:+.1f} & {pf(r.p_m_adj)} & {pf(r.tost) if r.A != 'GWO' else '--'}\\\\")
    L.append(r'\midrule')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
open(os.path.join(T, 'tab_ext200_tests.tex'), 'w').write('\n'.join(L))

# ---- minimum detectable effect (paired, alpha=0.05/3 for the three operator-vs-base contrasts, power 0.8)
za = stats.norm.ppf(1 - 0.05 / 6); zb = stats.norm.ppf(0.8); mde = {}
for env, tag in ENVS:
    s = t2[(t2.env == tag) & (t2.B == 'MSIPSO') & (t2.A.isin(['REP', 'MUT', 'CD-MSIPSO']))]
    diffs = [sub(env, a).best_fitness.values - sub(env, 'MSIPSO').best_fitness.values for a in ['REP', 'MUT', 'CD-MSIPSO']]
    sdd = np.mean([x.std(ddof=1) for x in diffs]); mde[tag] = dict(sd_diff=sdd, mde_fit=(za + zb) * sdd / np.sqrt(N), mde_rel=100 * (za + zb) * sdd / np.sqrt(N) / sub(env, 'MSIPSO').best_fitness.mean())
out['mde'] = mde

# ---- exact-in-loop
ex = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'followup_exactloop_*.csv'))])
rows = []; L = [r'\begin{tabular}{llrrrr}', r'\toprule', r'Env & Algorithm & Fitness & Path len. & Feas.\ 25 (\%) & Feas.\ exact (\%) [95\% CI]\\', r'\midrule']
for env, tag in ENVS:
    for i, a in enumerate(['GWO', 'MSIPSO', 'CD-MSIPSO']):
        x = ex[(ex.env == env) & (ex.algo == a)]; n = len(x); assert n == 30
        L.append(f"{tag if i == 0 else ''} & {a} & {x.best_fitness.mean():.0f} & {x.path_length.mean():.0f} & {100*x.feas25.mean():.0f} & {pct(int(x.feas_exact.sum()), n)}\\\\")
    L.append(r'\midrule')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_exactloop.tex'), 'w').write('\n'.join(L))
ps = []
for env, tag in ENVS:
    g = ex[(ex.env == env) & (ex.algo == 'GWO')].sort_values('run'); c = ex[(ex.env == env) & (ex.algo == 'CD-MSIPSO')].sort_values('run'); m = ex[(ex.env == env) & (ex.algo == 'MSIPSO')].sort_values('run')
    ps.append(dict(env=tag, p_gwo_vs_cd=mcnemar_exact(g.feas_exact.values.astype(bool), c.feas_exact.values.astype(bool))[0], p_cd_vs_msipso=mcnemar_exact(c.feas_exact.values.astype(bool), m.feas_exact.values.astype(bool))[0]))
out['exactloop_tests'] = ps

# ---- operator diagnostics
op = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'followup_opstats_*.csv'))])
L = [r'\begin{tabular}{llrrrrr}', r'\toprule', r'Env & Arm & Repulsion active (\%) & Mean disp.\ (u) & Mutated wp.\ (\%) & Fitness & Feas.\ exact (\%)\\', r'\midrule']
ops = {}
for env, tag in ENVS:
    for i, a in enumerate(['REP', 'MUT', 'CD-MSIPSO', 'CD-MSIPSO-strong']):
        x = op[(op.env == env) & (op.algo == a)]; n = len(x)
        ra = 100 * x.rep_active.sum() / x.rep_wp.sum() if x.rep_wp.sum() else np.nan
        rd = x.rep_disp.sum() / x.rep_active.sum() if x.rep_active.sum() else np.nan
        ma = 100 * x.mut_active.sum() / x.mut_wp.sum() if x.mut_wp.sum() else np.nan
        ops[f'{tag}_{a}'] = dict(rep_active=ra, rep_disp=rd, mut_active=ma, fit=x.best_fitness.mean(), fex=100*x.feas_exact.mean(), n=n)
        fm = lambda v, f: '--' if np.isnan(v) else f.format(v)
        L.append(f"{tag if i == 0 else ''} & {a} & {fm(ra,'{:.1f}')} & {fm(rd,'{:.2f}')} & {fm(ma,'{:.2f}')} & {x.best_fitness.mean():.0f} & {pct(int(x.feas_exact.sum()), n)}\\\\")
    L.append(r'\midrule')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_opstats.tex'), 'w').write('\n'.join(L)); out['ops'] = ops
base = {a: d[(d.algo == a) & (d.run < 20)].groupby('env').feas_exact.mean().to_dict() for a in ['MSIPSO', 'CD-MSIPSO']}

# ---- p_gwo sweep with exact feasibility
pg = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'followup_pgwo_*.csv'))]); pg['p'] = pg.algo.str.replace('pgwo=', '').astype(float)
L = [r'\begin{tabular}{lrrrrrr}', r'\toprule', r'$p_{\rm gwo}$ & E1 fit & E1 feas.\ (\%) & E2 fit & E2 feas.\ (\%) & E3R fit & E3R feas.\ (\%)\\', r'\midrule']
pgs = {}
for p in sorted(pg.p.unique()):
    cells = []
    for env, tag in ENVS:
        x = pg[(pg.env == env) & (pg.p == p)]; cells += [f'{x.best_fitness.mean():.0f}', f'{100*x.feas_exact.mean():.0f}']; pgs[f'{tag}_{p}'] = (x.best_fitness.mean(), 100*x.feas_exact.mean())
    L.append(f'{p:.2f} & ' + ' & '.join(cells) + r'\\')
L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(T, 'tab_pgwo.tex'), 'w').write('\n'.join(L)); out['pgwo'] = pgs

# ---- larger-budget confirmation (100 paired runs): MSIPSO vs CD-MSIPSO
cfgs = [('E1', 'I=300', '30_300_env0'), ('E2', 'I=300', '30_300'), ('E3R', 'I=300', '30_300_env3'), ('E2', '$n_p$=90', '90_100')]
rows = []; pm = []
for tag, lab, f in cfgs:
    m = pd.read_csv(os.path.join(R, f'budget_confirm_{f.split("_env")[0] if False else f}_MSIPSO.csv')).sort_values('run'); c = pd.read_csv(os.path.join(R, f'budget_confirm_{f}_CD-MSIPSO.csv')).sort_values('run')
    p, n10, n01 = mcnemar_exact(c.feas_exact.values.astype(bool), m.feas_exact.values.astype(bool)); pm.append(p)
    rows.append((tag, lab, m, c, p, stats.wilcoxon(c.best_fitness.values - m.best_fitness.values).pvalue))
adj = holm(pm); L = [r'\begin{tabular}{llrrrrr}', r'\toprule', r'Env & Budget & MSIPSO fit/feas.\ (\%) & CD-MSIPSO fit/feas.\ (\%) & $p$ fit & $p$ feas. & $p_{\rm Holm}$ feas.\\', r'\midrule']
bc = {}
for (tag, lab, m, c, p, pf_), pa in zip(rows, adj):
    L.append(f"{tag} & {lab} & {m.best_fitness.mean():.0f} / {100*m.feas_exact.mean():.0f} & {c.best_fitness.mean():.0f} / {100*c.feas_exact.mean():.0f} & {pf_:.2f} & {p:.3f} & {pa:.3f}\\\\")
    bc[f'{tag}_{lab}'] = dict(msipso=100*m.feas_exact.mean(), cd=100*c.feas_exact.mean(), p=p, p_holm=pa)
L += [r'\bottomrule', r'\end{tabular}']; open(os.path.join(T, 'tab_budgetconfirm.tex'), 'w').write('\n'.join(L)); out['budgetconfirm'] = bc
json.dump(out, open(os.path.join(R, 'numbers_ext.json'), 'w'), indent=1, default=float)
print(t1.to_string()); print(t2[['env','A','B','dmean','rbc','p_w_adj','dfeas','p_m_adj','tost','margin']].round(3).to_string()); print(json.dumps(out, indent=0, default=float)[:3500])
