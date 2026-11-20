"""Generates every table / number used in the revised manuscript from the raw per-run CSV files."""
import os, json
import numpy as np, pandas as pd
from scipy import stats
from analyze_revision import wilson, holm, mcnemar_exact
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results'); T = os.path.join(HERE, '..', 'paper', 'tables')
os.makedirs(T, exist_ok=True)
d = pd.read_csv(os.path.join(R, 'revision_all_runs.csv'))
ENVS = [('E1_CrossingTraffic', 'E1'), ('E2_DenseField', 'E2'), ('E3R_FastDynamicThreats_Revised', 'E3R')]
MAIN = ['PSO', 'GWO', 'WOA', 'MSIPSO', 'CD-MSIPSO']
OUT = {}

def sub(env, arm, n=None):
    x = d[(d.env == env) & (d.algo == arm)].sort_values('run')
    return x if n is None else x[x.run < n]

def pct(p): return f'{100*p:.0f}'
def ci(k, n):
    lo, hi = wilson(k, n); return f'[{100*lo:.0f},{100*hi:.0f}]'

# ---------------- Table: main comparison (n = 20 paired runs) ----------------
rows = []
for env, tag in ENVS:
    for a in MAIN:
        x = sub(env, a, 20); n = len(x); k = int(x.feas_exact.sum())
        rows.append(dict(env=tag, algo=a, n=n, fit=x.best_fitness.mean(), sd=x.best_fitness.std(), plen=x.path_length.mean(),
                         f25=x.feas25.mean(), fex=k/n, ci=ci(k, n), sep=x.min_uav_sep.median(), t=x.time_sec.mean()))
main = pd.DataFrame(rows); main.to_csv(os.path.join(R, 'table_main.csv'), index=False)
L = [r'\begin{tabular}{llrrrrrr}', r'\hline', r'Env & Algorithm & Fitness (mean $\pm$ SD) & Path len. & Feas$_{25}$ (\%) & Feas$_{\rm cont}$ (\%) [95\% CI] & $s_{\min}$ & Time (s) \\', r'\hline']
for env, tag in ENVS:
    m = main[main.env == tag]
    for i, r in enumerate(m.itertuples()):
        L.append(f'{tag if i==0 else ""} & {r.algo} & ${r.fit:.0f}\\pm{r.sd:.0f}$ & {r.plen:.0f} & {pct(r.f25)} & {pct(r.fex)} {r.ci} & {r.sep:.1f} & {r.t:.2f} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}')
open(os.path.join(T, 'tab_main.tex'), 'w').write('\n'.join(L))

# ---------------- paired statistics (n=20, Holm over the 10 pairs per environment) ----------------
def paired_rb(a, b):
    """matched-pairs rank-biserial correlation of (a-b); >0 means a larger."""
    diff = np.asarray(a) - np.asarray(b); diff = diff[diff != 0]
    if len(diff) == 0: return 0.0
    rk = stats.rankdata(np.abs(diff)); Wp = rk[diff > 0].sum(); Wm = rk[diff < 0].sum()
    return (Wp - Wm) / (Wp + Wm)
rows = []
from itertools import combinations
for env, tag in ENVS:
    fr = stats.friedmanchisquare(*[sub(env, a, 20).best_fitness.values for a in MAIN])
    OUT[f'friedman_{tag}'] = (float(fr.statistic), float(fr.pvalue))
    ps = []; recs = []
    for A, B in combinations(MAIN, 2):
        xa, xb = sub(env, A, 20), sub(env, B, 20)
        w = stats.wilcoxon(xa.best_fitness.values, xb.best_fitness.values)
        rb = paired_rb(xa.best_fitness.values, xb.best_fitness.values)
        pm, n10, n01 = mcnemar_exact(xa.feas_exact.values.astype(bool), xb.feas_exact.values.astype(bool))
        recs.append(dict(env=tag, A=A, B=B, mean_diff=float((xa.best_fitness.values - xb.best_fitness.values).mean()),
                         p_fit=float(w.pvalue), rb=rb, p_feas=pm, A_only=n10, B_only=n01)); ps.append(w.pvalue)
    adj = holm(ps); pf = holm([r['p_feas'] for r in recs])
    for r, a, b in zip(recs, adj, pf): r['p_fit_holm'] = float(a); r['p_feas_holm'] = float(b); rows.append(r)
st = pd.DataFrame(rows); st.to_csv(os.path.join(R, 'table_paired_stats.csv'), index=False)
L = [r'\begin{tabular}{llccc}', r'\hline', r'A & B & E1 & E2 & E3R \\', r'\hline']
for A, B in combinations(MAIN, 2):
    cells = []
    for env, tag in ENVS:
        r = st[(st.env == tag) & (st.A == A) & (st.B == B)].iloc[0]
        p = r.p_fit_holm; ptxt = '$<$0.001' if p < 0.001 else f'{p:.3f}'
        sign = '$-$' if r.mean_diff < 0 else '$+$'      # A lower (better) than B -> '-'
        cells.append(f'{ptxt} ({sign}) {"*" if p < 0.05 else ""}')
    L.append(f'{A} & {B} & ' + ' & '.join(cells) + r' \\')
L += [r'\hline', r'\end{tabular}']
open(os.path.join(T, 'tab_stats.tex'), 'w').write('\n'.join(L))

# ---------------- factorial ablation (n = 60) ----------------
ABL = ['MSIPSO', 'REP', 'MUT', 'CD-MSIPSO']
rows = []
for env, tag in ENVS:
    for a in ABL:
        x = sub(env, a); n = len(x); k = int(x.feas_exact.sum())
        rows.append(dict(env=tag, arm=a, n=n, fit=x.best_fitness.mean(), sd=x.best_fitness.std(), f25=x.feas25.mean(), fex=k/n, ci=ci(k, n),
                         sep=x.min_uav_sep.median(), dyn=x.min_dyn_margin.median(), plen=x.path_length.mean(), t=x.time_sec.mean()))
ab = pd.DataFrame(rows); ab.to_csv(os.path.join(R, 'table_ablation.csv'), index=False)
names = {'MSIPSO': 'Neither (MSIPSO)', 'REP': 'Repulsion only', 'MUT': 'Mutation only', 'CD-MSIPSO': 'Both (CD-MSIPSO)'}
L = [r'\begin{tabular}{llrrrrr}', r'\hline', r'Env & Variant & Fitness & Feas$_{25}$ & Feas$_{\rm cont}$ [95\% CI] & $s_{\min}$ & Time \\', r'\hline']
for env, tag in ENVS:
    for i, r in enumerate(ab[ab.env == tag].itertuples()):
        L.append(f'{tag if i==0 else ""} & {names[r.arm]} & ${r.fit:.0f}\\pm{r.sd:.0f}$ & {pct(r.f25)} & {pct(r.fex)} {r.ci} & {r.sep:.1f} & {r.t:.2f} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_ablation.tex'), 'w').write('\n'.join(L))

rows = []
pairs = [('MSIPSO', 'REP'), ('MSIPSO', 'MUT'), ('MSIPSO', 'CD-MSIPSO'), ('REP', 'CD-MSIPSO'), ('MUT', 'CD-MSIPSO')]
for env, tag in ENVS:
    recs = []; ps = []
    for A, B in pairs:
        xa = sub(env, A).set_index('run'); xb = sub(env, B).set_index('run'); idx = xa.index.intersection(xb.index)
        fa, fb = xa.loc[idx, 'best_fitness'].values, xb.loc[idx, 'best_fitness'].values
        w = stats.wilcoxon(fa, fb).pvalue
        pm, n10, n01 = mcnemar_exact(xa.loc[idx, 'feas_exact'].values.astype(bool), xb.loc[idx, 'feas_exact'].values.astype(bool))
        rng = np.random.default_rng(0); dd = fb - fa
        bs = [rng.choice(dd, len(dd)).mean() for _ in range(5000)]
        recs.append(dict(env=tag, A=A, B=B, n=len(idx), mean_diff=float(dd.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                         p_fit=float(w), p_feas=pm, A_only=n10, B_only=n01)); ps.append(w)
    adj = holm(ps); pf = holm([r['p_feas'] for r in recs])
    for r, a, b in zip(recs, adj, pf): r['p_fit_holm'] = float(a); r['p_feas_holm'] = float(b); rows.append(r)
abst = pd.DataFrame(rows); abst.to_csv(os.path.join(R, 'table_ablation_tests.csv'), index=False)
L = [r'\begin{tabular}{llrcrr}', r'\hline', r'Env & Contrast (B$-$A) & $\Delta$ fitness [95\% CI] & $p_{\rm Holm}$ (fit) & $p_{\rm Holm}$ (feas.) \\', r'\hline']
for env, tag in ENVS:
    for i, r in enumerate(abst[abst.env == tag].itertuples()):
        L.append(f'{tag if i==0 else ""} & {r.B} $-$ {r.A} & ${r.mean_diff:+.1f}$ [{r.lo:+.1f}, {r.hi:+.1f}] & {r.p_fit_holm:.3f} & {r.p_feas_holm:.3f} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_ablation_tests.tex'), 'w').write('\n'.join(L).replace('{llrcrr}', '{llrrr}'))

# ---------------- time-consistent reformulation (n = 20) ----------------
TC = ['MSIPSO', 'REP', 'TC-REP', 'MUT', 'TC-MUT', 'CD-MSIPSO', 'CD-MSIPSO-TC']
rows = []
for env, tag in ENVS:
    for a in TC:
        x = sub(env, a, 20); n = len(x); k = int(x.feas_exact.sum())
        rows.append(dict(env=tag, arm=a, fit=x.best_fitness.mean(), sd=x.best_fitness.std(), fex=k/n, ci=ci(k, n), t=x.time_sec.mean()))
tc = pd.DataFrame(rows); tc.to_csv(os.path.join(R, 'table_tc.csv'), index=False)
L = [r'\begin{tabular}{lccc}', r'\hline', r'Variant & E1 & E2 & E3R \\', r'\hline']
for a in TC:
    cells = []
    for env, tag in ENVS:
        r = tc[(tc.env == tag) & (tc.arm == a)].iloc[0]; cells.append(f'${r.fit:.0f}$ / {pct(r.fex)}\\%')
    L.append(f'{a} & ' + ' & '.join(cells) + r' \\')
L += [r'\hline', r'\end{tabular}']; open(os.path.join(T, 'tab_tc.tex'), 'w').write('\n'.join(L))

# ---------------- feasibility vs verification resolution (n = 20 main arms) ----------------
rows = []
for env, tag in ENVS + [('E3_FastDynamicThreats', 'E3')]:
    for a in MAIN:
        x = sub(env, a, 20)
        rows.append(dict(env=tag, algo=a, f25=x.feas25.mean(), f100=x.feas100.mean(), f1000=x.feas1000.mean(), fex=x.feas_exact.mean(),
                         overstated=int((x.feas25 & ~x.feas_exact).sum()), n25=int(x.feas25.sum())))
res = pd.DataFrame(rows); res.to_csv(os.path.join(R, 'table_resolution.csv'), index=False)
L = [r'\begin{tabular}{llrrrrr}', r'\hline', r'Env & Algorithm & 25 pts & 100 pts & 1000 pts & Continuous & Overstated \\', r'\hline']
for tag in ['E1', 'E2', 'E3R']:
    for i, r in enumerate(res[res.env == tag].itertuples()):
        L.append(f'{tag if i==0 else ""} & {r.algo} & {pct(r.f25)} & {pct(r.f100)} & {pct(r.f1000)} & {pct(r.fex)} & {r.overstated}/{r.n25} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_resolution.tex'), 'w').write('\n'.join(L))

# ---------------- best-feasible archive ----------------
rows = []
for env, tag in ENVS:
    for a in MAIN:
        x = sub(env, a, 20)
        bf = x[x.bf_found]
        rows.append(dict(env=tag, algo=a, bf_found=x.bf_found.mean(), bf_cont=(bf.bf_feas_exact.mean() if len(bf) else np.nan),
                         out_fit=x.best_fitness.mean(), bf_fit=bf.bf_fitness.mean() if len(bf) else np.nan,
                         bf_len=bf.bf_path_length.mean() if len(bf) else np.nan))
bfd = pd.DataFrame(rows); bfd.to_csv(os.path.join(R, 'table_bestfeasible.csv'), index=False)
L = [r'\begin{tabular}{llrrrr}', r'\hline', r'Env & Algorithm & Archive & Archive & Output & Archive \\', r' & & non-empty (\%) & cont.-feas. (\%) & fitness & fitness \\', r'\hline']
for tag in ['E1', 'E2', 'E3R']:
    for i, r in enumerate(bfd[bfd.env == tag].itertuples()):
        L.append(f'{tag if i==0 else ""} & {r.algo} & {pct(r.bf_found)} & {pct(r.bf_cont) if r.bf_cont==r.bf_cont else "--"} & {r.out_fit:.0f} & {r.bf_fit:.0f} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_bestfeasible.tex'), 'w').write('\n'.join(L))

# ---------------- fine-grid / robustness / budget ----------------
def summarize(g, keys):
    z = pd.read_csv(os.path.join(R, f'robust_{g}.csv'))
    return z.groupby(keys, sort=False).agg(n=('run', 'count'), fit=('best_fitness', 'mean'), f25=('feas25', 'mean'), fex=('feas_exact', 'mean'), k=('feas_exact', 'sum')).reset_index()
fg = summarize('finegrid', ['env', 'algo']); fg.to_csv(os.path.join(R, 'table_finegrid.csv'), index=False)
rb = summarize('robust', ['config', 'algo']); rb.to_csv(os.path.join(R, 'table_robust.csv'), index=False)
bd = summarize('budget', ['config', 'algo']); bd.to_csv(os.path.join(R, 'table_budget.csv'), index=False)
def grid_tex(df, rowkey, fname, arms=('GWO', 'MSIPSO', 'CD-MSIPSO')):
    L = [r'\begin{tabular}{l' + 'r' * len(arms) + '}', r'\hline', ('Setting' if rowkey!='env' else 'Env') + ' & ' + ' & '.join(arms) + r' \\', r'\hline']
    for key in df[rowkey].unique():
        cells = []
        for a in arms:
            r = df[(df[rowkey] == key) & (df.algo == a)].iloc[0]; cells.append(f'${r.fit:.0f}$ / {100*r.fex:.0f}\\%')
        lab = str(key).replace('N=', '$n_p{=}$').replace(',T=', ', $I{=}$'); L.append(f'{lab} & ' + ' & '.join(cells) + r' \\')
    L += [r'\hline', r'\end{tabular}']; open(os.path.join(T, fname), 'w').write('\n'.join(L))
dn = summarize('density', ['config', 'algo']); dn.to_csv(os.path.join(R, 'table_density.csv'), index=False); grid_tex(dn, 'config', 'tab_density.tex')
grid_tex(rb, 'config', 'tab_robust.tex'); grid_tex(bd, 'config', 'tab_budget.tex')
fg2 = fg.copy(); fg2['env'] = fg2.env.map(lambda s: s.split('_')[0]); grid_tex(fg2, 'env', 'tab_finegrid.tex')

# ---------------- scalability / margin ----------------
sc = pd.read_csv(os.path.join(R, 'scalability_results_v3.csv'))
s = sc.groupby(['n_uav', 'algo']).agg(fit=('fitness', 'mean'), f25=('feas25', 'mean'), fex=('feas_exact', 'mean'), k=('feas_exact', 'sum'), n=('seed', 'count'), t=('time_sec', 'mean')).reset_index()
s.to_csv(os.path.join(R, 'table_scal.csv'), index=False)
L = [r'\begin{tabular}{rlrrrr}', r'\hline', r'$N_{\rm UAV}$ & Algorithm & Fitness & Feas$_{25}$ (\%) & Feas$_{\rm cont}$ (\%) & Time (s) \\', r'\hline']
for n in sorted(s.n_uav.unique()):
    for i, r in enumerate(s[s.n_uav == n].itertuples()):
        L.append(f'{n if i==0 else ""} & {r.algo} & {r.fit:.0f} & {pct(r.f25)} & {pct(r.fex)} & {r.t:.2f} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_scal.tex'), 'w').write('\n'.join(L))
mg = pd.read_csv(os.path.join(R, 'margin_scaling_test_v3.csv'))
L = [r'\begin{tabular}{rlrrr}', r'\hline', r'$N_{\rm UAV}$ & Margin rule & $m$ & Fitness & Feas$_{\rm cont}$ [95\% CI] \\', r'\hline']
for n in sorted(mg.n_uav.unique()):
    for i, r in enumerate(mg[mg.n_uav == n].itertuples()):
        k = round(r.feasibility_rate * 12); L.append(f'{n if i==0 else ""} & {r.condition.replace('fixed_m9','fixed').replace('sqrt_scale','$\\sqrt{}$-scaled').replace('linear_scale','linear')} & {r.margin:.1f} & {r.mean_fitness:.0f} & {pct(r.feasibility_rate)} {ci(k, 12)} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_margin.tex'), 'w').write('\n'.join(L))

# ---------------- decomposed fitness components (n = 20) ----------------
rows = []
for env, tag in ENVS:
    for a in MAIN:
        z = sub(env, a, 20)
        rows.append(dict(env=tag, algo=a, jlen=z.J_len.mean(), pen=(60*z.J_stat + 80*z.J_dyn + 100*z.J_coop).mean(), smooth=(0.4*z.J_smooth).mean(), fit=z.best_fitness.mean(), fex=z.feas_exact.mean()))
cp = pd.DataFrame(rows); cp.to_csv(os.path.join(R, 'table_components.csv'), index=False)
L = [r'\begin{tabular}{llrrrrr}', r'\hline', r'Env & Algorithm & $J_{\rm len}$ & Weighted penalty & Weighted smooth. & Fitness & Feas$_{\rm cont}$ (\%) \\', r'\hline']
for tag in ['E1', 'E2', 'E3R']:
    for i, r in enumerate(cp[cp.env == tag].itertuples()):
        L.append(f'{tag if i==0 else ""} & {r.algo} & {r.jlen:.0f} & {r.pen:.1f} & {r.smooth:.1f} & {r.fit:.0f} & {pct(r.fex)} \\\\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(os.path.join(T, 'tab_components.tex'), 'w').write('\n'.join(L))

# ---------------- timing alignment (final CD-MSIPSO solutions, 20 runs) ----------------
ta = pd.read_csv(os.path.join(R, 'timing_alignment.csv'))
g = ta.groupby('env').mean(numeric_only=True)
L = [r'\begin{tabular}{lrrrr}', r'\hline', r'Env & Mean $|\Delta t|$ & Per-solution max $|\Delta t|$ (mean) & Mean dist.\ gap & Per-solution max gap (mean) \\', r'\hline']
for env, tag in ENVS:
    r = g.loc[env]; L.append(f'{tag} & {r.mean_abs_dt:.3f} & {r.max_abs_dt:.3f} & {r.mean_abs_dist_gap:.1f} & {r.max_abs_dist_gap:.1f} \\\\')
L += [r'\hline', r'\end{tabular}']; open(os.path.join(T, 'tab_timing.tex'), 'w').write('\n'.join(L))

# ---------------- runtime overhead (n=60 where available) ----------------
ov = {}
for env, tag in ENVS:
    a, b = sub(env, 'MSIPSO'), sub(env, 'CD-MSIPSO'); ov[tag] = 100 * (b.time_sec.mean() / a.time_sec.mean() - 1)
OUT['overhead_pct_CD_vs_MSIPSO'] = ov
rt = d[d.algo.isin(MAIN) & (d.run < 20) & d.env.isin([e for e, _ in ENVS])].groupby('algo').time_sec.agg(['min', 'max', 'mean'])
OUT['runtime_range_by_algo'] = rt.round(3).to_dict()
t = d[(d.run < 20) & d.algo.isin(MAIN) & d.env.isin([e for e, _ in ENVS])].groupby(['env', 'algo']).time_sec.mean().unstack().round(3); print(t)
json.dump(OUT, open(os.path.join(R, 'numbers.json'), 'w'), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
