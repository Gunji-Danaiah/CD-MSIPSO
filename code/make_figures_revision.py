"""Figures for the revised manuscript (all generated from the stored raw data)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_revision import wilson
from environment import make_environments
from cdmsipso import run_variant
from verify import exact_report, uav_trajectory
R = os.path.join(HERE, '..', 'results'); F = os.path.join(HERE, '..', 'figures')
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
d = pd.read_csv(os.path.join(R, 'revision_all_runs.csv'))
ALG = ['PSO', 'GWO', 'WOA', 'MSIPSO', 'CD-MSIPSO']
COL = {'PSO': '#7f8c8d', 'GWO': '#2471a3', 'WOA': '#8e44ad', 'MSIPSO': '#e67e22', 'CD-MSIPSO': '#c0392b',
       'REP': '#16a085', 'MUT': '#b7950b'}
ENVS = [('E1_CrossingTraffic', 'E1: crossing traffic (3 UAV)', 0), ('E2_DenseField', 'E2: dense field (4 UAV)', 1),
        ('E3R_FastDynamicThreats_Revised', 'E3R: relocated obstacles (3 UAV)', 3)]

# Fig 1 convergence
hist = {}
for i in (0, 1, 3): hist.update(json.load(open(os.path.join(R, f'revision_hist_env{i}.json'))))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (env, ttl, _) in zip(axes, ENVS):
    for a in ALG:
        ax.plot(np.mean(hist[f'{env}__{a}'], axis=0), color=COL[a], lw=1.6, label=a)
    ax.set_yscale('log'); ax.set_title(ttl, fontsize=10); ax.set_xlabel('Iteration'); ax.grid(alpha=.3)
axes[0].set_ylabel('Best fitness (mean of 20 runs)'); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig1_convergence.png'), dpi=200); plt.close()

# Fig 2 boxplots (n=20)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (env, ttl, _) in zip(axes, ENVS):
    data = [d[(d.env == env) & (d.algo == a) & (d.run < 20)].best_fitness.values for a in ALG]
    bp = ax.boxplot(data, tick_labels=ALG, patch_artist=True, showmeans=True)
    for p, a in zip(bp['boxes'], ALG): p.set_facecolor(COL[a]); p.set_alpha(.6)
    ax.set_title(ttl, fontsize=10); ax.tick_params(axis='x', rotation=30); ax.grid(alpha=.3, axis='y')
axes[0].set_ylabel('Final best fitness')
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig2_boxplot.png'), dpi=200); plt.close()

# Fig 3 sampled vs continuous feasibility
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (env, ttl, _) in zip(axes, ENVS):
    x = np.arange(len(ALG)); w = 0.38
    s = [100 * d[(d.env == env) & (d.algo == a) & (d.run < 20)].feas25.mean() for a in ALG]
    c = [100 * d[(d.env == env) & (d.algo == a) & (d.run < 20)].feas_exact.mean() for a in ALG]
    ax.bar(x - w / 2, s, w, color='#bdc3c7', edgecolor='k', label='25 sample points (optimiser grid)')
    ax.bar(x + w / 2, c, w, color=[COL[a] for a in ALG], edgecolor='k', label='continuous-time (exact)')
    ax.set_xticks(x); ax.set_xticklabels(ALG, rotation=30); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3, axis='y')
axes[0].set_ylabel('Feasible runs (%)'); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, loc='lower center', ncol=2)
plt.tight_layout(rect=[0, 0.09, 1, 1]); plt.savefig(os.path.join(F, 'fig3_feasibility.png'), dpi=200); plt.close()

# Fig 4 factorial ablation (n=60)
ABL = ['MSIPSO', 'REP', 'MUT', 'CD-MSIPSO']; LAB = ['neither\n(MSIPSO)', 'repulsion\nonly', 'mutation\nonly', 'both\n(CD-MSIPSO)']
fig, axes = plt.subplots(2, 3, figsize=(13, 5.6))
for j, (env, ttl, _) in enumerate(ENVS):
    m, e, f, lo, hi = [], [], [], [], []
    for a in ABL:
        x = d[(d.env == env) & (d.algo == a)]; n = len(x)
        m.append(x.best_fitness.mean()); e.append(1.96 * x.best_fitness.std() / np.sqrt(n))
        k = int(x.feas_exact.sum()); l, h = wilson(k, n); f.append(100 * k / n); lo.append(100 * (k / n - l)); hi.append(100 * (h - k / n))
    cols = ['#e67e22', '#16a085', '#b7950b', '#c0392b']
    axes[0, j].bar(range(4), m, yerr=e, color=cols, edgecolor='k', capsize=3); axes[0, j].set_title(ttl, fontsize=10)
    axes[0, j].set_ylim(min(m) * 0.9, max(m) * 1.08); axes[0, j].grid(alpha=.3, axis='y')
    axes[1, j].bar(range(4), f, yerr=[lo, hi], color=cols, edgecolor='k', capsize=3); axes[1, j].set_ylim(0, 105); axes[1, j].grid(alpha=.3, axis='y')
    for r in (0, 1): axes[r, j].set_xticks(range(4)); axes[r, j].set_xticklabels(LAB, fontsize=8)
axes[0, 0].set_ylabel('Final fitness (mean, 95% CI)'); axes[1, 0].set_ylabel('Continuous-time feasible (%)\n(Wilson 95% CI)')
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig4_ablation.png'), dpi=200); plt.close()

# Fig 5 trajectories with safety indicators
def draw_sphere(ax, c, r, color, alpha):
    u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:10j]
    ax.plot_surface(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v), c[2] + r * np.cos(v), color=color, alpha=alpha, linewidth=0)
envs = make_environments(); fig = plt.figure(figsize=(13, 5.4))
for k, gi in enumerate((0, 3)):
    env = envs[gi]; ax = fig.add_subplot(1, 2, k + 1, projection='3d')
    best = None
    for run in range(20):
        x, f, _ = run_variant(env, 30, 100, 1000 * gi + run)
        rep = exact_report(env, x)
        if rep['feasible_exact']: best = (x, f, rep, run); break
    x, f, rep, run = best
    for (cx, cy, cz, r) in env.static_obs: draw_sphere(ax, (cx, cy, cz), r, '#7f8c8d', .45)
    for (cx, cy, cz, r, vx, vy, vz) in env.dynamic_obs:
        ax.plot([cx, cx + vx], [cy, cy + vy], [cz, cz + vz], '--', color='#c0392b', lw=1.5)
        draw_sphere(ax, (cx, cy, cz), r, '#c0392b', .35); draw_sphere(ax, (cx + vx, cy + vy, cz + vz), r, '#c0392b', .15)
    pal = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
    for u, (T, P) in enumerate(uav_trajectory(env, x)):
        ax.plot(P[:, 0], P[:, 1], P[:, 2], '-o', color=pal[u], ms=3, lw=2); ax.text(P[0, 0], P[0, 1], P[0, 2] + 3, f'UAV{u+1}', color=pal[u], fontsize=9, weight='bold')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 50); ax.view_init(elev=28, azim=-60)
    ax.set_title(f'{env.name.split("_")[0]} (seed run {run}): min UAV sep {rep["min_uav_sep"]:.1f}; static clr. {rep["min_static_margin"]:+.1f}; dyn. clr. {rep["min_dyn_margin"]:+.1f}', fontsize=8.5)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig5_paths3d.png'), dpi=200); plt.close()

# Fig 6 scalability (fleet size)
sc = pd.read_csv(os.path.join(R, 'scalability_results_v3.csv'))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for a in ['GWO', 'MSIPSO', 'CD-MSIPSO']:
    g = sc[sc.algo == a].groupby('n_uav')
    axes[0].plot(g.fitness.mean().index, g.fitness.mean().values, '-o', color=COL[a], label=a)
    axes[1].plot(g.feas_exact.mean().index, 100 * g.feas_exact.mean().values, '-o', color=COL[a], label=a + ' (continuous)')
    axes[1].plot(g.feas25.mean().index, 100 * g.feas25.mean().values, ':', color=COL[a], label=a + ' (25 pts)')
axes[0].set_xlabel('Fleet size $N_{UAV}$'); axes[0].set_ylabel('Mean final fitness'); axes[1].set_xlabel('Fleet size $N_{UAV}$'); axes[1].set_ylabel('Feasible runs (%)')
axes[0].legend(fontsize=8); axes[1].legend(fontsize=6.5, ncol=2)
for ax in axes: ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig6_scalability.png'), dpi=200); plt.close()

# Fig 7 sensitivity to p_gwo (post hoc; CD-MSIPSO; 20 runs per value; includes p_gwo = 0)
import glob
pg = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'followup_pgwo_*.csv'))]); pg['p'] = pg.algo.str.replace('pgwo=', '').astype(float)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for env, ttl, gi in ENVS:
    z = pg[pg.env == env]; m = z.groupby('p').best_fitness.mean(); f = 100 * z.groupby('p').feas_exact.mean()
    axes[0].plot(m.index, m / m.min(), '-o', label=ttl.split(':')[0]); axes[1].plot(f.index, f, '-o', label=ttl.split(':')[0])
for ax in axes: ax.axvline(0.85, color='gray', ls='--'); ax.set_xlabel('$p_{gwo}$'); ax.grid(alpha=.3)
axes[0].set_ylabel('Mean fitness / best in environment'); axes[1].set_ylabel('Continuous-time feasible (%)'); axes[0].legend(fontsize=8)
axes[0].set_title('dashed: value 0.85 (fixed beforehand)', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(F, 'fig7_sensitivity.png'), dpi=200); plt.close()
print('figures done')
