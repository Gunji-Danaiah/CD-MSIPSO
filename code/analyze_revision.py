"""Statistics for the revision: tables of means, Wilson CIs, paired tests, Holm correction."""
import sys, os, glob
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results')

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m-rank)*p[i]); adj[i] = min(1, run)
    return adj

def mcnemar_exact(a, b):
    """a,b boolean arrays (paired). exact two-sided binomial on discordant pairs."""
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b)); n = n10 + n01
    if n == 0: return 1.0, n10, n01
    return stats.binomtest(n10, n, 0.5).pvalue, n10, n01

def load():
    main = pd.read_csv(os.path.join(R, 'revision_main.csv')) if os.path.exists(os.path.join(R,'revision_main.csv')) else None
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, 'revision_env*.csv')))]
    parts += [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, 'revision_ext*.csv')))]
    d = pd.concat(parts, ignore_index=True).drop_duplicates(['env', 'algo', 'run'])
    return d

if __name__ == '__main__':
    d = load(); d.to_csv(os.path.join(R, 'revision_all_runs.csv'), index=False)
    print(d.groupby(['env','algo']).size().unstack().fillna(0).astype(int))
