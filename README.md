# CD-MSIPSO: code and data

Code, scenarios, seeds and per-run results for the paper
**"Cooperative Dynamic-Obstacle-Aware Multi-Strategy Improved Particle Swarm Optimization for Multi-UAV Path Planning"**
(R. H. Shaik, G. Danaiah, R. Krishna M).

Pure Python (NumPy, SciPy, pandas, Matplotlib); no MATLAB or proprietary toolboxes. Python 3.10+.

```bash
pip install -r code/requirements.txt
```

## Layout

| Path | Content |
|---|---|
| `code/environment.py` | Scenarios E1, E2, E3 (infeasible by construction), E3R, obstacle/start/goal definitions, fitness (25-sample and optional exact-clearance) |
| `code/baselines.py` | PSO, GWO, WOA |
| `code/cdmsipso.py` | MSIPSO core, cooperative repulsion, dynamic-obstacle-aware mutation, time-consistent variants, `run_variant` |
| `code/verify.py` | Independent exact continuous-time verifier (closed-form closest approach per piece) |
| `code/run_revision_experiments.py` | Main comparison and factorial ablation (paired runs, seed = 1000*env + run) |
| `code/followup_experiments.py`, `robustness_experiments.py`, `budget_confirm.py`, `sensitivity_gwoprob.py`, `scalability_experiment.py`, `margin_scaling_test.py`, `timing_test.py`, `verifier_validation.py` | Further studies (exact-clearance fitness, operator activation, p_gwo sweep, budget, density, scaling, margin, timing, verifier validation) |
| `code/analyze_revision.py`, `make_revision_tables.py`, `make_extended_tables.py`, `make_figures_revision.py` | Statistics, LaTeX tables, figures |
| `results/` | Per-run CSVs and summary tables for every table and figure in the paper |
| `figures/` | The seven result figures |
| `paper/tables/` | The generated LaTeX tables |

## Quick check (about 1 minute)

```bash
cd code
python run_revision_experiments.py 0 0 2 smoke_test      # E1, runs 0-1, all arms
```
Output: `results/revision_smoke_test.csv`. The same seed always gives the same result.

## Reproducing the paper

Environment indices: 0 = E1, 1 = E2, 3 = E3R (2 = original infeasible E3).
Run from `code/`; every script writes to `../results/`.

```bash
# 1. main comparison + ablation, runs 0-19, all four environments
for e in 0 1 2 3; do python run_revision_experiments.py $e 0 20 env$e; done
# 1b. runs 20-59 for GWO, MSIPSO, REP, MUT, CD-MSIPSO in E1, E2, E3R
for e in 0 1 3; do python run_revision_experiments.py $e 20 60 ext$e GWO,MSIPSO,REP,MUT,CD-MSIPSO; done
# 2. runs 60-199 for the same five arms (200 runs in total; resumable)
bash run_ext.sh
# 3. follow-up studies
bash run_follow.sh ; bash run_follow2.sh
python budget_confirm.py 30 300 ;  python budget_confirm.py 90 100
python sensitivity_gwoprob.py 0                          # also 1 and 3
python robustness_experiments.py <group> 12              # see script header for groups
python scalability_experiment.py ; python margin_scaling_test.py
python verifier_validation.py
# 4. tables and figures
python make_revision_tables.py ; python make_extended_tables.py ; python make_figures_revision.py
```
The full set takes several CPU-hours. The raw outputs we used are already in `results/`, so tables and figures can be regenerated without re-running anything.

## Key definitions

* Timing model: mission time tau in [0,1], constant speed, common arrival; waypoint time = cumulative arc length / path length.
* Feasibility is reported at the optimiser's 25 samples (`feas25`) and in exact continuous time (`feas_exact`, tolerance 1e-6). `feas_exact` is the primary metric.
* Statistics: paired Wilcoxon, matched-pairs rank-biserial effect size, Friedman, exact McNemar, Holm correction, Wilson intervals.

## Citation

See `CITATION.cff`. Cite the archived Zenodo release (DOI to be added after publication).

## License

MIT, see `LICENSE`.
