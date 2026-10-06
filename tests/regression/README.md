# Regression checks

One script per functionality of tam, run on one frozen dataset, reported in **text files that are committed**: `RESULTS.txt` for the stable modules and `RESULTS_EXP.txt` for the beta and experimental ones.
When a change moves a number, the file changes, the pull request shows the diff, and the maintainer reviews it like any other change.
There is no baseline to download, no tolerance to tune and no list of declared changes: the diff is the declaration.

```bash
pip install -e .
python tests/regression/run.py            # rewrites tests/regression/RESULTS.txt and RESULTS_EXP.txt (about two minutes on a CPU)
git diff tests/regression/                # what your change moved
```

If the diff is what you intended, commit the new results files in the same pull request and say in the description why the numbers moved.
If it is not, you found a regression.

## The checks (`checks/<number>_<functionality>.py`)
Each script fits a few models on the frozen dataset and records numbers with `res`. A forecast is recorded as its error against the truth, its bias, its mean,
its spread and five quantiles, so that a change of one part in a million is visible without storing the whole vector. A step that raises is recorded as an
`ERROR` line (the other steps still run), so a broken feature also shows in the diff. The scripts use only the public API of tam, so the same checks run on any version.

| Scripts | What they check |
| --- | --- |
| 01 to 12 | the effects, one script each: linear and categorical (all topologies), spline (knots, degree, penalty order, extrapolation), Fourier, Chebyshev, wavelet, RBF, tree, linear tree, neural, tensor product, universal physics, PID |
| 13, 14 | fixed penalties (per effect and model-wide); penalties chosen by GCV (gamma, bounds, steps, tensor, summary table) |
| 15 | losses: gamma, Poisson, binomial, expectile, Huber, Student-t |
| 16, 17, 18 | distributional fits (quantiles, median, CDF, CRPS, anomaly score, tail families, mixtures); conformal intervals (split, studentised, Mondrian, ACI); extreme-value tail and Gaussian copula |
| 19 | the prediction API: any row order, duplicate labels, extra columns, a subset of the groups, no group column; decomposition into effects and its names; `summary()` |
| 20 to 22 | `AdaptiveTAM`: windows, update interval, horizon; correcting a base model; equivalence with the loop an operator runs |
| 23 to 25 | `KalmanTAM`: blocks, noises, calibration; one noise per term; equivalence with the loop an operator runs |
| 26, 27, 28 | `OperaTAM` (algorithms, learning rate, loss, horizon, weights); `HierarchicalTAM` (coherence, strength of the constraint, own formulas); `NeuralTAM` (seeds, activations, splits, decomposition) |
| 29, 30 | the errors a user meets (what is raised and what it says) and the warnings (how many, which class, which message) |
| 31, 32, 33, 34 | hyper-parameter search; the evaluation helpers; a very small `AutoTAM` run; component plots (axes and curves, not pixels) |
| 36 | a formula mixing a spline, a linear term, a cyclic Fourier term and a categorical (and a bigger one with a tensor product and an ordinal categorical): the components sum to the forecast and none collapses to zero (the spread of each is recorded) |
| 37, 38 | interactions: the `others` parameter of `rbf()` and `n()`, tensor products of two and three margins |
| 35 | spline extrapolation in isolation: fit on winter (January to March), forecast summer (June to August, mostly outside the training range) with the default spline, `extrapolate='continue'`, `'constant'`, `'linear'` and a linear effect |

## The dataset (`data/force_2023.csv`)
One year of the FORCE dataset (Zenodo 10.5281/zenodo.21109134, CC-BY 4.0), hourly (24 groups), 8,736 rows, 12 columns: the calendar, the temperature, the RTE national load
and the Enedis load by segment (which also gives a hierarchy). The day of the spring clock change is dropped so that every hour has 364 days.
It is **frozen**: CI never rebuilds it (`make_data.py` does, from the Zenodo file) and `data/force_2023.sha256` fails a test if the file is touched.
Rebuilding it is a deliberate change, and `RESULTS.txt` changes with it. It does not depend on `tam/data` or on any application dataset.
Static checks hold out every fifth day (so no effect is asked to extrapolate); the sequential checks (adaptive, Kalman, Opera) fit the first half-year and simulate the second.

## The results files
A check that covers a beta or experimental module (Kalman, hierarchical, the neural, physics, tree and linear-tree effects, `NeuralTAM`, `AutoTAM`) sets `EXPERIMENTAL = True` and reports into `RESULTS_EXP.txt`; the others report into `RESULTS.txt`.
A change in `RESULTS.txt` fails CI; a change in `RESULTS_EXP.txt` is shown as a warning and does not.
The sequential checks (adaptive, Kalman, Opera) build on a base model without extrapolating effects (a cyclic Fourier term for the day of the year, a linear temperature), so they test the loops and not the extrapolation of splines, which has its own check (35). The 28-day window with a spline is kept in the adaptive check as a deliberate stress case.

One `<check>.<name> = <value>` line per number, sorted, six significant digits, no version and no timing: it only changes when a result does.
A difference that should be zero (two ways of computing the same thing) is recorded as `<= 1e-06` (below that it is rounding noise, which differs between machines).

## In CI (`.github/workflows/regression.yml`)
On every pull request into `main`, `automl` and `feat/**`: install tam, run `run.py`, then `git diff --exit-code tests/regression/RESULTS.txt`.
The job is red when that file differs; the experimental file only raises a warning. Both regenerated files are uploaded as an artifact.
Floating-point results can differ in the last digits between operating systems, so the committed `RESULTS.txt` is the one CI produces: if your local run differs from CI on
a few last digits, commit the artifact of the failed job.

## Every argument is exercised
Every argument of every effect (categorical topologies and difference order, spline knots / degree / penalty order, Fourier harmonics / smoothness / period scale, polynomial degree, wavelet scales and locations, RBF centres / bandwidth / smoothness `nu` / partner features, tree and linear-tree depth / leaves / `sp_alpha` / split strategy, neural width / activation / depth / seed, tensor margins, physics basis and differential weights, PID window and derivative penalty, the extrapolation mode and the penalty weight of each) and every loss is used by at least one check; `tests/test_regression_runner.py` fails if a check stops using one.
The default penalty weight (1e-9) hides the effect of several arguments (the difference order, the physics weights, the PID derivative penalty): those arguments are also tested with an explicit penalty.

## Adding a check
Create `checks/<next number>_<functionality>.py` with a docstring saying what it checks and a function `run(res)`; use `res.static(name, formula)` to fit and record a
`StaticTAM`, `res.forecast`, `res.value`, `res.gap` and `res.attempt`. Keep it to a few seconds. Run `run.py` and commit the new lines.
