# driftwatch — drift and quality monitoring for a model already in production

[![ci](https://github.com/RaghavBanuni/mlops-drift-monitoring-service/actions/workflows/ci.yml/badge.svg)](https://github.com/RaghavBanuni/mlops-drift-monitoring-service/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A monitoring sidecar for a deployed model. It takes a window of production traffic, decides
whether anything moved **by more than that window's sampling noise**, checks whether the model's
measured quality actually fell, and then says which of four situations you are in and what to do.

The hard part of drift monitoring is not computing PSI. It is not alerting on everything. A
monitor that fires on every seasonal wobble is muted within a month, and a muted monitor is worse
than none: it costs money and provides false assurance. Every opinion below is about restraint,
and every one of them is tested.

| Decision | Why |
| --- | --- |
| The reference is **pinned to the model artefact**, never rolling | A reference that follows the data makes gradual drift invisible: every step looks small, the dashboard stays green, and a year later the model scores a population it never saw |
| **Schema checks run before any statistic** | A PSI on a column that is 90% null this week is a number, not evidence. Most incidents are broken joins and type changes, not subtle shifts |
| **Effect size decides; p-values only demote** | With 50 000 rows a 0.3% shift is significant and irrelevant. Significance answers "did anything move"; a retraining decision needs "how much" |
| The threshold **adapts to the window size** | PSI has a noise floor of roughly `(k-1)(1/n_ref + 1/n_cur)`. A fixed 0.1 cries wolf on small windows and goes blind on large ones |
| Multiplicity is corrected **once, across the window** (Benjamini-Hochberg) | 40 features a day at alpha = 0.05 is about two false alarms a day from a perfectly stable pipeline |
| An alert needs **confirmation in 2 of 3 windows**, then goes quiet | One odd window is sampling noise; a persistent problem is one incident, not sixty tickets |
| Feature drift is reported **together with measured quality** | Neither means anything alone - which is the entire point below |

---

## The reading that actually matters

A drift alert is evidence about the *inputs*. It is not evidence about the *model*. Only the
pair of signals identifies what to do, and one of the four cases is invisible to every
feature-drift detector ever written.

| Feature drift | Quality drop | Reading | Action |
| --- | --- | --- | --- |
| no | no | stable | nothing; keep the baseline pinned |
| **yes** | no | covariate shift the model absorbed | note it; do **not** retrain on a moved population without a reason |
| no | **yes** | **concept drift** - the relationship changed, not the inputs | retrain on recent labelled data; no feature monitor can see this |
| **yes** | **yes** | population moved and the model followed it down | investigate the pipeline *first* (a simultaneous move usually means an upstream change), then retrain |

`Monitor.interpret()` returns exactly this, plus a fifth honest state - *outcomes pending* - for
the usual case where labels have not matured. It refuses to treat "no labels" as "quality fine",
which is the most common way a monitoring report misleads its reader.

```python
>>> monitor.interpret()["case"]
'concept_drift'
>>> monitor.interpret()["action"]
'retrain on recent labelled data: the inputs did not move, the relationship did, and no
feature-drift monitor can see this'
```

---

## Architecture

```
        window of scored traffic                    outcomes, weeks later
                   |                                          |
                   v                                          v
        +--------------------+                     +---------------------+
        | schema.py          |  blocking issues    | performance.py      |
        | contract and data  |----- suppress ----->| AUC / Brier with a  |
        | quality checks     |      the test       | bootstrap CI, label |
        +---------+----------+                     | buffer, coverage,   |
                  |                                | lag, base-rate test |
                  v                                +----------+----------+
        +--------------------+                                |
        | detect.py          |  effect size, adaptive         |
        | per-feature        |  floor, chi-square,            |
        | grading            |  BH across the window          |
        +---------+----------+                                |
                  v                                           |
        +--------------------+                                |
        | alerts.py          |  confirmation, cooldown,       |
        | what pages a human |  escalation, volume cap        |
        +---------+----------+                                |
                  +-------------------+--------------------- -+
                                      v
                            +--------------------+
                            | monitor.py         |  history, sequential
                            | the reading        |  prediction watch,
                            +---------+----------+  interpret()
                                      v
                        service.py (HTTP)    cli.py (batch)
```

| Module | Responsibility |
| --- | --- |
| `stats.py` | Tail probabilities (normal, chi-square, incomplete gamma, Kolmogorov), Benjamini-Hochberg, Bonferroni, percentile bootstrap. **No scipy** - hand-implemented and checked against published critical values |
| `metrics.py` | PSI with per-bin decomposition, KS, 1-Wasserstein, total variation, Jensen-Shannon, Cramer's V, chi-square homogeneity with expected-count pooling |
| `baseline.py` | The pinned reference: quantile edges, **stored** bin masses, category shares with a rare-level bucket, null rates, reference AUC. JSON in, JSON out, no raw rows |
| `schema.py` | Missing / retyped / constant / all-null columns, null-rate spikes, out-of-range and sign-flip values, unseen and vanished levels - graded `blocking` / `warning` / `info` |
| `detect.py` | Per-feature grading, adaptive thresholds, FDR correction, the drift-versus-quality decision table |
| `sequential.py` | Page-Hinkley and EWMA, for slow drifts no single-window test will catch |
| `performance.py` | Delayed-label buffer joined by id, coverage and lag reporting, AUC with a bootstrap CI, base-rate shift test |
| `alerts.py` | The policy layer that decides what a human actually sees |
| `monitor.py` | Stateful orchestration across windows, and `interpret()` |
| `simulate.py` | Synthetic traffic with **known change points**, plus a harness that scores competing detectors |
| `service.py`, `cli.py` | HTTP sidecar and batch entry points. Neither contains any statistics |

---

## The maths, and why each piece is there

### Population Stability Index, and its noise floor

For bins with reference mass `p` and current mass `q`:

```
PSI = sum_i (q_i - p_i) * ln(q_i / p_i)
```

This is the symmetrised Kullback-Leibler (Jeffreys) divergence. Its second-order expansion is
the chi-square statistic divided by `n`, which gives the most useful number in this repository -
the PSI you should expect between two samples of the **same** distribution:

```
E[PSI | no drift] ~ (k - 1) * (1/n_ref + 1/n_cur)
```

So "PSI above 0.1 means investigate" is not a property of drift. It is a property of your sample
sizes. With ten bins:

| reference rows | window rows | expected PSI, no drift | effective warn threshold |
| --- | --- | --- | --- |
| 4 000 | 200 | 0.047 | **0.142** |
| 4 000 | 500 | 0.020 | 0.100 |
| 4 000 | 2 000 | 0.007 | 0.100 |
| 1 000 | 300 | 0.039 | **0.117** |
| 500 | 200 | 0.063 | **0.189** |
| 20 000 | 20 000 | 0.0009 | 0.100 |
| 1 000 | 300 (20 bins) | 0.082 | **0.247** |

The threshold used is `max(configured, 3 x floor)`, so a 200-row window is not allowed to raise
an alarm inside its own noise. `tests/test_detect.py` measures this rather than trusting it: it
draws twenty undrifted windows and asserts the median observed PSI sits within a factor of the
predicted floor, and that none of them crosses the adaptive threshold.

### Total variation for categoricals - the same trap, worse

```
TVD = 0.5 * sum_i |q_i - p_i|
E[TVD | no drift] = 0.5 * sum_i sqrt( 2 * p_i(1 - p_i)(1/n_ref + 1/n_cur) / pi )
```

A 5-level column on a 4 000 / 800 split has a floor near **0.031**. A 40-level column on a
500 / 500 split has a floor near **0.158** - larger than the 0.10 threshold people alert on.
High-cardinality categoricals are where fixed thresholds produce permanent noise, which is why
rare levels are folded into one bucket at fit time and the floor is computed per column.

### Significance: chi-square on the stored bins, not KS

The baseline deliberately stores no raw reference rows, only edges and masses, so the
significance test is a chi-square homogeneity test on those bins, with expected counts below 5
pooled and the surviving degrees of freedom reported in `detail.chi2_dof`. That is a real trade:
binning costs a little power against a two-sample KS test. `metrics.ks_test` remains available
wherever both raw samples exist, and the evaluation harness scores KS side by side so the cost is
visible instead of assumed.

Significance can only ever **demote** a verdict, never create one, and the note says which of the
two ingredients was missing:

```
effect 0.1873 clears the alert threshold but is not statistically supported after FDR
correction at this window size, so it is reported as 'warn' rather than paged
```

### Benjamini-Hochberg across the window

Testing 40 features every day at alpha = 0.05 yields about two significant results per day from a
stable pipeline - roughly 600 false alarms a year, which is how monitoring gets muted. BH bounds
the expected *false-discovery share among the features actually flagged*, and unlike Bonferroni it
keeps its power at 40 tests. The q-values are made monotone with a running minimum, and the
step-up rule rejects everything up to the largest passing rank.

An unseen categorical level is escalated on its own, without needing significance: a level the
model has never seen is a coding problem, not a distribution shift.

### Quality under delayed labels

AUC is computed by the Mann-Whitney identity with midrank ties - a model emitting one constant
score scores exactly 0.5, not something flattering - and wrapped in a percentile bootstrap CI. A
drop is called `degraded` only when the **interval clears the baseline**, because a 900-row window
gives an AUC standard error near 0.02 and a 0.01 "drop" is noise however alarming the dashboard
looks. Coverage and median label lag are reported next to every number, and a thin window says so
in words: *"only 8% of the window is labelled, so treat this as provisional"*. Predictions are
buffered by id and joined when the outcome lands, so labels can arrive in any order; a label for a
row the buffer has already evicted is counted as too late rather than dropped in silence.

### Sequential detectors

A 0.06 sd shift per window is invisible to any single-window test and obvious after ten.
Page-Hinkley accumulates a stream's drift against a slack `delta` and fires when the excursion
from its own minimum exceeds `lambda`; the EWMA monitor uses the correct smoothed control limit
`L * sigma * sqrt(lambda / (2 - lambda))` rather than a naive `3 sigma` band that would never
trigger. `Monitor` runs Page-Hinkley on the mean prediction, the one signal that needs no labels.

---

## Does it actually work? Reproduce the comparison

Showing that a detector fires on injected drift proves nothing: a detector with no threshold
discipline fires on everything. The interesting question is the **trade** - how many windows late
does it fire, and how often does it fire when nothing happened. That needs ground truth, so
`simulate.py` generates traffic around a *frozen* deployed model (frozen coefficients, frozen
scaling, frozen median imputation) and moves the world instead.

| Scenario | What changes | What a good monitor should do |
| --- | --- | --- |
| `stable` | nothing | never fire; every alarm here is a false alarm |
| `benign_seasonal` | channel mix oscillates | ideally stay quiet: real drift, no harm. Alerting here is what teaches people to ignore alerts |
| `gradual_covariate` | income creeps 0.06 sd per window | catch it eventually; this is what sequential detectors are for |
| `sudden_covariate` | income +0.8 sd, utilisation +0.5 sd | fire within a window or two, and name those two features |
| `concept` | **features unchanged**, label relationship changes | report *no* feature drift and a measured quality drop |
| `new_category` | an unseen region takes 12% of traffic | flag it without needing statistical support |
| `null_spike` | credit score arrives 35% null | catch it in the **schema** layer, not by PSI on the survivors |

```bash
python -m driftwatch.cli evaluate --scenario all      # delay against false alarms
python -m driftwatch.cli replay  --scenario concept   # full monitor, with delayed labels
```

`evaluate` scores eight rules on identical data - uncorrected per-feature KS, KS + Bonferroni,
KS + BH, a fixed PSI > 0.1, the driftwatch policy, the policy with 2-of-3 confirmation,
Page-Hinkley on the prediction mean, and a labelled AUC drop - and prints, per detector,
`false_alarm_windows`, `false_alarm_rate`, `detected` and `delay_windows`. Read it by column: **a
detector that fires instantly and also fires when nothing happened is not better, it is louder.**
The numbers depend on the sample sizes you pass, which is exactly the point, so they are not
hard-coded here - run the command.

The structural results are asserted in the test suite rather than left to the reader:

- on `concept`, the feature-drift policy detects **nothing** and only the labelled AUC check fires
  (`tests/test_simulate.py::TestEvaluationHarness::test_only_the_labelled_detector_sees_concept_drift`);
- confirmation can only ever reduce false alarms, never add them;
- on `sudden_covariate` the first page lands **no earlier than the injected change point** and at
  most two windows after it, naming `income` / `utilisation` but not `region` or `tenure_months`
  (`tests/test_monitor.py::TestCovariateShift`);
- on `null_spike` the schema layer reports the spike while PSI on the surviving rows stays quiet -
  the honest outcome, and the reason the two layers are separate.

---

## Usage

```bash
pip install -e .            # or: pip install -r requirements.txt
```

### Library

```python
from driftwatch.baseline import fit_baseline
from driftwatch.monitor import Monitor, MonitorConfig

baseline = fit_baseline(
    reference_frame,                 # a stable, representative period
    features=["income", "credit_score", "utilisation", "region"],
    prediction_column="prediction",
    target_column="label",           # optional, but without it there is no quality check
    model_version="risk-model-4.2",
)
baseline.save("artefacts/risk-model-4.2.baseline.json")   # ship it with the model

monitor = Monitor(baseline, config=MonitorConfig(id_column="application_id"))

outcome = monitor.ingest(todays_traffic, window=day)
# {'verdict': 'investigate', 'flagged': ['utilisation'], 'alerts': [...], ...}

monitor.add_labels(ids=matured_ids, labels=matured_outcomes, window=day)
check = monitor.quality(last_windows=7)
reading = monitor.interpret()
```

### CLI

| Command | What it does |
| --- | --- |
| `driftwatch baseline --data reference.csv --out baseline.json` | Fit and pin a reference; prints the expected PSI floor for your window size |
| `driftwatch scan --baseline baseline.json --data window.parquet` | Score one window: schema table, drift table, verdict, what would page |
| `driftwatch evaluate --scenario all` | The detector comparison above |
| `driftwatch replay --scenario concept --label-lag 3` | Full monitor over a scenario, with outcomes arriving late |
| `driftwatch decisions` | Print the drift-versus-quality decision table |
| `driftwatch serve --baseline baseline.json` | Run the HTTP sidecar |

### HTTP sidecar

```bash
docker build -t driftwatch . && docker run -p 8000:8000 driftwatch
```

| Route | Purpose |
| --- | --- |
| `GET /health` | Liveness, and whether a baseline is actually loaded |
| `POST /baseline/fit`, `POST /baseline/load` | Fit from rows, or load a saved artefact |
| `GET /baseline` | The pinned reference as JSON (open bin tails serialise as `null`) |
| `POST /windows`, `GET /windows/{n}` | Score a window; re-read a scored one |
| `POST /labels` | Attach outcomes that arrived later, by id |
| `GET /quality?last_windows=7` | Measured AUC with its CI, coverage and lag |
| `GET /interpretation` | The four-case reading and the recommended action |
| `GET /alerts`, `GET /history`, `GET /summary` | What paged, what happened, where things stand |
| `GET /decisions` | The decision table, served so it cannot be forgotten |

With no baseline loaded, every monitoring route returns **409**, not `200` with nulls: an endpoint
that answers cheerfully while monitoring nothing is the failure mode this repository exists to
argue against. Non-finite statistics cross the boundary as `null`, so the payloads are strict JSON
any client can parse.

---

## Tests

```bash
pytest
```

The suite is the argument, not decoration. It pins:

- **the hand-rolled distributions** against published critical values (chi-square at 1/2/5/9 dof,
  the normal and Kolmogorov tails, and `Q(1,x) = exp(-x)` for the incomplete gamma). Without
  this, every p-value in the service would be unverified;
- **divergence identities**: PSI zero against itself and symmetric, Wasserstein recovering a pure
  translation, TVD and Jensen-Shannon at their bounds, chi-square pooling thin categories;
- **the stored-shares decision**: a feature that is 70% ties scores PSI = 0 against its own
  reference, while assuming uniform bins would report 1.17 - drift on the reference itself;
- **schema severity semantics**, including the deliberate asymmetry that an *empty* window is
  blocking while a merely *thin* one only widens the thresholds;
- **the adaptive floor**, measured on undrifted samples rather than asserted;
- **alert restraint**: one odd window never pages, a persistent problem pages once and escalates
  if it worsens, eight simultaneous flags collapse to three plus a digest, and a broken schema
  bypasses confirmation entirely;
- **the four readings end to end**, including that `concept` produces no feature alerts at all.

---

## Limitations, stated plainly

- **In-process state.** Window history and the label buffer live in memory. The buffer is bounded
  and unmatched predictions are counted rather than silently dropped, but a production deployment
  wants the baseline in object storage and the history in a database. `Monitor` holds no HTTP or
  storage concepts, so that swap never touches the statistics.
- **Binned reference.** No raw reference rows are stored, which keeps the artefact small,
  shareable and free of subject data, but makes every reference-side statistic a discretised one.
- **Univariate.** Correlation drift - margins unchanged, joint structure moved - is invisible
  here. A residual or density-ratio detector is the honest next step, not another marginal test.
- **The prediction-shift watch is conservative.** Page-Hinkley is calibrated on the per-row
  prediction spread, so it reacts to sustained moves rather than to one shifted window; the
  per-feature tests are what catch abrupt changes.
- **Synthetic data.** The scenarios are generated because ground truth is required to measure
  detection delay honestly. They are calibrated to resemble credit-application traffic; they are
  not a claim about any real dataset.

## Layout

```
driftwatch/   stats - metrics - baseline - schema - detect - sequential
              performance - alerts - monitor - simulate - service - cli
tests/        one module per source module, plus end-to-end replays
Dockerfile    non-root monitoring sidecar, no ML training stack
```

MIT licensed - see [LICENSE](LICENSE).
