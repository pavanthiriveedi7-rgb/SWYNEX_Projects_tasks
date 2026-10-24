# SWYNEX-Final-Data-Science-Project

**What really drives the speed of an Indian train, and can we predict it honestly?**
An end-to-end data science case study: problem statement, data preparation, EDA, modeling, and conclusions.

*SWYNEX Technologies internship, final project. By Pavan, R.K. College of Engineering.*

![Project summary](docs/project_summary.png)

## Executive summary

- **Data:** 5,208 trains and 8,990 stations from the public [DataMeet Indian Railways](https://github.com/datameet/railways) dataset, delivered as messy nested GeoJSON.
- **Preparation:** found 290 missing zones hidden as `"?"` and blank text (a plain `isna()` shows zero), quarantined 15 unusable records instead of deleting them, and refused to impute a field whose imputation tested at only 72.9% accuracy.
- **Finding:** **train type explains 68% of speed variation; railway zone only 13%.** Zone gaps that look large mostly reflect a different mix of train types.
- **Model:** a Random Forest predicts a train's scheduled speed with **R² 0.82 and an average error of 3.9 km/h** on unseen trains (baseline: R² 0 and 10.7 km/h). Predicted journey time is within 15% of the truth for **80% of trains** (baseline: 34%).
- **Honest evaluation:** a leaky model scored 0.97 and a random split inflated scores by 0.04. Both were caught and corrected.
- **Feedback loop:** the model's biggest errors revealed unreliable records, which improved the data-preparation rule. 137 records are flagged for review.

## Contents

1. [Problem statement](#1-problem-statement)
2. [Data](#2-data)
3. [Stage 1: Data preparation](#3-stage-1-data-preparation)
4. [Stage 2: Exploratory data analysis](#4-stage-2-exploratory-data-analysis)
5. [Stage 3: Modeling and evaluation](#5-stage-3-modeling-and-evaluation)
6. [The data-quality feedback loop](#6-the-data-quality-feedback-loop)
7. [Conclusions and recommendations](#7-conclusions-and-recommendations)
8. [Limitations and future work](#8-limitations-and-future-work)
9. [Decision log: what I got wrong and fixed](#9-decision-log-what-i-got-wrong-and-fixed)
10. [Reproduce this project](#10-reproduce-this-project)

---

## 1. Problem statement

**Context.** Planning a new train service means estimating how long the journey will take. That depends on how fast the train will run, which depends on what kind of train it is, how far it goes, and where it runs.

**Questions**

| # | Question |
|---|---|
| Q1 | What drives the average scheduled speed of a train? |
| Q2 | Can speed (and from it, journey time) be predicted from information known *before* the timetable is fixed? |
| Q3 | Which records in the dataset are too unreliable to trust? |
| Q4 | What should planners, data teams and model builders do with these findings? |

**Success criteria (set before modeling)**

1. Beat a naive baseline (always guess the median speed) by a clear margin.
2. Evaluate honestly: no target leakage, no near-duplicate rows straddling train and test, uncertainty reported.
3. Make every assumption and limitation explicit.
4. Fully reproducible from one command.

## 2. Data

| | |
|---|---|
| Source | [DataMeet Indian Railways](https://github.com/datameet/railways): `trains.json`, `stations.json` |
| Size | 5,208 trains, 8,990 stations (GeoJSON with route geometry) |
| Target | `avg_speed_kmph` = distance / scheduled duration (timetable speed, not measured speed) |
| Why this dataset | India-specific, rarely used in tutorials, and genuinely messy |

Raw files are downloaded automatically by the pipeline.

## 3. Stage 1: Data preparation

Notebook: [`notebooks/01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) | Script: [`scripts/01_data_preparation.py`](scripts/01_data_preparation.py)

### What the raw data contained

| Problem | Count | Decision |
|---|---|---|
| Missing values hidden as `"?"`, `"None"` or blank | 290 zones; 15 each for times, distance, type | Counted placeholders in the audit; converted to real nulls |
| `classes` column completely empty | 5,208 | Dropped; rebuilt from six class-flag columns |
| Trains with no times, distance, duration, type or zone | 15 | **Quarantined**, not imputed |
| Zero distance (impossible) | 27 | Set to null and flagged |
| Implausible speeds (< 5 or > 130 km/h) | 3 | Flagged, excluded from speed analysis |
| Slip-coach numbers (`11039-Slip`, `17603-Slip2`, `14887S`) | 69 | Parsed into `train_id`, `train_no`, `is_slip_coach`, `slip_no` |
| **Valid station code `NAN` matches the placeholder "nan"** | 1 | Identifier columns protected from placeholder logic; round-trip test added |
| Web-scraper HTML leaked into a `return_train` value | 1 | Repaired to the real number |
| `Orissa` / `Odisha`, `Delhi NCT`, Bangladesh stored as a "state" | 67 / 28 / 2 | Unified; Bangladesh moved to a `country` column |
| `XX-` dummy stations | 15 | Quarantined |

![Hidden missing values](reports/figures/prep/hidden_missing_values.png)

### Key decisions

- **Quarantine, don't delete.** Removed rows go to separate files, and an assertion checks that clean + quarantined = raw.
- **Imputation must be validated.** For the 275 trains with unknown zone, I tested "use the most common zone of other trains from the same origin station" with leave-one-out validation. Accuracy was **72.9%**, below my 90% threshold, so **I did not impute**.
- **Cross-checks passed:** arrival minus departure equals the stated duration for all 5,192 timed trains; stated distance matches the mapped route (median ratio 0.97).
- 13 automated validation checks pass.

| | Raw | Clean | Quarantined |
|---|---|---|---|
| Trains | 5,208 | 5,193 | 15 |
| Stations | 8,990 | 8,975 | 15 |

## 4. Stage 2: Exploratory data analysis

Notebook: [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb) | Script: [`scripts/02_eda.py`](scripts/02_eda.py) | 11 charts in `reports/figures/eda/`

| # | Insight | Evidence | Model / decision impact |
|---|---|---|---|
| 1 | Distance is heavily right-skewed | Skew 1.86 -> -0.12 after log; mean 548 km vs median 228 km | Use `log(distance)`; keep long-haul "outliers" (real trains) |
| 2 | Distance, duration and route length are near-duplicates | Spearman 0.92-0.99; speed = distance / duration | Keep one; **never use duration to predict speed** (leakage) |
| 3 | **Train type drives speed, not zone** | R²: type 0.68 vs zone 0.13 | Type first; zone is a secondary refinement |
| 4 | Speed rises with distance but saturates | Median 31.7 km/h (< 100 km) -> 55.1 km/h (> 1500 km); 56% of trains run < 300 km | Non-linear: log, bands or trees |
| 5 | Zone gaps mostly reflect train mix | Raw spread 19.1 km/h; 12.3 within Passenger, 7.8 within Express; still significant (p < 1e-35) | Include zone *with* type; do not rank zones on raw averages |
| 6 | **Missing zone is informative** | Unknown-zone trains are 95% Passenger/MEMU (vs 51% elsewhere); median 56 km vs 256 km | Keep `Unknown` as a category; do not drop or mode-impute |
| 7 | Distance sets the service regime | Overnight 1% -> 100%; AC availability 1% -> 97% across distance bands | Distance bands near 300 and 600 km |
| 8 | Return pairs are a free consistency check | 95.5% of 4,528 pairs agree within 1%; 84 differ by > 10% | Review the mismatches |

![Speed vs distance](reports/figures/eda/05_speed_vs_distance.png)

![Station map](reports/figures/eda/11_station_map.png)

## 5. Stage 3: Modeling and evaluation

Notebook: [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Script: [`scripts/03_modeling.py`](scripts/03_modeling.py) | 7 charts in `reports/figures/model/`

### Setup

- **Target:** `avg_speed_kmph` (5,163 usable trains)
- **Features:** `log(distance)`, train type (rare types pooled), zone (with `Unknown`), route points, class availability, departure hour, has-return-train
- **Deliberately excluded (leakage audit):** `duration_min`, arrival time, `is_overnight`, `days_en_route` (all rebuild the target), identifiers, and station codes
- **Validation:** 5-fold **grouped** cross-validation, where each train and its return/slip family are kept together (5,163 trains in 3,102 groups), then a held-out 20% of groups and bootstrap confidence intervals

### Model comparison (grouped 5-fold CV)

| Model | R² | MAE (km/h) |
|---|---|---|
| Dummy (always the median) | -0.01 | 10.47 |
| Linear: log(distance) only | 0.47 | 7.25 |
| Linear: distance + train type | 0.71 | 5.20 |
| Linear: all features | 0.75 | 4.87 |
| Hist Gradient Boosting | 0.80 | 4.19 |
| **Random Forest (selected)** | **0.81** | **4.06** |

### Held-out test result

| | Baseline | Random Forest |
|---|---|---|
| R² | -0.03 | **0.82** (95% CI 0.79 to 0.84) |
| MAE | 10.7 km/h | **3.9 km/h** (95% CI 3.65 to 4.14) |
| Journey-time error (median) | 23.6% | **6.6%** |
| Trains within 15% of true journey time | 34% | **80%** |

![Predicted vs actual](reports/figures/model/03_predicted_vs_actual.png)

### Two ways a score can lie

| | Random split R² | Grouped split R² | Inflation |
|---|---|---|---|
| Linear | 0.748 | 0.746 | +0.002 |
| Hist Gradient Boosting | 0.830 | 0.800 | +0.030 |
| Random Forest | 0.846 | 0.805 | +0.041 |

And a deliberately **leaky** model given `duration_min` scores **R² 0.968**, which looks brilliant but is useless because duration is unknown when planning a new service.

![Leakage checks](reports/figures/model/02_leakage_checks.png)

### What the model relies on

| Variant (retrained) | R² | Change |
|---|---|---|
| Full model | 0.805 | - |
| Without train type | 0.724 | -0.081 |
| Without zone | 0.769 | -0.036 |
| Without type and zone | 0.677 | -0.128 |
| Without class features | 0.802 | -0.003 |

This confirms Stage 2: **type matters most, zone is a smaller refinement.** Permutation importance ranked `n_classes` third, yet removing all class features costs only 0.003 R². Class availability is tied to type and distance, so the model rebuilds the signal; shuffling a correlated feature exaggerates its importance.

### Where the model struggles

| Finding | Evidence |
|---|---|
| Shrinks extremes toward the average | Trains above 65 km/h are under-predicted by 7.1 km/h (n = 38); trains under 30 km/h are over-predicted by 3.9 km/h |
| Small train types are unreliable | Garib Rath 9.2 km/h (n = 10), Duronto 6.9 (n = 10), pooled "Other" 9.7 (n = 12) |
| Best where data is plentiful | Superfast 2.5, Express 3.7, Passenger 4.2 km/h |

## 6. The data-quality feedback loop

The largest model errors included records that looked wrong, such as a toy train listed at 126 km running at 75 km/h. I tested this with out-of-fold predictions for every train and the geometry ratio from Stage 1 (mapped route length / stated distance):

| Tail | Trains | Model error (MAE) | Verdict |
|---|---|---|---|
| ratio < 0.8 (distance much longer than mapped route) | 136 | 8.0 vs 3.95 km/h (Mann-Whitney p = 4.5e-15) | **Informative: flag** |
| ratio > 1.2 (route much longer than stated distance) | 92 | 4.6 vs 3.94 km/h (p = 0.34) | Not significant: not flagged |

Trains with ratio < 0.8 are 2.6% of the data but **20% of the 100 largest errors** (7.6x over-represented). This fed back into Stage 1: the final rule flags **ratio < 0.8**, giving **137 review candidates**, and **22 high-priority** ones (also mispredicted by > 15 km/h) are in [`reports/model/data_quality_review_candidates.csv`](reports/model/data_quality_review_candidates.csv).

![Errors vs geometry](reports/figures/model/07_errors_vs_geometry.png)

*Caution:* a low ratio can also come from a simplified route line, so these are review candidates, not proven errors. The threshold was chosen after seeing this analysis, and the flag is never used as a model feature, so it does not affect any reported metric.

## 7. Conclusions and recommendations

### Answers

| Question | Answer |
|---|---|
| **Q1** What drives speed? | Train type (68% alone), then distance (47% alone, non-linear). Zone adds little once type and distance are known (+0.03 to +0.04 R²). |
| **Q2** Can it be predicted? | Yes. R² 0.82 and a 3.9 km/h average error on unseen trains; journey time within 15% for 80% of trains versus 34% for a naive guess. |
| **Q3** Which records are unreliable? | 137 trains whose stated distance is much longer than their mapped route (22 high priority); 84 return-train pairs that disagree on distance by > 10%. |
| **Q4** What to do? | See below. |

### Recommendations

1. **For planners estimating a new service's journey time:** use train type and distance first; expect a typical error of about 4 km/h (median about 27 minutes), and treat extremes (very fast or very slow services) with extra caution.
2. **For prioritising improvements:** short local services (under 300 km) are 56% of all trains and the slowest. This is an association, not proof that changes would work, but it is where most trains are.
3. **For the data team:** review the 22 high-priority records first, then the other flagged ones and the 84 mismatched return pairs.
4. **For anyone building models on this data:** never use duration or arrival time to predict speed, split by train family rather than randomly, and keep `Unknown` zone as its own category.

## 8. Limitations and future work

**Limitations**

- **Association, not causation.** The model predicts speed; it does not explain why trains are fast.
- **Target is timetable speed**, not measured running speed, so delays are not captured.
- **Shrinks extremes,** and rare train types are too small for reliable conclusions.
- **Limited tuning.** I compared model families with sensible defaults rather than a large hyperparameter search.
- **Snapshot data.** One timetable snapshot, so no time-based validation; future performance is unknown.
- **Grouping is partial protection.** Trains on the same corridor still resemble each other, so performance on a brand-new route may be lower.
- **Noisy target.** Records with questionable distances add error no model can remove.
- **Geographic detail is limited:** state is known for only 49% of stations.
- 30 trains (3 implausible speeds, 27 with no valid distance) are not modeled.

**Future work**

- Add stop-level information (for example stops per km from the stop-level schedule data DataMeet also publishes), the most promising missing feature.
- Validate on a later timetable snapshot.
- Tune models properly and try per-train-type models.
- Predict with uncertainty intervals.

## 9. Decision log: what I got wrong and fixed

| What happened | What I did |
|---|---|
| A random split inflated Random Forest R² from 0.805 to 0.846 | Switched to grouped splits and reported both numbers |
| My placeholder list deleted the valid station code `NAN` | Protected identifier columns; added a round-trip test |
| I assumed a pandas CSV read would null `NAN` (it does not for uppercase) | Tested it, corrected the comment and README |
| I claimed class features add little (Stage 2) but permutation importance said otherwise | Ran an ablation: the claim held (-0.003 R²), and I documented why the two tools disagree |
| I expected the largest errors to match my Stage 1 distance flag; they mostly did not (5% vs 2.3% base rate) | Re-examined with a different threshold, found the real signal (ratio < 0.8) |
| My first flag threshold (ratio < 0.33) was too loose; I also kept an untested upper tail (> 1.2) | Tightened to 0.8 and **dropped** the upper tail after testing it (p = 0.34) |
| A chart title claimed "no obvious bias" | Measured the bias instead and replaced the title |

## 10. Reproduce this project

```bash
pip install -r requirements.txt
python run_pipeline.py          # runs all three stages (about 3 minutes)
```

To rebuild the executed notebooks:

```bash
./build_notebooks.sh
```

Raw data downloads automatically on the first run. All stage outputs are deterministic (fixed random seeds).

### Project structure

```
SWYNEX-Final-Data-Science-Project/
├── README.md                         # this case study
├── run_pipeline.py                   # one command runs everything
├── build_notebooks.sh                # scripts -> executed notebooks
├── requirements.txt
├── scripts/
│   ├── 01_data_preparation.py
│   ├── 02_eda.py
│   └── 03_modeling.py
├── notebooks/                        # executed, with outputs and charts
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── data/
│   ├── raw/                          # downloaded automatically (not committed)
│   └── processed/                    # clean + quarantined tables
├── reports/
│   ├── prep/  eda/  model/           # result tables (CSV)
│   └── figures/prep  eda  model/     # 20 charts
├── models/speed_model.joblib         # final model, trained on all usable trains (11 MB)
└── docs/project_summary.png
```

Using the saved model:

```python
import joblib
model = joblib.load("models/speed_model.joblib")
# columns: log_distance, route_points, dep_hour, has_return, has_ac, sleeper,
#          chair_car, n_classes, type_grp, zone
speed_kmph = model.predict(new_trains_dataframe)
```

When reading the cleaned CSVs, keep text codes intact:

```python
pd.read_csv("data/processed/stations_clean.csv", keep_default_na=False, na_values=[""])
```

## Author

**Pavan** - R.K. College of Engineering | Internship: SWYNEX Technologies
