# SWYNEX-Modeling-and-Evaluation

**SWYNEX Technologies internship - Task 3: Modeling and Evaluation**

Predicting the **scheduled journey time of an Indian train** (`duration_min`) from information known at planning time, with honest baselines, grouped cross-validation, a held-out test set and a discussion of where the model fails.

**Main file: [`modeling_and_evaluation.ipynb`](modeling_and_evaluation.ipynb)** (executed, with all outputs).

## Where this fits

| Task | Repository | What it contributed |
|---|---|---|
| 1 - Data Preparation | [SWYNEX-Data-Preparation](https://github.com/pavanthiriveedi7-rgb/SWYNEX-Data-Preparation) | Clean tables, quality flags (`speed_suspect`, `distance_geo_mismatch`) |
| 2 - EDA | [SWYNEX-Exploratory-Data-Analysis](https://github.com/pavanthiriveedi7-rgb/SWYNEX-Exploratory-Data-Analysis) | Insights: log distance, type beats zone, unknown zone is informative |
| **3 - Modeling** | this repo | Tests those insights out of sample |

All three live together in [SWYNEX_Projects_tasks](https://github.com/pavanthiriveedi7-rgb/SWYNEX_Projects_tasks).

## The problem

Supervised **regression**. Target: `duration_min`. Features known when a service is planned: distance, route length, train type, zone, coach classes, departure time, origin/destination coordinates.
Dataset: [DataMeet Indian Railways](https://github.com/datameet/railways), cleaned in Task 1 (5,163 usable trains).

## Method (and the traps avoided)

- **Leakage control.** `avg_speed_kmph`, `arrival_min`, `is_overnight`, `days_en_route` are derived from the target and are excluded.
- **Group-aware split.** A train and its return train run the same route with near-identical times, so both always go to the same side of the split (`GroupShuffleSplit` and `GroupKFold`). A plain random split would be optimistic.
- **Baselines first:** median duration, distance / global speed, distance / speed *per train type*.
- **Models:** linear (distance, log distance, all features), random forest, gradient boosting (tuned with grouped CV).
- **Test set touched once**, after model selection by cross-validation.

## Results (held-out test set, 1,062 unseen trains)

| Model | MAE (min) | Median % error | Within 15% |
|---|---|---|---|
| Baseline 1: median duration | 532 | 71.4% | 12% |
| Baseline 2: distance / global speed | 227 | 24.4% | 32% |
| Linear (distance only) | 112 | 18.6% | 43% |
| Baseline 3: distance / speed per type (strongest hand-made rule) | 74 | 10.2% | 63% |
| Random forest | 64 | 7.7% | 75% |
| **Gradient boosting (tuned)** | **60** | **7.0%** | **77%** |

Versus the best hand-made baseline: **-19% MAE, -31% median error**, 14 more points of trains within 15%.
R2 is about 0.98 for the best models but is not very informative here (the baseline already scores 0.97), so relative-error metrics are used.

## Key findings

1. Most of the signal is **distance + train type**. Adding train type cuts MAE by 31 min, adding zone only 5 more (Task 2's "type beats zone", now confirmed on held-out data).
2. Coach-class and departure-time features add nothing measurable.
3. Long journeys are more predictable in **relative** terms (4-5% error above 700 km) than short ones (8-9% below 300 km), though absolute error is larger.
4. Trains with unknown zone (short local services) are predicted as well as the rest, supporting the Task 1 decision not to impute zone.
5. The largest misses are a ready-made **review list**: for example a 70 km train scheduled for 365 minutes is either very slow or a data error.

## Limitations

- **Number of stops is missing from the data**, and it likely explains much of the remaining error.
- Targets are the *scheduled* time, not real-world delays; the data is a community snapshot.
- The split is by route group but all trains come from one snapshot (no true "future" test).
- Very long journeys (60+ hours) are under-predicted; tree models do not extrapolate.
- Small groups (rare train types, 53 unknown-zone trains) give noisy metrics; the final number comes from a single split.
- Feature importance describes the model, not causes.

## Project structure

```
SWYNEX-Modeling-and-Evaluation/
├── data/processed/
│   ├── trains_clean.csv         # from Task 1
│   └── stations_clean.csv       # from Task 1
├── reports/
│   ├── model_comparison.png
│   ├── predicted_vs_actual.png
│   ├── feature_importance.png
│   ├── target_distribution.png
│   └── test_metrics.csv
├── modeling_and_evaluation.ipynb
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook modeling_and_evaluation.ipynb
```

The notebook reads the cleaned CSVs with `keep_default_na=False, na_values=[""]` so the real station code `NAN` is not mistaken for a missing value.

## Author

Pavan Kumar Tiruveedhi
