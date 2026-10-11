# SWYNEX Internship - Data Science Tasks 1 & 2

**Indian Railways: from messy raw data to evidence-based insights**

By **Pavan**, R.K. College of Engineering | Internship with **SWYNEX Technologies**

This project covers the first two internship tasks as one connected pipeline: Task 1 prepares a raw, messy dataset, and Task 2 explores the cleaned result to find insights that could change a model or a decision.

| Task | Topic | Repository |
|---|---|---|
| **Task 1** | Data Preparation | [SWYNEX-Data-Preparation](https://github.com/pavanthiriveedi7-rgb/SWYNEX-Data-Preparation) |
| **Task 2** | Exploratory Data Analysis | [SWYNEX-Exploratory-Data-Analysis](https://github.com/pavanthiriveedi7-rgb/SWYNEX-Exploratory-Data-Analysis) |

## The pipeline

```
Raw GeoJSON (DataMeet)  ->  Task 1: clean, type, validate  ->  Task 2: explore, test, explain
5,208 trains, 8,990 stations    5,193 trains, 8,975 stations        8 insights with evidence
```

**Dataset:** [DataMeet Indian Railways](https://github.com/datameet/railways), a public community dataset of Indian trains and stations with route geometry.

---

## Task 1 - Data Preparation

**Goal:** clean missing values, handle types, and document assumptions.

I chose a deliberately messy dataset instead of a ready-made one. The raw data had placeholder strings, dummy stations, scraper leftovers and impossible values.

### What I found and how I handled it

| Problem in the raw data | Count | Decision |
|---|---|---|
| Missing values hidden as `"?"`, `"None"` or blank (a plain `isna()` shows 0 missing zones, but 290 are) | 290 zones | Counted placeholders in the audit, converted to real nulls |
| `classes` column completely empty | 5,208 | Dropped; rebuilt from the class flag columns |
| Trains with no times, distance, duration, type or zone | 15 | **Quarantined**, not imputed |
| Zero distance (impossible) | 27 | Set to null and flagged |
| Implausible speeds (<5 or >130 km/h) | 3 | Flagged, not changed |
| Stated distance disagrees with the mapped route | 121 | Flagged for review |
| Slip-coach train numbers (`11039-Slip`, `17603-Slip2`, `14887S`) | 69 | Parsed into `train_id`, `train_no`, `is_slip_coach`, `slip_no` |
| **Valid station code `NAN` matches the placeholder "nan"** | 1 | Protected identifier columns; added a round-trip test |
| Scraper HTML leaked into a `return_train` value | 1 | Repaired to the real number |
| `Orissa` / `Odisha`, `Delhi NCT` | 67, 28 | Unified |
| `XX-` dummy stations | 15 | Quarantined |

### Key decisions

- **Quarantine, don't delete.** Removed rows are saved separately, and a check asserts that clean + quarantined = raw.
- **Imputation must be validated.** I tested filling the 275 unknown railway zones from the origin station. Leave-one-out accuracy was **72.9%**, below my 90% threshold, so I chose **not** to impute.
- **Flag suspicious values instead of silently changing them.**
- **Cross-checks passed:** arrival minus departure equals the stated duration for all 5,192 timed trains, and the stated distance matches the mapped route (median ratio 0.97).

### Result

| | Raw | Clean | Quarantined |
|---|---|---|---|
| Trains | 5,208 | 5,193 | 15 |
| Stations | 8,990 | 8,975 | 15 |

13 automated validation checks pass.

---

## Task 2 - Exploratory Data Analysis

**Goal:** analyse the prepared dataset with statistics and visualisations, and explain at least five insights that could influence a model or decision.

**Guiding question:** *What drives how fast an Indian train runs, and what should a model or a planner pay attention to?*

### Key insights

| # | Insight | Evidence | Model / decision impact |
|---|---|---|---|
| 1 | Distance is heavily right-skewed | Skew 1.86 -> -0.12 after log; mean 548 km vs median 228 km | Use `log(distance)`; keep the 555 long-haul "outliers" (real trains) |
| 2 | Distance, duration and route length are near-duplicates | Spearman 0.92-0.99; speed = distance / duration | Keep one; **never use duration to predict speed** (target leakage) |
| 3 | **Train type drives speed, not zone** | R-squared: type 0.68 vs zone 0.13; adding zone to distance + type adds only +0.031 | Type first, zone secondary; group rare types |
| 4 | Speed rises with distance but saturates | Median 31.7 km/h (<100 km) -> 55.1 km/h (>1500 km); 56% of trains run <300 km | Non-linear: use log, bands or trees; local services touch the most trains |
| 5 | Zone gaps mostly reflect train mix | Raw spread 19.1 km/h; 12.3 within Passenger, 7.8 within Express; still significant (Kruskal p = 2.9e-53 and 7.7e-37) | Include zone *with* type; do not rank zones on raw averages |
| 6 | **Missing zone is informative, not random** | Unknown-zone trains are 95% Passenger/MEMU (vs 51% elsewhere); median 56 km vs 256 km | Keep `Unknown` as a category; do not drop or mode-impute |
| 7 | Distance sets the service regime | Overnight share 1% -> 100% and AC availability 1% -> 97% across distance bands | Use distance bands near 300 and 600 km |
| 8 | Return pairs are a free consistency check | 95.5% of 4,528 pairs agree within 1% on distance; **84 pairs differ by >10%** | Review the mismatches; use pairs to validate imputation |

### Visual highlights

**Speed is organised by train type and distance**

![Speed vs distance](https://raw.githubusercontent.com/pavanthiriveedi7-rgb/SWYNEX-Exploratory-Data-Analysis/main/reports/figures/05_speed_vs_distance.png)

**Stations trace the rail network** (drawn from coordinates only, no map library)

![Station map](https://raw.githubusercontent.com/pavanthiriveedi7-rgb/SWYNEX-Exploratory-Data-Analysis/main/reports/figures/11_station_map.png)

---

## How the two tasks connect

Task 1 decisions directly shaped Task 2 findings:

| Task 1 decision | Task 2 payoff |
|---|---|
| Did not impute the 275 unknown zones | Insight 6: those trains are mostly short local services, so imputing would have been wrong |
| Flagged 3 implausible speeds instead of deleting | Excluded from speed analysis, so a 540 km/h row did not distort the results |
| Protected station code `NAN` | Station joins and the station map stayed complete |
| Built clean `train_id` and distance fields | Enabled the return-pair consistency check (insight 8) |

## Tech stack

Python, pandas, NumPy, SciPy, Matplotlib, Jupyter

## How to run

```bash
# Task 1 - data preparation
git clone https://github.com/pavanthiriveedi7-rgb/SWYNEX-Data-Preparation
cd SWYNEX-Data-Preparation
pip install -r requirements.txt
python data_preparation.py

# Task 2 - exploratory data analysis
git clone https://github.com/pavanthiriveedi7-rgb/SWYNEX-Exploratory-Data-Analysis
cd SWYNEX-Exploratory-Data-Analysis
pip install -r requirements.txt
python eda.py
```

When reading the cleaned CSVs, keep text codes intact:

```python
pd.read_csv("stations_clean.csv", keep_default_na=False, na_values=[""])
```

## Limitations

- **Association, not causation.** The analysis shows what relates to speed, not what causes it.
- **R-squared values are in-sample**, so they describe this dataset, not prediction accuracy on new trains.
- **Speed is timetable speed** (distance / scheduled duration), not measured running speed.
- **Geographic detail is limited:** state is known for only 49% of stations.
- The data is a community snapshot and may not match current railway timetables.

## Key learnings

1. Missing data is not only `NaN`. Placeholder text can hide most of a column.
2. Knowing when **not** to impute is as important as knowing how.
3. Test an assumption before relying on it. "Use the origin station's zone" sounded right but was only 72.9% accurate.
4. The most obvious feature is not always the most important. Zone looked decisive, but train type explained five times more.

## Author

**Pavan Kumar Tiruveedhi** 
Internship: SWYNEX Technologies
