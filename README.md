# SWYNEX Technologies Internship - Projects
 
Three connected tasks on one real-world dataset: **Indian Railways** (5,208 trains, 8,990 stations) from the public [DataMeet](https://github.com/datameet/railways) project. Each task builds on the previous one.
 
```
Task 1: Clean it  ->  Task 2: Understand it  ->  Task 3: Predict with it
```
 
| Task | Goal | Repository |
|---|---|---|
| 1 | Data Preparation | [SWYNEX-Data-Preparation](https://github.com/pavanthiriveedi7-rgb/SWYNEX-Data-Preparation) |
| 2 | Exploratory Data Analysis | [SWYNEX-Exploratory-Data-Analysis](https://github.com/pavanthiriveedi7-rgb/SWYNEX-Exploratory-Data-Analysis) |
| 3 | Modeling and Evaluation | [SWYNEX-Modeling-and-Evaluation](https://github.com/pavanthiriveedi7-rgb/SWYNEX-Modeling-and-Evaluation) |
 
---
 
## Task 1 - Data Preparation
 
**Goal:** turn messy nested GeoJSON into trustworthy tables.
 
- Found missing values hidden as text (`"?"`, `"None"`): a plain `isna()` reported 0 missing zones, but 290 were really missing.
- Protected identifier columns from placeholder replacement (the real station code `NAN` looks like a missing value).
- **Quarantined** 15 empty trains and 15 dummy stations to separate files instead of deleting them; a check confirms `clean + quarantined = raw`.
- Flagged, rather than silently fixing, suspicious values (3 impossible speeds, 121 distance vs route-geometry mismatches).
- Refused to impute `zone` for 275 trains after validation showed only 72.9% accuracy (below the 90% threshold).
- Result: **5,193 clean trains, 8,975 clean stations**, 13 automated assertions passed.
## Task 2 - Exploratory Data Analysis
 
**Goal:** understand what drives train speed and journey time.
 
Main insights that carried into Task 3:
 
1. Distance is right-skewed, so `log(distance)` is the right scale.
2. Never use duration to explain speed (speed is computed from it, so that is leakage).
3. **Train type drives speed far more than zone does.**
4. Speed rises with distance but saturates (a non-linear relationship).
5. Missing zone is informative: it mostly marks short local services.
6. Stated distance and drawn route length are strongly correlated.
Limitation noted in Task 2: its R² values were measured on the same data they were fitted on (in-sample).
 
## Task 3 - Modeling and Evaluation
 
**Goal:** build a baseline predictive model, evaluate it properly, and discuss limitations.
 
- **Problem:** predict scheduled journey time (`duration_min`) from information known at planning time.
- **Leakage control:** speed, arrival time and overnight columns excluded.
- **Group-aware split:** a train and its return train always stay on the same side of the split.
- **Baselines first**, then linear models, random forest and tuned gradient boosting, compared with grouped cross-validation; the test set was used once.
| Model | MAE (min) | Median % error | Within 15% |
|---|---|---|---|
| Median duration (baseline) | 532 | 71.4% | 12% |
| Distance / speed per train type (best baseline) | 74 | 10.2% | 63% |
| **Gradient boosting (tuned)** | **60** | **7.0%** | **77%** |
 
- 19% lower MAE and 31% lower median error than the best hand-made rule.
- Task 2's claims held up on unseen data: train type adds 31 minutes of accuracy, zone only about 5 more.
- Known weaknesses: no stop counts in the data, very long journeys under-predicted, small groups give noisy metrics.
---
 
## How the tasks connect
 
| Task 1 decision or finding | Used again in |
|---|---|
| Keep zone `Unknown`, do not impute | Task 2 (it is informative), Task 3 (modeled as its own category, predicted well) |
| `speed_suspect` flag | Task 3 excludes those rows from training |
| `distance_geo_mismatch` flag | Task 3 checks whether the biggest model misses are data errors |
| Clean CSVs with `keep_default_na=False` | Read the same way in Tasks 2 and 3 |
 
## Tech stack
 
Python, pandas, NumPy, Matplotlib, scikit-learn, Jupyter.
 
## Author
 
Pavan Kumar Tiruveedhi
