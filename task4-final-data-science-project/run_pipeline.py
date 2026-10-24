"""Run the full case study end to end: prepare -> explore -> model.

    python run_pipeline.py

Each stage is a plain Python script (cell-formatted, so it also opens as a notebook).
Raw data is downloaded automatically on the first run.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("Stage 1: data preparation", "scripts/01_data_preparation.py"),
    ("Stage 2: exploratory data analysis", "scripts/02_eda.py"),
    ("Stage 3: modeling and evaluation", "scripts/03_modeling.py"),
]

env = {**os.environ, "MPLBACKEND": "Agg"}  # no pop-up windows; figures are saved to reports/figures/
total = time.time()
for title, script in STAGES:
    print(f"\n=== {title} ({script}) ===", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"{title} failed (exit code {result.returncode}). Fix the error above and re-run.")
    print(f"--- done in {time.time() - start:.0f}s", flush=True)

print(f"\nAll stages finished in {time.time() - total:.0f}s.")
print("Outputs: data/processed/, reports/, models/")
