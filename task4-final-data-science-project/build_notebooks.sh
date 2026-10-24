#!/usr/bin/env bash
# Convert the three stage scripts to executed notebooks (outputs and charts included).
# Requires: pip install jupytext nbconvert ipykernel
set -e
cd "$(dirname "$0")"
for name in 01_data_preparation 02_eda 03_modeling; do
  jupytext --to ipynb "scripts/${name}.py" -o "notebooks/${name}.ipynb" -q
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 "notebooks/${name}.ipynb"
done
echo "Notebooks built in notebooks/"
