#!/usr/bin/env bash
source scripts/folder_setup.sh

# Evaluating algorithm for association rule mining
python experiments/association_experiments.py \
  --input "${DATA}/meta_kaggle_python_with_params.pkl" \
  --k 1 2 3 4 5 \
  --max_len 1 \
  --alpha 0.5 \
  --n_splits 10 \
  --random_state 42 \
  --output "${RESULTS}/rule-mining/"
