#!/usr/bin/env bash
source scripts/folder_setup.sh

# Build up embeddings for API
python -m core.extract_sklearn_api \
    --output "${DATA}/sklearn_api.pkl"


# Extract meta kaggle scripts that import sklearn
python -m core.extract_kaggle_scripts \
    --input "${DATA}/meta-kaggle" \
    --output "${DATA}/meta_kaggle_python.pkl" \
    --api sklearn


# Extract call arguments from python scripts
python -m core.extract_parameters \
  --input "${DATA}/meta_kaggle_python.pkl" \
  --output "${DATA}/meta_kaggle_python_with_params.pkl" \
  --api "${DATA}/sklearn_api.pkl"


# Summarize the parameters in a table with sampling distributions
python -m core.summarize_parameters \
  --input "${DATA}/meta_kaggle_python_with_params.pkl" \
  --api "${DATA}/sklearn_api.pkl" \
  --output "${DATA}/params.pkl"


# Extract association rules for API calls from python scripts
python -m core.mine_association_rules \
  --input "${DATA}/meta_kaggle_python_with_params.pkl" \
  --output "${DATA}/meta_kaggle_association_rules.pkl"
