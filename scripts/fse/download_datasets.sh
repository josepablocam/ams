#!/usr/bin/env bash

source scripts/folder_setup.sh

python experiments/download_datasets.py \
  --output "${DATA}/benchmarks-datasets"
