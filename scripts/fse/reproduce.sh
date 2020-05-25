#!/usr/bin/env bash
source scripts/folder_setup.sh

# run experiments
bash scripts/fse/reproduce_manual_experiments.sh
bash scripts/fse/reproduce_performance_experiments.sh
bash scripts/fse/reproduce_other_experiments.sh

# produce figures/tables for paper
bash scripts/fse/reproduce_analysis.sh
