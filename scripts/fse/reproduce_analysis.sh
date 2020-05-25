#!/usr/bin/env bash
# This script takes outputs produced in other scripts
# (reproduce_*_experiments.sh)
# and may run some additional experiments
# to produce the figures in the FSE paper

source scripts/folder_setup.sh

# Table 2 and Figure 3
mkdir -p "${ANALYSIS_DIR}/rules/"
# Note that this plot is created in reproduce_complementary_experiments.sh
# if using downloaded results, will already be there
cp "${RESULTS}/rule-mining/precision.pdf" "${ANALYSIS_DIR}/rules/"
python analysis/association_rules_analysis.py \
  --model "${DATA}/meta_kaggle_association_rules.pkl" \
  --api "${DATA}/sklearn_api.pkl" \
  --output "${ANALYSIS_DIR}/rules/"


# Figure 4
# note that the *_relevance.csv files
# are manually marked up by the authors
# if you overwrite, you will want to mark them
# yourself as well.
python analysis/relevance_markings.py \
  --inputs \
    manual_results/embeddings_relevance.csv \
    manual_results/bm25_relevance.csv \
    manual_results/random_relevance.csv \
  --names Embeddings BM25 Random \
  --output "${ANALYSIS_DIR}/relevance/plot.pdf"


# Figures 5 and 6
# Note that this actually runs new experiments
# for hyper-parameter data characterization
# so just be aware it may take an hour or two to run
mkdir -p "${ANALYSIS_DIR}/hyperparams"
python analysis/distribution_hyperparameters.py \
  --params_raw "${DATA}/params.pkl-raw" \
  --params_summary "${DATA}/params.pkl-summary" \
  --num_clfs 5 \
  --num_params 3 \
  --num_values 3 \
  --scoring f1_macro \
  --datasets diabetes heart-c hepatitis dermatology liver-disorder \
  --api "${DATA}/sklearn_api.pkl" \
  --output "${ANALYSIS_DIR}/hyperparams"


# Figure 7: Genetic Programming
python analysis/performance_analysis.py \
  --input "${RESULTS}/tpot/" \
  --output "${ANALYSIS_DIR}/performance/tpot/" \
  --filter_incomplete \
  --systems sys manual spec manual_components_only \
  --performance_metric "Macro-Avg. F1 Score" \
  --min_win_diff 0.01

# Figure 7: Random Search
python analysis/performance_analysis.py \
  --input "${RESULTS}/random/" \
  --output "${ANALYSIS_DIR}/performance/random/" \
  --filter_incomplete \
  --systems sys manual spec manual_components_only \
  --performance_metric "Macro-Avg. F1 Score" \
  --min_win_diff 0.01

# Figure 7: Combine results into single plot from paper
python analysis/combined_wins_plot.py \
  --input "${ANALYSIS_DIR}/performance/tpot/wins.csv" "${ANALYSIS_DIR}/performance/random/wins.csv" \
  --search "Genetic Programming" "Random Search" \
  --output "${ANALYSIS_DIR}/performance/combined_wins.pdf"


# Figure 8
mkdir -p "${ANALYSIS_DIR}/tpot-sys-ops/"
python analysis/frequency_operators.py \
    --input results/tpot/ \
    --name sys \
    --k 10 \
    --output "${ANALYSIS_DIR}/tpot-sys-ops/" \
    --specs "${RESULTS}" \
    --title "Spec: {}" \
    --combine
