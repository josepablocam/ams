#!/usr/bin/env bash
source scripts/folder_setup.sh

NUM_COMPONENTS=4
NUM_ASSOC_RULES=1
ALPHA_ASSOC_RULES=0.5
NUM_PARAMS=3
NUM_PARAM_VALUES=3
# we know these are not actual hyperparameters, so just exclude them
EXCLUDE_PARAMS="random_state n_jobs verbose cv"
OUTPUT=/tmp/ams-output.txt

python -m core.generate_search_space \
    --classification \
    --code "$@" \
    --api "${DATA}/sklearn_api.pkl" \
    --rule_mining_model "${DATA}/meta_kaggle_association_rules.pkl" \
    --num_association_rules ${NUM_ASSOC_RULES} \
    --alpha ${ALPHA_ASSOC_RULES} \
    --num_components ${NUM_COMPONENTS} \
    --strategy bm25 \
    --params "${DATA}/params.pkl-summary" \
    --num_params ${NUM_PARAMS} \
    --num_param_values ${NUM_PARAM_VALUES} \
    --exclude_params ${EXCLUDE_PARAMS} \
    --output ${OUTPUT}

cat ${OUTPUT}
