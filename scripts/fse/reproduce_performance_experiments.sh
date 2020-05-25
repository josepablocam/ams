#!/usr/bin/env bash
# Runs all performance experiments for figure 7
# This can take a significant amount of time to execute
# And note that given the multiprocessing libraries used
# there are occasional contentions that can result in deadlocks
# In such a case, the easiest fix is to kill the run and restart it
# For ease of analysis, we have included the results from the FSE
# paper in pre-packaged form.

source scripts/folder_setup.sh

mkdir -p ${RESULTS}

TPOT_DATASETS="Hill_Valley_without_noise "
TPOT_DATASETS+="Hill_Valley_with_noise "
TPOT_DATASETS+="breast-cancer-wisconsin "
TPOT_DATASETS+="car-evaluation "
TPOT_DATASETS+="glass "
TPOT_DATASETS+="ionosphere "
TPOT_DATASETS+="spambase "
TPOT_DATASETS+="wine-quality-red "
TPOT_DATASETS+="wine-quality-white "

# ams params
NUM_COMPONENTS=4
NUM_ASSOC_RULES=3
ALPHA_ASSOC_RULES=0.5
NUM_PARAMS=3
NUM_PARAM_VALUES=3

# evaluation params
MAX_TIME_MINS=5
CV=5
SEED=42
N_JOBS=1
# number of processes to use to run experiments
# set to 30 (our server has 40 CPUs)
N_PROC=20
SCORING_FUN="f1_macro"

# logistic regression
QUERIES[0]="lr"
QUERIES[1]="scale lr"
QUERIES[2]="poly scale lr"
QUERIES[3]="poly scale var lr"
QUERIES[4]="poly scale pca var lr"

# random forest
QUERIES[5]="rf"
QUERIES[6]="scale rf"
QUERIES[7]="poly scale rf"
QUERIES[8]="poly scale var rf"
QUERIES[9]="poly scale pca var rf"

# decision tree
QUERIES[10]="dt"
QUERIES[11]="scale dt"
QUERIES[12]="poly scale dt"
QUERIES[13]="poly scale var dt"
QUERIES[14]="poly scale pca var dt"

# run using task-spooler
# https://vicerveza.homeunix.net/~viric/soft/ts/
tsp -S ${N_PROC}

for query_id in ${!QUERIES[@]}
do
    query_folder="${RESULTS}/q${query_id}"
    query=${QUERIES[${query_id}]}
    echo "Generating configurations for folder: ${query_folder}"

    # generate the weak specifications used in experiments
    python experiments/generate_experiment.py \
      --experiment ${query} \
      --output "${query_folder}" \
      --use_tpot_hyperparams # use expert-defined hyperparameters

    # generate search space from code query using AMS
    python -m core.generate_search_space \
        --classification \
        --code "${query_folder}/code.txt" \
        --api "${DATA}/sklearn_api.pkl" \
        --rule_mining_model "${DATA}/meta_kaggle_association_rules.pkl" \
        --num_association_rules ${NUM_ASSOC_RULES} \
        --alpha ${ALPHA_ASSOC_RULES} \
        --num_components ${NUM_COMPONENTS} \
        --strategy bm25 \
        --params "${DATA}/params.pkl-summary" \
        --num_params ${NUM_PARAMS} \
        --num_param_values ${NUM_PARAM_VALUES} \
        --exclude_params random_state n_jobs verbose cv \
        --output "${query_folder}/gen_config.json"
done


for query_id in ${!QUERIES[@]}
do
    for search in tpot random
    do
        query_folder="${RESULTS}/q${query_id}"
        output_folder="${RESULTS}/${search}/q${query_id}"
        echo "Running experiments for folder: ${output_folder}"
        # in case doesn't exist
        mkdir -p ${output_folder}

        # run weak spec directly as a pipeline
        # Figure 7: Weak Spec.
        tsp python experiments/run_experiment.py \
            --search simple \
            --dataset ${TPOT_DATASETS} \
            --cv ${CV} \
            --random_state ${SEED} \
            --name "spec" \
            --scoring ${SCORING_FUN} \
            --n_jobs ${N_JOBS} \
            --output "${output_folder}/spec_results.pkl" \
            --config "${query_folder}/simple_config.json"


        # run search strategy with only the components specified in spec
        # Figure 7: Weak Spec. + Search
        tsp python experiments/run_experiment.py \
            --search ${search} \
            --components_only \
            --dataset ${TPOT_DATASETS} \
            --cv ${CV} \
            --max_time_mins ${MAX_TIME_MINS} \
            --random_state ${SEED} \
            --name "manual_components_only" \
            --scoring ${SCORING_FUN} \
            --n_jobs ${N_JOBS} \
            --output "${output_folder}/manual_components_only_results.pkl" \
            --config "${query_folder}/simple_config_with_params_dict.json"


        # run manually written expert configuration for hyperparams with search strategy
        # Figure 7: Expert + Search
        tsp python experiments/run_experiment.py \
            --search ${search} \
            --dataset ${TPOT_DATASETS} \
            --cv ${CV} \
            --max_time_mins ${MAX_TIME_MINS} \
            --random_state ${SEED} \
            --name "manual" \
            --scoring ${SCORING_FUN} \
            --n_jobs ${N_JOBS} \
            --output "${output_folder}/manual_results.pkl" \
            --config "${query_folder}/simple_config_with_params_dict.json"


        # run search over automatically generated configuration (i.e our system)
        # Figure 7: AMS + search
        tsp python experiments/run_experiment.py \
            --search ${search} \
            --dataset ${TPOT_DATASETS} \
            --cv ${CV} \
            --max_time_mins ${MAX_TIME_MINS} \
            --random_state ${SEED} \
            --name "sys" \
            --scoring ${SCORING_FUN} \
            --n_jobs ${N_JOBS} \
            --output "${output_folder}/sys_results.pkl" \
            --config "${query_folder}/gen_config.json"
      done
done
