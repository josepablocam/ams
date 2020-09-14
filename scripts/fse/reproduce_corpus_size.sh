#!/usr/bin/env bash
# Runs all corpus size experiments for figure 9
# For ease of analysis, we have included the results from the FSE
# paper in pre-packaged form.

source scripts/folder_setup.sh

mkdir -p ${RESULTS}

SEED=42
N_PROC=10
tsp -S ${N_PROC}

CORPUS_SIZES[0]=0.1
CORPUS_SIZES[1]=0.2
CORPUS_SIZES[2]=0.3
CORPUS_SIZES[3]=0.4
CORPUS_SIZES[4]=0.5
CORPUS_SIZES[5]=0.6
CORPUS_SIZES[6]=0.7
CORPUS_SIZES[7]=0.8
CORPUS_SIZES[8]=0.9
N_ITERS=5

for corpus_size_ix in ${!CORPUS_SIZES[@]}
do
    corpus_size=${CORPUS_SIZES[${corpus_size_ix}]}

    for corpus_iter in $(seq 1 ${N_ITERS})
    do
          folder_suffix="/corpus-size-${corpus_size}-iter-${corpus_iter}/"
          sampled_data_folder="${DATA}/${folder_suffix}"
          mkdir -p ${sampled_data_folder}

          tsp python experiments/build_corpus_size_experiment.py \
            --orig_data_folder ${DATA} \
            --sampled_data_folder ${sampled_data_folder} \
            --sample_rate ${corpus_size} \
            --seed ${SEED}
    done
done
