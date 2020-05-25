#!/usr/bin/env bash

# Generate the data needed for manual marking of
# functionally related component retrieval experiments
# (Figure 4)
# Note that if you execute this script
# it will overwrite the results

source scripts/folder_setup.sh


WARNING_MSG="Running this script will overwrite manual annotation results.\n\
Are you sure you want to execute? [N/Y]"
echo -e ${WARNING_MSG}
read -n 1 -r
echo

if [ $REPLY != "Y" ]
then
  echo "Not executing"
  exit 0
fi

echo "Warning: you are overwriting manual annotation results"

python experiments/relevance_retrieval_api.py \
  --api ${DATA}/sklearn_api.pkl \
  --n 50 \
  --k 10 \
  --seed 42 \
  --method embeddings \
  --output ${MANUAL_RESULTS}/embeddings_relevance.csv

python experiments/relevance_retrieval_api.py \
  --api ${DATA}/sklearn_api.pkl \
  --sampled ${MANUAL_RESULTS}/embeddings_relevance.csv \
  --k 10 \
  --seed 42 \
  --method bm25 \
  --output ${MANUAL_RESULTS}/bm25_relevance.csv

python experiments/relevance_retrieval_api.py \
  --api ${DATA}/sklearn_api.pkl \
  --sampled ${MANUAL_RESULTS}/embeddings_relevance.csv \
  --k 10 \
  --seed 42 \
  --method random \
  --output ${MANUAL_RESULTS}/random_relevance.csv
