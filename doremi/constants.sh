#!/bin/bash

# This is a sample of the constants file. Please write any env variables here
# and rename the file to constants.sh

CACHE=/root/diplom_doremi/doremi/cache
DOREMI_DIR=/root/diplom_doremi/doremi
PILE_DIR=/root/diplom_doremi/doremi/data
PREPROCESSED_PILE_DIR=/root/diplom_doremi/doremi/preprocessed  # will be created by scripts/run_filter_domains.sh
MODEL_OUTPUT_DIR=/root/diplom_doremi/doremi/model_output
WANDB_API_KEY=key  # Weights and Biases key for logging
PARTITION=LocalQ # for slurm
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
source ${DOREMI_DIR}/venv/bin/activate  # if you installed doremi in venv
export WANDB_API_KEY=${WANDB_API_KEY}

alias python=python3
export MAX_JOBS=4
