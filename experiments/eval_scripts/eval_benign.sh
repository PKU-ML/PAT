#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=25
export model=$1 # llama2 or vicuna
export device=$2


# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

benign_file="/path/to/benign_file.json"
log_name="/path/to/defense_log.json"

CUDA_VISIBLE_DEVICES=${device} python -u ../evaluate_transfer.py \
    --config=../configs/individual_${model}.py \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.n_train_data=100   \
    --config.data_offset=0   \
    --config.logfile="${log_name}.json" \
    --config.benign_file=${benign_file} \
    --config.run_benign=True
