#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=25
export model=$1 # llama2 or vicuna
export len=$2
declare -i device=0
export benign_weight=$3
export device=$4
export refuse_target_weight=$(echo "1 - ${benign_weight}" | bc)

# Create results folder if it doesn't exist
if [ ! -d "../results/" ]; then
    mkdir "../results/"
    mkdir "../eval/"
    echo "Folder '../results/' created."
else
    echo "Folder '../results/1-28' already exists."
fi

CUDA_VISIBLE_DEVICES=${device} python -u ../main.py \
    --config="../configs/transfer_${model}_multi.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix=${log_name} \
    --config.progressive_goals=False \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=${n} \
    --config.n_test_data=${n} \
    --config.run_defense=True  \
    --config.n_steps=100 \
    --config.test_steps=10 \
    --config.batch_size=256 \
    --config.def_control_len=${len} \
    --config.target_weight=0    \
    --config.refuse_target_weight=${refuse_target_weight} \
    --config.benign_weight=${benign_weight} \
    --config.benign_file="../../data/benign/benign_${model}.csv" 