#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=25
export num_train_models=2
export model=$1 # llama2 or vicuna or vicuna_guanaco
export len=$2
export benign_weight=$3
export device=$4
export refuse_target_weight=$(echo "1 - ${benign_weight}" | bc)

# Create results folder if it doesn't exist
if [ ! -d "../results/" ]; then
    mkdir "../results/"
    mkdir "../eval/"
    echo "Folder '../results/' created."
else
    echo "Folder '../results/' already exists."
fi

CUDA_VISIBLE_DEVICES=${device},${device+1} python -u ../main.py \
    --config="../configs/transfer_${model}.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix=${log_name} \
    --config.progressive_goals=False \
    --config.stop_on_success=True \
    --config.allow_non_ascii=False \
    --config.num_train_models=$num_train_models \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=100 \
    --config.test_steps=10 \
    --config.batch_size=256 \
    --config.run_defense=True   \
    --config.def_control_len=${len} \
    --config.target_weight=0    \
    --config.refuse_target_weight=${refuse_target_weight} \
    --config.benign_weight=${benign_weight} \
    --config.benign_file="../csv/benign_{}.csv"