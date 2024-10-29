'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags
import os
import json

os.sys.path.append("../../")
from llm_attacks import get_goals_and_targets, get_workers, get_refuse_targets

_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def process(raw_data, r):
    res = []
    for i in range(len(raw_data)):
        if (i + 1) % r == 0:
            res.append(raw_data[i])
    return res

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value
    if params.run_defense:
        def_control_init=' '.join(["!"] * params.def_control_len)
    elif '.json' in params.def_control_init:    # 若为json名，则读取json，从中选出def_control
        with open(params.def_control_init, 'r') as file:
            data = json.load(file)
        def_control_init = data["def_controls"]
        def_control_init = process(def_control_init, data["params"]["n_steps"] // data["params"]["test_steps"] + 1)
        print(def_control_init)
    else:
        def_control_init = params.def_control_init

    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)
    train_refuse_targets = get_refuse_targets(params)

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]
    # while len(def_control_init) < len(train_targets): 
    #     def_control_init.append("! ! ! ! ! ! ! ! ! ! ") 

    workers, test_workers = get_workers(params)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            train_refuse_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            def_control_init=def_control_init,
            logfile=f"{params.result_prefix}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
            benign_file=params.benign_file,
        )
    else:
        attack = attack_lib.IndividualPromptAttack(
            train_goals,
            train_targets,
            train_refuse_targets,
            workers,
            control_init=params.control_init,
            def_control_init=def_control_init,
            logfile=f"{params.result_prefix}.json",
            managers=managers,
            test_goals=getattr(params, 'test_goals', []),
            test_targets=getattr(params, 'test_targets', []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        refuse_target_weight=params.refuse_target_weight,
        benign_weight=params.benign_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
        attack_freq=params.attack_freq,
        defense_freq=params.defense_freq,
        run_defense=params.run_defense,
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)
