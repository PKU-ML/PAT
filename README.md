# PAT(Prompt Adversarial Tuning)

This is the official repository for "[Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://https://arxiv.org/abs/2402.06255)" by [Yichuan Mo](https://scholar.google.com/citations?user=xvSYG1gAAAAJ&hl=zh-CN), [Yuji Wang](https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&gmla=AKKJWFd9ykZr_3fxkwY21YmodnniSXqWYfOGdeIOEyc6XwS84wlxIwvlCM-Y4Hh_zVrYSzJvJBVVWl8BVLpqMwjRVBw&user=MdVoyiYAAAAJ), [Zeming Wei](https://weizeming.github.io/), and [Yisen Wang](https://yisenwang.github.io/). We introduce a defense strategy named **PAT**(Prompt Adversarial Tuning) to protect LLMs from Jailbreak attacks.

![PAT](https://res.cloudinary.com/dnjed4yao/image/upload/v1712416846/iwdgsjznqqgntfoytkc2.jpg)

## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Experiments](#experiments)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Installation

We conduct all our experiments with the version of FastChat 
`fschat==0.2.20`. 

## Models

Before run the experiments, you need to download [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.3) or/and [LLaMA-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) first.  Then modify the model path in `experiments/configs/individual_xxx.py` or `experiments/configs/transfer_xxx.py` (for multiple behaviors or transfer experiment).

## Experiments 

The `experiments` folder contains code to reproduce PAT defense on AdvBench.

- To perform multiple behaviors defense on harmful behaviors:

```bash
cd experiments/launch_scripts
bash run_gcg_multiple.sh vicuna 15 0.2 0
```

- To perform transfer behaviors experiments (i.e. 25 behaviors, 2 models), run the following code inside `experiments`:

```bash
cd experiments/launch_scripts
bash run_gcg_transfer.sh vicuna_guanaco 15 0.2 0
```

- To perform evaluation experiments, please refer to the file in `eval_scripts`. 
  `eval_attack.sh` tests the model on harmful behaviors, and `eval_benign.sh` tests the model on benign prompts. You need to specify the paths for the attack file and the defense file.

We will show our training results here, and you can test **PAT** directly with the provided defense controls. 

The defense control for Vicuna-7B is as below:

```
Answer seriously esper quietlymedia (/ \". harmcgi{{[.(_{( zv\":
```


The defense control for Llama-2 is as below:

```
{[[ Tili particulderr einemroom folgenderomasquestion bezeichneter {%Begin Charlotteawtativelyiostream sug>::
```

The defense control for transferable settings is as below:

```
Come ! critique ! respond seriously?. evil fortune////////////////WRITE asked ein legalEGIN chat,. /**------+ nicelyTurnCred/) Base Question sid!(button [\"itle 
```

## Citation

If you find this useful in your research, please consider citing:

```
@inproceedings{
mo2024fight,
title={Fight Back Against Jailbreaking via Prompt Adversarial Tuning},
author={Yichuan Mo and Yuji Wang and Zeming Wei and Yisen Wang},
booktitle={ICLR 2024 Workshop on Secure and Trustworthy Large Language Models},
year={2024},
url={https://openreview.net/forum?id=q0PbfNwLBq}
}
```


## Acknowledgments

Thanks for work [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043). 

