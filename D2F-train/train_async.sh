#!/bin/bash
# Async pipeline test: training on GPU 4, rollout on GPU 5.
export HF_HUB_OFFLINE=1
export PATH=/home/cw/workspace/miniconda3/envs/d2f/bin:$PATH
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --config_file config/acc_config_debug \
    --num_processes 1 \
    --main_process_port 29580 \
    train.py --config config/llada_on_policy_async.yaml
