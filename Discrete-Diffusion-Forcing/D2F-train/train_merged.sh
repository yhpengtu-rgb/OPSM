#!/bin/bash
# Merged optimisation test: KV-cache + dynamic padding + async dual-GPU.
# Training on GPU 4, rollout on GPU 5.
#
# To run sync (single GPU, no async pipeline): set CUDA_VISIBLE_DEVICES=4
# and remove / blank ``async_rollout_device`` in the config.
export HF_HUB_OFFLINE=1
export PATH=/home/cw/workspace/miniconda3/envs/d2f/bin:$PATH
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --config_file config/acc_config_debug \
    --num_processes 1 \
    --main_process_port 29581 \
    train.py --config config/llada_on_policy_merged.yaml
