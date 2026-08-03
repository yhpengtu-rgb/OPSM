#!/bin/bash
# Batch-size optimisation test: dynamic padding + batch_size=4 + epoch-based.
export HF_HUB_OFFLINE=1
export PATH=/home/cw/workspace/miniconda3/envs/d2f/bin:$PATH
CUDA_VISIBLE_DEVICES=4 accelerate launch \
    --config_file config/acc_config_debug \
    --num_processes 1 \
    --main_process_port 29579 \
    train.py --config config/llada_on_policy_batch.yaml
