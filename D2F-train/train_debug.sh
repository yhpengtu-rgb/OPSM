#!/bin/bash
# Debug smoke-test script for on-policy distillation (LLaDA) on a single GPU.
# Uses a small seq_len=128 config and 3 iterations to verify the pipeline.
export HF_HUB_OFFLINE=1
export PATH=/home/cw/workspace/miniconda3/envs/d2f/bin:$PATH
CUDA_VISIBLE_DEVICES=4 accelerate launch \
    --config_file config/acc_config_debug \
    --num_processes 1 \
    --main_process_port 29578 \
    train.py --config config/llada_on_policy_debug.yaml
