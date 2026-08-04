#!/bin/bash
# DMD on-policy distillation: Gumbel-free differentiable rollout + EMA fake
# model + density-ratio loss.
#
# Single GPU (DMD rollout is differentiable and CANNOT use the async
# dual-GPU pipeline, which runs rollout under no_grad on a separate GPU).
export HF_HUB_OFFLINE=1
export PATH=/home/cw/workspace/miniconda3/envs/d2f/bin:$PATH
CUDA_VISIBLE_DEVICES=4 accelerate launch \
    --config_file config/acc_config_debug \
    --num_processes 1 \
    --main_process_port 29582 \
    train.py --config config/llada_on_policy_dmd.yaml
