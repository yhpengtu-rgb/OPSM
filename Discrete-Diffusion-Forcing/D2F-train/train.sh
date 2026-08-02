# export CUDA_LAUNCH_BLOCKING=1
export HF_HUB_OFFLINE=1
export PATH=/home/cw/workspace/miniconda3/envs/d2f/bin:$PATH
CUDA_VISIBLE_DEVICES=4 accelerate launch   --config_file config/acc_config --num_processes 1 --main_process_port 29577 train.py --config config/llada_on_policy.yaml
