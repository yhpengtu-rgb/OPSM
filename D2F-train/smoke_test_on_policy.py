"""Standalone GPU smoke test for the LLaDA on-policy distillation step.

Loads the real LLaDA-8B-Instruct model (fp16) + LoRA, builds a single tiny
batch, and runs one on-policy forward + loss + backward + optimizer step.
This exercises the same code path as train.py without accelerate/deepspeed.
"""
import os
import sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model import get_model_by_config
from utils.loss import compute_loss_by_config


def build_tiny_batch(tokenizer, L=128):
    messages = [{"role": "user", "content": "What is 2+2? Explain briefly."}]
    question = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    ).input_ids[0]
    answer = tokenizer("The answer is 4. Two plus two equals four.", add_special_tokens=False).input_ids
    answer = torch.tensor(answer, dtype=torch.long)
    answer = torch.cat([answer, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)])

    qlen = question.shape[0]
    combined = torch.cat([question, answer])
    if combined.shape[0] > L:
        combined = combined[:L]
    pad_len = L - combined.shape[0]
    if pad_len > 0:
        pad = torch.full((pad_len,), tokenizer.eos_token_id, dtype=combined.dtype)
        combined = torch.cat([combined, pad])
    input_ids = combined.unsqueeze(0)  # [1, L]
    question_length = torch.tensor([min(qlen, L)])
    return input_ids, question_length


def main():
    cfg = OmegaConf.load('config/llada_on_policy_debug.yaml')
    device = torch.device('cuda')

    print('Loading model (fp16) ...', flush=True)
    denoiser, tokenizer = get_model_by_config(cfg)
    denoiser = denoiser.half().to(device)
    denoiser.train()

    n_train = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f'Trainable params: {n_train/1e6:.2f}M', flush=True)

    input_ids, question_length = build_tiny_batch(tokenizer, L=128)
    input_ids = input_ids.to(device)
    question_length = question_length.to(device)
    print(f'batch: input_ids={tuple(input_ids.shape)} qlen={question_length.tolist()}', flush=True)

    params = [p for p in denoiser.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.95), weight_decay=5e-2)

    print('Running on-policy distillation step ...', flush=True)
    losses = compute_loss_by_config(
        input_ids, denoiser, question_length,
        block_size    = cfg.train.block_size,
        mask_id       = cfg.denoiser.encoder.mask_id,
        enable_shift  = cfg.train.enable_shift,
        share_steps   = cfg.train.share_steps,
        self_align    = cfg.train.self_align,
        feature_align = cfg.train.feature_align,
        self_step     = cfg.train.self_step,
        eos_id        = tokenizer.eos_token_id,
        config        = cfg,
    )
    loss = losses['loss']
    print(f"loss={loss.item():.6f} decoded_len={losses['student_decoded_length'].item():.1f} "
          f"rollout_len={losses['teacher_rollout_length'].item():.1f} "
          f"finite={torch.isfinite(loss).item()}", flush=True)

    # NOTE: the real train.sh uses DeepSpeed (fp32 master weights + fp16 compute).
    # Here the model is pure fp16, so we skip GradScaler and do a plain backward
    # just to confirm gradient flow through the on-policy loss.
    print('Backward + optimizer step ...', flush=True)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    gsum = sum(p.grad.abs().sum().item() for p in params if p.grad is not None)
    gnans = sum(int(torch.isnan(p.grad).any().item()) for p in params if p.grad is not None)
    optimizer.step()
    print(f'grad sum={gsum:.6f} grad_nans={gnans} -> backward+step OK', flush=True)
    print('SMOKE TEST PASSED', flush=True)


if __name__ == '__main__':
    main()
