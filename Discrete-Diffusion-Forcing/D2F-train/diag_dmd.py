"""Diagnostic: run one DMD step and print all intermediate values to find
why loss == 0.

Replicates the minimal train.py setup (accelerate + fp16 + LoRA + EMA) for a
single batch, then calls compute_dmd_loss with verbose diagnostics.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from utils.util import flatten_dict, shift_logits, build_custom_float_attention_mask
from utils.data import get_dataloader_by_config
from utils.model import get_model_by_config
from utils.loss import compute_dmd_loss
from utils.ema_lora import EMALoRA


def main():
    config = OmegaConf.load("config/llada_on_policy_dmd.yaml")
    accel = Accelerator(mixed_precision="fp16")
    denoiser, tokenizer = get_model_by_config(config)
    # Pass max_length from config so the diag batch matches the training
    # setup (seq_len=128).  Without this, get_dataloader_by_config defaults
    # to max_length=1024, making the differentiable rollout do ~64 block-wise
    # grad forward passes (vs ~8 at 128) — far slower and likely OOM.
    data_max_length = config.data.get('max_length', 1024)
    dataloader = get_dataloader_by_config(tokenizer, config.data, config, max_length=data_max_length)
    denoiser, dataloader = accel.prepare(denoiser, dataloader)

    ema_lora = EMALoRA(denoiser, decay=0.999)
    print(f"[diag] model dtype: base embed = {denoiser.get_base_model().get_input_embeddings().weight.dtype}")
    lora_p = next(p for p in denoiser.parameters() if p.requires_grad)
    print(f"[diag] LoRA param dtype: {lora_p.dtype}")

    # Grab one batch
    batch = next(iter(dataloader))
    input_ids = batch["data"]
    question_length = batch["question_length"]
    lengths = batch.get("length", None)
    print(f"[diag] input_ids shape: {input_ids.shape}")
    print(f"[diag] question_length: {question_length.tolist()}")
    print(f"[diag] lengths: {None if lengths is None else lengths.tolist()}")
    print(f"[diag] L = {input_ids.shape[1]}, block_size = {config.train.block_size}")

    # --- Manually run the DMD rollout + forwards with diagnostics ----------
    from utils.on_policy_rollout import student_blockwise_rollout_dmd
    training_mode = config.get("training_mode", "dream")
    is_llada = (training_mode == "llada")
    shift = (not is_llada) and config.train.enable_shift
    base = denoiser.get_base_model() if hasattr(denoiser, "get_base_model") else denoiser
    try:
        vsize = base.config.vocab_size
    except Exception:
        vsize = 128000

    student_decoded, decoded_positions, rollout_logits = student_blockwise_rollout_dmd(
        input_ids=input_ids,
        student_model=denoiser,
        question_length=question_length,
        block_size=config.train.block_size,
        num_decode_steps=config.train.student_decode_steps,
        mask_id=126336,
        eos_id=tokenizer.eos_token_id,
        temperature=config.train.temperature,
        top_p=config.train.top_p,
        device=input_ids.device,
        vocab_size=vsize,
        is_llada=is_llada,
        shift=shift,
        use_grad_checkpoint=config.train.get('dmd_grad_checkpoint', True),
    )
    print("\n=== ROLLOUT DIAGNOSTICS ===")
    print(f"decoded_positions.sum(): {decoded_positions.sum().item()}")
    print(f"rollout_logits.requires_grad: {rollout_logits.requires_grad}")
    print(f"rollout_logits.grad_fn: {rollout_logits.grad_fn}")
    print(f"rollout_logits dtype: {rollout_logits.dtype}")
    if decoded_positions.any():
        rl = rollout_logits[decoded_positions]
        print(f"rollout_logits[decoded] shape: {rl.shape}, absmax: {rl.abs().max().item():.4f}, mean: {rl.mean().item():.4f}")
        print(f"rollout_logits[decoded] all zero? {(rl == 0).all().item()}")
    else:
        print("!! decoded_positions EMPTY after rollout")

    # valid_mask filtering (same as loss fn)
    L = input_ids.shape[1]
    device = input_ids.device
    if lengths is not None:
        tp = torch.arange(L, device=device).unsqueeze(0)
        valid_mask = tp < lengths.to(device).unsqueeze(1)
        decoded_filt = decoded_positions & valid_mask
        print(f"\nvalid_mask sum: {valid_mask.sum().item()}")
        print(f"decoded_positions AFTER valid_mask filter: {decoded_filt.sum().item()}")
    else:
        decoded_filt = decoded_positions

    if decoded_filt.any():
        sampled = student_decoded[decoded_filt]
        print(f"sampled tokens (first 10): {sampled[:10].tolist()}")
        log_p_student = F.log_softmax(rollout_logits[decoded_filt], dim=-1)
        log_p_student = log_p_student.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        print(f"log_p_student: shape={log_p_student.shape}, mean={log_p_student.mean().item():.4f}, requires_grad={log_p_student.requires_grad}")

    # --- Teacher + fake forwards ---
    # Teacher = base model (disable_adapter) + full bidirectional mask,
    # matching the original D2F ref_logits path.
    # Fake   = EMA LoRA + student block-causal mask.
    attn_student = build_custom_float_attention_mask(
        student_decoded, question_length, config.train.block_size, device=device).to(torch.float16)
    attn_teacher = torch.zeros([1, 1, L, L], dtype=torch.float16, device=device)
    student_kw = {"attention_bias": attn_student} if is_llada else {"attention_mask": attn_student}
    teacher_kw = {"attention_bias": attn_teacher} if is_llada else {"attention_mask": attn_teacher}
    print(f"\n[diag] student_block_size={config.train.block_size}, teacher=full_bidirectional (disable_adapter)")
    with torch.no_grad():
        with denoiser.disable_adapter():
            logits_teacher = denoiser(student_decoded, **teacher_kw).logits  # base model, full bidirectional
            if shift:
                logits_teacher = shift_logits(logits_teacher)
    with torch.no_grad():
        with ema_lora.swap(denoiser):
            logits_fake = denoiser(student_decoded, **student_kw).logits  # EMA LoRA, student block
            if shift:
                logits_fake = shift_logits(logits_fake)

    print("\n=== DENSITY RATIO c ===")
    if decoded_filt.any():
        sampled = student_decoded[decoded_filt]
        with torch.no_grad():
            lpt = F.log_softmax(logits_teacher[decoded_filt], dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            lpf = F.log_softmax(logits_fake[decoded_filt], dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            c = lpt - lpf
            print(f"log_p_teacher: mean={lpt.mean().item():.4f}")
            print(f"log_p_fake:    mean={lpf.mean().item():.4f}")
            print(f"c = teacher - fake: mean={c.mean().item():.6f}, absmax={c.abs().max().item():.6f}, all_zero?={(c==0).all().item()}")
            print(f"teacher == fake? {torch.allclose(lpt, lpf, atol=1e-4)}")

    # --- Full loss via compute_dmd_loss ---
    print("\n=== FULL compute_dmd_loss ===")
    losses = compute_dmd_loss(
        input_ids, denoiser, ema_lora, question_length, 126336, config.train.block_size,
        config.train.enable_shift, config.train.share_steps, config.train.self_align,
        config.train.feature_align, config.train.self_step, tokenizer.eos_token_id,
        student_decode_steps=config.train.student_decode_steps,
        teacher_rollout_steps=config.train.teacher_rollout_steps,
        temperature=config.train.temperature, top_p=config.train.top_p,
        config=config, lengths=lengths,
    )
    loss = losses["loss"]
    print(f"loss: {loss.item():.8f}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")
    print(f"student_decoded_length: {losses['student_decoded_length'].item():.1f}")
    print(f"remaining_mask_length: {losses.get('remaining_mask_length', torch.tensor(float('nan'))).item():.1f}")

    # Backward to check grads flow
    if loss.requires_grad:
        loss.backward()
        grads = [p.grad for p in denoiser.parameters() if p.requires_grad and p.grad is not None]
        if grads:
            gnorm = sum(g.norm().item() ** 2 for g in grads) ** 0.5
            print(f"\n=== BACKWARD ===")
            print(f"LoRA grad norm: {gnorm:.6e} (num params with grad: {len(grads)})")
            print("GRAD FLOWS" if gnorm > 0 else "!! GRAD IS ZERO")
        else:
            print("\n!! No LoRA params received grad")
    else:
        print("\n!! loss does not require grad — no backward possible")


if __name__ == "__main__":
    main()
