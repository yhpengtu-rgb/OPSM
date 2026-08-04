"""
On-policy rollout functions for D2F training.

This module implements the on-policy distillation strategy where the student
model performs n-step decoding within each block using block-wise **causal**
attention (inter-block causal: later blocks attend to earlier blocks'
decoded tokens).

Teacher rollout is currently disabled — the distillation target comes from a
single teacher forward pass in ``compute_on_policy_loss``.

Optimisation notes
-------------------
Blocks **must** be processed sequentially to preserve inter-block causality
(block *b*+1 sees block *b*'s decoded tokens).  The following optimisations
are applied on top of the sequential loop:

1. **Attention mask built once** — the block-causal mask depends only on
   (seq_len, prompt_length, block_size), not on token values.
2. **No .item() in the hot loop** — all scalar values (prompt lengths, block
   boundaries) are pre-computed as Python ints before the loop.
3. **No unused tensor allocations** — ``decode_log_probs`` (623 MB) removed.
4. **Vectorised top-p sampling** — ``_top_p_sample`` handles arbitrary
   leading dims in one call.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from utils.util import build_custom_float_attention_mask, shift_logits


def _attn_kwargs(is_llada: bool, block_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Build the attention kwarg dict according to the model family.

    LLaDA's forward expects the 4D float block mask as ``attention_bias`` (it is
    added directly to the SDPA scores), while the Dream model expects it as
    ``attention_mask``.
    """
    if is_llada:
        return {"attention_bias": block_mask}
    return {"attention_mask": block_mask}


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, peeling off DeepSpeed / DDP wrappers."""
    return model.module if hasattr(model, "module") else model


def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Vectorised top-p (nucleus) sampling.

    Args:
        logits: [..., vocab_size] raw logits.
        temperature: sampling temperature.
        top_p: nucleus probability threshold.

    Returns:
        sampled tokens [..., 1].
    """
    probs = F.softmax(logits / temperature, dim=-1)
    if top_p >= 1.0:
        return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])


def student_blockwise_rollout(
    input_ids: torch.Tensor,
    student_model: torch.nn.Module,
    question_length: torch.Tensor,
    block_size: int,
    num_decode_steps: int,
    mask_id: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    vocab_size: int = 128000,
    is_llada: bool = False,
    shift: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Student model performs on-policy decoding within each block.

    Blocks are processed **sequentially** (block 0 → block 1 → …) so that
    later blocks see earlier blocks' decoded tokens, preserving the
    inter-block causal attention of D2F.  Within each block, decode steps
    are also sequential (step 1 sees step 0's decoded token).

    The block-causal attention mask is built **once** and reused for all
    forward passes (it depends only on sequence layout, not token values).

    Args:
        input_ids: Original input tokens [B, L].
        student_model: Student model (LoRA-wrapped).
        question_length: Prompt length per sample [B].
        block_size: Block size.
        num_decode_steps: Steps to decode per block (n < block_size).
        mask_id: Mask token ID.
        eos_id: EOS token ID.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
        device: Device.
        vocab_size: (unused, kept for API compat).
        is_llada: Whether the model is LLaDA (controls attention kwarg).
        shift: Whether to shift logits (Dream only).

    Returns:
        student_decoded: Decoded sequence [B, L].
        decoded_positions: Boolean mask of decoded positions [B, L].
    """
    B, L = input_ids.shape
    if device is None:
        device = input_ids.device

    # --- Initialise: keep prompt, mask everything else --------------------
    student_decoded = input_ids.clone()
    token_positions = torch.arange(L, device=device).expand(B, L)
    prompt_mask = token_positions < question_length.unsqueeze(1)
    student_decoded[~prompt_mask] = mask_id

    decoded_positions = torch.zeros_like(input_ids, dtype=torch.bool)

    # --- Build the block-causal attention mask once -----------------------
    # The mask only depends on (seq_len, prompt_length, block_size), not on
    # token values, so it is valid for every forward pass in the loop below.
    attention_mask = build_custom_float_attention_mask(
        student_decoded, question_length, block_size, device=device
    )
    attn_kw = _attn_kwargs(is_llada, attention_mask)

    # --- Pre-compute all scalar values to avoid .item() in the hot loop ---
    prompt_lens = [int(question_length[i].item()) for i in range(B)]
    non_prompt_lens = [L - pl for pl in prompt_lens]
    total_blocks = [(npl + block_size - 1) // block_size for npl in non_prompt_lens]
    max_blocks = max(total_blocks) if total_blocks else 0

    model = _unwrap(student_model)

    # --- Sequential block loop (inter-block causal) -----------------------
    for block_idx in range(max_blocks):
        for i in range(B):
            if block_idx >= total_blocks[i]:
                continue

            start = prompt_lens[i] + block_idx * block_size
            end = min(start + block_size, L)
            steps = min(num_decode_steps, end - start)

            for step in range(steps):
                pos = start + step

                # Forward pass — sees all previously decoded tokens in
                # earlier blocks (inter-block causal) and in this block.
                with torch.no_grad():
                    outputs = model(student_decoded, **attn_kw)
                    logits = outputs.logits

                if shift:
                    logits = shift_logits(logits)

                current_logits = logits[i, pos, :].unsqueeze(0)  # [1, V]
                sampled = _top_p_sample(current_logits, temperature, top_p)  # [1, 1]

                student_decoded[i, pos] = sampled.squeeze()
                decoded_positions[i, pos] = True

    return student_decoded, decoded_positions


def on_policy_distillation_step(
    input_ids: torch.Tensor,
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    question_length: torch.Tensor,
    block_size: int,
    student_decode_steps: int,
    teacher_rollout_steps: int,
    mask_id: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    vocab_size: int = 128000,
    is_llada: bool = False,
    shift: bool = True,
) -> Dict[str, torch.Tensor]:
    """Perform one step of on-policy distillation (student-only rollout).

    The teacher rollout is currently disabled.  The distillation target is
    obtained from a single teacher forward pass in ``compute_on_policy_loss``
    (using ``disable_adapter`` for the frozen base weights).

    Args:
        input_ids: Original input tokens [B, L].
        student_model: Student model.
        teacher_model: (unused — kept for API compat).
        question_length: Prompt length [B].
        block_size: Block size.
        student_decode_steps: Number of student decode steps per block.
        teacher_rollout_steps: (unused — kept for API compat).
        mask_id: Mask token ID.
        eos_id: EOS token ID.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        device: Device.
        vocab_size: (unused).
        is_llada: Whether the model is LLaDA.
        shift: Whether to shift logits (Dream only).

    Returns:
        Dictionary with ``student_decoded`` and ``decoded_positions``.
    """
    student_decoded, decoded_positions = student_blockwise_rollout(
        input_ids=input_ids,
        student_model=student_model,
        question_length=question_length,
        block_size=block_size,
        num_decode_steps=student_decode_steps,
        mask_id=mask_id,
        eos_id=eos_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        vocab_size=vocab_size,
        is_llada=is_llada,
        shift=shift,
    )

    return {
        'student_decoded': student_decoded,
        'decoded_positions': decoded_positions,
    }
