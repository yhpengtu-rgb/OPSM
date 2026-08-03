"""
On-policy rollout functions for D2F training.
This module implements the on-policy distillation strategy where:
1. Student model performs n-step decoding within each block using block-wise
   causal attention.
2. (Teacher rollout is currently disabled — the distillation target comes from
   a single teacher forward pass in ``compute_on_policy_loss``.)

Optimisation notes
-------------------
The original implementation looped over every block × step and ran a full
forward pass for each combination, resulting in ~52 × 2 = 104 forward passes
per batch.  The current version batches **all blocks at the same decode step**
into a single forward pass, reducing the count to ``num_decode_steps`` (e.g. 2)
forward passes — a ~50× reduction.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from utils.util import build_custom_float_attention_mask, shift_logits


def _attn_kwargs(is_llada: bool, block_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Build the attention kwarg dict according to the model family.

    LLaDA's forward expects the 4D float block mask as ``attention_bias`` (it is
    added directly to the SDPA scores), while the Dream model expects it as
    ``attention_mask``. Detecting the family via ``type(model).__name__`` is
    unreliable once the model is wrapped by peft / accelerate / deepspeed, so the
    caller passes ``is_llada`` explicitly.
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

    # Sort descending to find the nucleus.
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Shift right so the first token above the threshold is kept.
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    # Scatter back to original ordering.
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

    All blocks at the same decode step are processed in a **single forward
    pass** (the attention mask is block-causal and independent of token values,
    so it is built once and reused).  This reduces the number of forward passes
    from ``num_blocks × num_decode_steps`` to just ``num_decode_steps``.

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
    # token values, so it is valid for every decode step.
    attention_mask = build_custom_float_attention_mask(
        student_decoded, question_length, block_size, device=device
    )

    # --- Pre-compute decode positions for every (block, step) pair --------
    # For sample *i*, block *b*, step *s*:
    #   pos = question_length[i] + b * block_size + s
    # We collect all valid positions per step into a flat index tensor so
    # that the inner loop only does cheap tensor indexing (no .item() calls).
    max_blocks = int(((L - question_length.min().item()) + block_size - 1) // block_size)

    # valid_positions[step] = list of (batch_idx, pos) to decode at this step
    valid_positions: list[list[tuple[int, int]]] = [[] for _ in range(num_decode_steps)]
    for b in range(max_blocks):
        for i in range(B):
            ql = int(question_length[i].item())
            for s in range(num_decode_steps):
                pos = ql + b * block_size + s
                if pos < L:
                    valid_positions[s].append((i, pos))

    # --- Decode: one forward pass per step --------------------------------
    for step in range(num_decode_steps):
        positions = valid_positions[step]
        if not positions:
            continue

        batch_indices = torch.tensor([p[0] for p in positions], device=device)
        pos_indices = torch.tensor([p[1] for p in positions], device=device)

        # Single forward pass for the entire sequence at this step.
        with torch.no_grad():
            outputs = _unwrap(student_model)(
                student_decoded, **_attn_kwargs(is_llada, attention_mask)
            )
            logits = outputs.logits  # [B, L, V]

        if shift:
            logits = shift_logits(logits)

        # Gather logits at all decode positions for this step: [num_pos, V]
        current_logits = logits[batch_indices, pos_indices]

        # Vectorised top-p sampling.
        sampled = _top_p_sample(current_logits, temperature, top_p)  # [num_pos, 1]

        # Scatter sampled tokens back.
        student_decoded[batch_indices, pos_indices] = sampled.squeeze(-1)
        decoded_positions[batch_indices, pos_indices] = True

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
