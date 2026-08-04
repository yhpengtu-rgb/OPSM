"""
On-policy rollout functions for D2F training.

Optimisation: partial-sequence forward (KV-cache equivalent)
-------------------------------------------------------------
Instead of passing the full [B, L] sequence to the model for every forward
pass, only pass the relevant subsequence [B, prompt_len + (block_idx+1)
* block_size]. Since attention is O(seq_len^2), early blocks (which only
need prompt + a few blocks) are dramatically cheaper.

The full block-causal attention mask is built ONCE and sliced (top-left
submatrix) for each block's subsequence, avoiding the Python loop in
build_custom_float_attention_mask per block.

Positions are preserved: the subsequence is just the first sub_L tokens
of the full sequence, so RoPE positional encodings are identical.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from utils.util import build_custom_float_attention_mask, shift_logits


def _attn_kwargs(is_llada: bool, block_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    if is_llada:
        return {"attention_bias": block_mask}
    return {"attention_mask": block_mask}


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
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

    Blocks are processed sequentially (inter-block causal). For block b,
    only the first prompt_len + (b+1)*block_size tokens are passed to the
    model, reducing attention cost from O(L^2) to O((prompt+b*bs)^2).

    The full attention mask is built once and sliced per block.
    """
    B, L = input_ids.shape
    if device is None:
        device = input_ids.device

    student_decoded = input_ids.clone()
    token_positions = torch.arange(L, device=device).expand(B, L)
    prompt_mask = token_positions < question_length.unsqueeze(1)
    student_decoded[~prompt_mask] = mask_id
    decoded_positions = torch.zeros_like(input_ids, dtype=torch.bool)

    prompt_lens = [int(question_length[i].item()) for i in range(B)]
    non_prompt_lens = [L - pl for pl in prompt_lens]
    total_blocks = [(npl + block_size - 1) // block_size for npl in non_prompt_lens]
    max_blocks = max(total_blocks) if total_blocks else 0

    model = _unwrap(student_model)

    # Build the FULL block-causal attention mask ONCE.
    # For each block, slice the top-left [sub_L, sub_L] submatrix.
    attention_mask_full = build_custom_float_attention_mask(
        student_decoded, question_length, block_size, device=device
    )

    for block_idx in range(max_blocks):
        active = [i for i in range(B) if block_idx < total_blocks[i]]
        if not active:
            continue

        max_needed = max(
            prompt_lens[i] + (block_idx + 1) * block_size for i in active
        )
        max_needed = min(max_needed, L)

        student_decoded_sub = student_decoded[:, :max_needed]
        attention_mask_sub = attention_mask_full[
            :, :, :max_needed, :max_needed
        ].contiguous()
        attn_kw_sub = _attn_kwargs(is_llada, attention_mask_sub)

        max_steps_block = 0
        block_starts = {}
        for i in active:
            start = prompt_lens[i] + block_idx * block_size
            end = min(start + block_size, max_needed)
            steps_i = min(num_decode_steps, end - start)
            block_starts[i] = (start, steps_i)
            max_steps_block = max(max_steps_block, steps_i)

        for step in range(max_steps_block):
            with torch.no_grad():
                outputs = model(student_decoded_sub, **attn_kw_sub)
                logits = outputs.logits

            if shift:
                logits = shift_logits(logits)

            for i in active:
                start_i, steps_i = block_starts[i]
                if step >= steps_i:
                    continue
                pos = start_i + step
                current_logits = logits[i, pos, :].unsqueeze(0)
                sampled = _top_p_sample(current_logits, temperature, top_p)
                token = sampled.squeeze()
                student_decoded[i, pos] = token
                student_decoded_sub[i, pos] = token
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
    """Perform one step of on-policy distillation (student-only rollout)."""
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


def student_blockwise_rollout_dmd(
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
    use_grad_checkpoint: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DMD rollout: forward WITH grad, sampling DETACHED.

    The DMD loss is ``-sg(c) · log_softmax(y1)[y1']`` where y1 is the
    rollout logits (with grad) and y1' is the sampled token ID (just an
    index, stop-gradient).  The gradient flows through y1 directly —
    NOT through the sampling operation.  So no Gumbel-softmax or soft
    embeddings are needed; regular detached sampling suffices.

    This function differs from ``student_blockwise_rollout`` (which uses
    ``no_grad``) only in that the forward passes keep gradients so the
    logits y1 can be used in the DMD loss.

    Args:
        use_grad_checkpoint: gradient checkpointing for the forward passes
            to fit multiple block-wise forwards in GPU memory.

    Returns:
        student_decoded: [B, L] decoded token IDs (hard, detached).
        decoded_positions: [B, L] bool mask of decoded positions.
        rollout_logits: [B, L, vocab_size] logits at decoded positions
            (WITH grad).  Zero at non-decoded positions.
    """
    B, L = input_ids.shape
    if device is None:
        device = input_ids.device

    model = _unwrap(student_model)

    student_decoded = input_ids.clone()
    token_positions = torch.arange(L, device=device).expand(B, L)
    prompt_mask = token_positions < question_length.unsqueeze(1)
    student_decoded[~prompt_mask] = mask_id
    decoded_positions = torch.zeros_like(input_ids, dtype=torch.bool)

    prompt_lens = [int(question_length[i].item()) for i in range(B)]
    non_prompt_lens = [L - pl for pl in prompt_lens]
    total_blocks = [(npl + block_size - 1) // block_size for npl in non_prompt_lens]
    max_blocks = max(total_blocks) if total_blocks else 0

    # Collect rollout logits (WITH grad) as a list of (batch_idx, pos, logits).
    # In-place assignment on a zero tensor doesn't create a gradient link,
    # so we collect logits and build the final tensor differentiably.
    rollout_logits_list = []

    # Build the FULL block-causal attention mask ONCE (same as KV-cache version).
    attention_mask_full = build_custom_float_attention_mask(
        student_decoded, question_length, block_size, device=device
    )

    for block_idx in range(max_blocks):
        active = [i for i in range(B) if block_idx < total_blocks[i]]
        if not active:
            continue

        max_needed = max(
            prompt_lens[i] + (block_idx + 1) * block_size for i in active
        )
        max_needed = min(max_needed, L)

        student_decoded_sub = student_decoded[:, :max_needed]
        attention_mask_sub = attention_mask_full[
            :, :, :max_needed, :max_needed
        ].contiguous()

        # --- Forward pass (WITH GRAD + optional gradient checkpointing) ---
        # The logits y1 must carry gradient for the DMD loss.  Sampling is
        # detached — no Gumbel-softmax needed because the gradient flows
        # through log_softmax(y1)[y1'], not through the sample y1'.
        def _fwd(ids, attn_bias):
            kw = {"attention_bias": attn_bias} if is_llada else {"attention_mask": attn_bias}
            return model(ids, **kw)

        if use_grad_checkpoint:
            outputs = torch.utils.checkpoint.checkpoint(
                _fwd, student_decoded_sub, attention_mask_sub,
                use_reentrant=False,
            )
        else:
            outputs = _fwd(student_decoded_sub, attention_mask_sub)

        logits = outputs.logits  # [B, sub_L, vocab] — WITH grad
        if shift:
            logits = shift_logits(logits)

        max_steps_block = 0
        block_starts = {}
        for i in active:
            start = prompt_lens[i] + block_idx * block_size
            end = min(start + block_size, max_needed)
            steps_i = min(num_decode_steps, end - start)
            block_starts[i] = (start, steps_i)
            max_steps_block = max(max_steps_block, steps_i)

        for step in range(max_steps_block):
            for i in active:
                start_i, steps_i = block_starts[i]
                if step >= steps_i:
                    continue
                pos = start_i + step
                current_logits = logits[i, pos, :]  # [vocab] — WITH grad

                # Sample DETACHED — y1' is just an index for the DMD loss,
                # no gradient needs to flow through the sampling.
                with torch.no_grad():
                    sampled = _top_p_sample(
                        current_logits.unsqueeze(0).detach(),
                        temperature, top_p,
                    )
                    token_id = sampled.squeeze()

                student_decoded[i, pos] = token_id
                decoded_positions[i, pos] = True
                rollout_logits_list.append((i, pos, current_logits))

    # --- Assemble rollout_logits [B, L, vocab] differentiably ------------
    # Accumulate each decoded position's logits via a one-hot mask so the
    # gradient link to ``current_logits`` is preserved.
    if rollout_logits_list:
        vocab = rollout_logits_list[0][2].shape[0]
        model_dtype = rollout_logits_list[0][2].dtype
        rollout_logits = torch.zeros(B, L, vocab, device=device, dtype=model_dtype)
        for bi, pos_i, lg in rollout_logits_list:
            mask = torch.zeros(B, L, 1, device=device, dtype=model_dtype)
            mask[bi, pos_i, 0] = 1.0
            rollout_logits = rollout_logits + mask * lg.view(1, 1, -1)
    else:
        rollout_logits = torch.zeros(
            B, L, vocab_size, device=device
        )

    return student_decoded, decoded_positions, rollout_logits
