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

NOTE: KV cache is NOT used here because block-causal attention requires
recomputing attention for all positions when new tokens are added at
specific block positions (not just appended). Each block's tokens are
decoded independently within the block, but subsequent blocks attend to
all prior tokens. A full forward pass per block iteration is required.

Block-internal decoding (matching original D2F generation.py)
--------------------------------------------------------------
Within each block, positions are NOT decoded in fixed order.  Instead:

    for step in range(ceil(block_size / k)):
        1. Forward pass → logits for ALL masked positions in the block
        2. Compute confidence = sum(p * log p)  (negative entropy)
        3. Select top-k positions by confidence  (k = num_decode_steps)
        4. Sample and decode those k positions
        5. Repeat until block has no more masked positions

This mirrors the ``generate_block()`` inference path where
``number_transfer_tokens = 1`` (k=1) and positions are selected by
``torch.topk(confidence, 1)`` — the most confident position is decoded
first, providing context for subsequent positions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, NamedTuple
from utils.util import build_custom_float_attention_mask, shift_logits


class _DecodeAssignment(NamedTuple):
    """A single position-to-token assignment to avoid in-place ops."""
    batch_idx: int
    position: int
    token_id: torch.Tensor


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


def _select_topk_positions(
    logits: torch.Tensor,
    mask_in_block: torch.Tensor,
    num_decode_steps: int,
) -> torch.Tensor:
    """Select top-k masked positions by confidence (negative entropy).

    Args:
        logits: [sub_L, vocab] logits for the block's subsequence.
        mask_in_block: [sub_L] bool, True at masked positions within the block.
        num_decode_steps: k, number of positions to select.

    Returns:
        decode_absolute_positions: [k] absolute (within-sub-L) positions to decode.
    """
    block_logits = logits[mask_in_block]  # [num_masked, vocab]

    probs = F.softmax(block_logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    confidence = (probs * log_probs).sum(dim=-1)  # [num_masked], higher = more confident

    k = min(num_decode_steps, int(mask_in_block.sum().item()))
    _, topk_relative_idx = torch.topk(confidence, k)

    masked_relative_positions = torch.nonzero(mask_in_block, as_tuple=True)[0]
    return masked_relative_positions[topk_relative_idx]


def _apply_assignments(
    decoded: torch.Tensor,
    assignments: List[_DecodeAssignment],
) -> torch.Tensor:
    """Apply all assignments to a tensor WITHOUT in-place operations.

    Uses index_put_ which creates a new tensor, avoiding autograd issues
    with gradient checkpointing.

    Args:
        decoded: [B, L] tensor to update.
        assignments: List of (batch_idx, position, token_id) tuples.

    Returns:
        New tensor with assignments applied.
    """
    if not assignments:
        return decoded

    batch_indices = torch.tensor([a.batch_idx for a in assignments],
                                device=decoded.device, dtype=torch.long)
    position_indices = torch.tensor([a.position for a in assignments],
                                    device=decoded.device, dtype=torch.long)
    token_values = torch.stack([a.token_id for a in assignments])

    return decoded.index_put_((batch_indices, position_indices), token_values)


def _assemble_rollout_logits(
    logits_list: List[Tuple[int, int, torch.Tensor]],
    B: int,
    L: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Assemble rollout logits efficiently using scatter, avoiding
    the previous loop+mask multiplication approach.

    Args:
        logits_list: List of (batch_idx, position, logit_tensor) tuples.
        B: Batch size.
        L: Sequence length.
        vocab_size: Vocabulary size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        rollout_logits: [B, L, vocab_size] with grad at decoded positions.
    """
    if not logits_list:
        return torch.zeros(B, L, vocab_size, device=device, dtype=dtype)

    # Stack all logit values: [num_decoded, vocab]
    all_logits = torch.stack([lg for _, _, lg in logits_list])

    # Create index tensors for scatter
    batch_idx = torch.tensor([bi for bi, _, _ in logits_list],
                             device=device, dtype=torch.long)
    pos_idx = torch.tensor([pi for _, pi, _ in logits_list],
                           device=device, dtype=torch.long)

    # Create empty result and scatter using index_put_ with accumulated values
    # We need to flatten the [B, L] grid into [B*L] for the scatter operation
    flat_target = torch.zeros(B * L, vocab_size, device=device, dtype=dtype)
    flat_indices = batch_idx * L + pos_idx

    # Use index_put_ for efficient scatter-add
    result = flat_target.index_put_((flat_indices,), all_logits, accumulate=False)

    return result.view(B, L, vocab_size)


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

    Within each block, positions are selected by confidence (negative
    entropy) and decoded k at a time (k = num_decode_steps), matching
    the original D2F inference algorithm.

    Uses non-in-place tensor updates via _apply_assignments to prevent
    autograd issues with gradient checkpointing.
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

        block_ranges = {}
        for i in active:
            start = prompt_lens[i] + block_idx * block_size
            end = min(start + block_size, max_needed)
            block_ranges[i] = (start, end)

        # Inner loop: decode positions by confidence until block is full
        while True:
            any_remaining = False
            for i in active:
                start_i, end_i = block_ranges[i]
                if (student_decoded_sub[i, start_i:end_i] == mask_id).any():
                    any_remaining = True
                    break
            if not any_remaining:
                break

            with torch.no_grad():
                outputs = model(student_decoded_sub, **attn_kw_sub)
                logits = outputs.logits
                if shift:
                    logits = shift_logits(logits)

            # Collect all assignments for this iteration, then apply them
            assignments: List[_DecodeAssignment] = []

            for i in active:
                start_i, end_i = block_ranges[i]
                block_slice = student_decoded_sub[i, start_i:end_i]
                mask_in_block = (block_slice == mask_id)

                if not mask_in_block.any():
                    continue

                with torch.no_grad():
                    decode_relative = _select_topk_positions(
                        logits[i, start_i:end_i], mask_in_block, num_decode_steps,
                    )

                for rel_pos in decode_relative:
                    pos = start_i + rel_pos.item()
                    current_logits = logits[i, pos, :].unsqueeze(0)
                    sampled = _top_p_sample(current_logits, temperature, top_p)
                    token = sampled.squeeze()
                    assignments.append(_DecodeAssignment(
                        batch_idx=i, position=pos, token_id=token
                    ))
                    decoded_positions[i, pos] = True

            # Apply all assignments safely (no in-place)
            student_decoded = _apply_assignments(student_decoded, assignments)
            student_decoded_sub = student_decoded[:, :max_needed]

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
    NOT through the sampling operation.

    Within each block, positions are selected by confidence (negative
    entropy) and decoded k at a time (k = num_decode_steps), matching
    the original D2F inference algorithm.

    Uses non-in-place tensor updates via _apply_assignments to prevent
    autograd issues with gradient checkpointing.

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

    rollout_logits_list: List[Tuple[int, int, torch.Tensor]] = []

    attention_mask_full = build_custom_float_attention_mask(
        student_decoded, question_length, block_size, device=device
    )

    def _fwd(ids, attn_bias):
        kw = {"attention_bias": attn_bias} if is_llada else {"attention_mask": attn_bias}
        return model(ids, **kw)

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

        block_ranges = {}
        for i in active:
            start = prompt_lens[i] + block_idx * block_size
            end = min(start + block_size, max_needed)
            block_ranges[i] = (start, end)

        # Inner loop: decode positions by confidence until block is full
        while True:
            any_remaining = False
            for i in active:
                start_i, end_i = block_ranges[i]
                if (student_decoded_sub[i, start_i:end_i] == mask_id).any():
                    any_remaining = True
                    break
            if not any_remaining:
                break

            # IMPORTANT: Make a contiguous copy of the sub-tensor before
            # passing to gradient checkpointing. This prevents issues when
            # the original tensor is later modified (via index_put_)
            # between the checkpointed forward and its recomputation.
            fwd_input = student_decoded_sub.clone()

            # Forward pass (WITH GRAD + optional gradient checkpointing)
            if use_grad_checkpoint:
                outputs = torch.utils.checkpoint.checkpoint(
                    _fwd, fwd_input, attention_mask_sub,
                    use_reentrant=False,
                )
            else:
                outputs = _fwd(fwd_input, attention_mask_sub)

            logits = outputs.logits  # [B, sub_L, vocab] — WITH grad
            if shift:
                logits = shift_logits(logits)

            # Collect all assignments for this iteration
            assignments: List[_DecodeAssignment] = []

            for i in active:
                start_i, end_i = block_ranges[i]
                block_slice = student_decoded_sub[i, start_i:end_i]
                mask_in_block = (block_slice == mask_id)

                if not mask_in_block.any():
                    continue

                # Select top-k positions by confidence (detached)
                with torch.no_grad():
                    decode_relative = _select_topk_positions(
                        logits[i, start_i:end_i].detach(),
                        mask_in_block,
                        num_decode_steps,
                    )

                for rel_pos in decode_relative:
                    pos = start_i + rel_pos.item()
                    current_logits = logits[i, pos, :]  # [vocab] — WITH grad

                    # Sample DETACHED
                    with torch.no_grad():
                        sampled = _top_p_sample(
                            current_logits.detach().unsqueeze(0),
                            temperature, top_p,
                        )
                        token_id = sampled.squeeze()

                    assignments.append(_DecodeAssignment(
                        batch_idx=i, position=pos, token_id=token_id
                    ))
                    decoded_positions[i, pos] = True
                    rollout_logits_list.append((i, pos, current_logits))

            # Apply all assignments safely (no in-place)
            student_decoded = _apply_assignments(student_decoded, assignments)
            student_decoded_sub = student_decoded[:, :max_needed]

    # --- Assemble rollout_logits [B, L, vocab] efficiently -----------------
    if rollout_logits_list:
        model_dtype = rollout_logits_list[0][2].dtype
        rollout_logits = _assemble_rollout_logits(
            rollout_logits_list, B, L, vocab_size, device, model_dtype,
        )
    else:
        rollout_logits = torch.zeros(
            B, L, vocab_size, device=device
        )

    return student_decoded, decoded_positions, rollout_logits
