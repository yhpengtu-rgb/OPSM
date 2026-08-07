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

Block-internal decoding (strictly aligned with eval_llada.py)
--------------------------------------------------------------
Within each block, ``num_decode_steps`` controls the number of rollout
forwards. Each forward decodes exactly one masked position:

    for _ in range(num_decode_steps):
        1. Forward pass → logits for ALL masked positions in the block
        2. _sample_tokens → confidence = probability of sampled token
           (NO margin_confidence / NO neg_entropy)
        3. Select the single highest-confidence masked position (top-1)
        4. Sample and decode that position

The remaining masked positions are intentionally retained for the loss.
This mirrors ``_generate_block_single`` in ``D2F-eval/eval_llada.py`` and
``sample_tokens`` in ``model_cache/dream/generation_utils.py``.
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
    confidence: torch.Tensor


def _attn_kwargs(is_llada: bool, block_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    if is_llada:
        return {"attention_bias": block_mask}
    return {"attention_mask": block_mask}


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering, matching eval_llada.py."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def _sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens and return (confidence, token_id), aligned with eval_llada.

    This mirrors ``model_cache/dream/generation_utils.py::sample_tokens``
    but WITHOUT the ``margin_confidence`` / ``neg_entropy`` branches: the
    returned confidence is simply the probability of the sampled token
    (``initial_confidence``). This is the exact on-policy alignment the
    train rollout requires.

    Args:
        logits: [..., vocab] logits.
        temperature: sampling temperature (0 = greedy argmax).
        top_p: nucleus filtering probability (None/>=1 disable).
        top_k: top-k logits filtering (None disables).

    Returns:
        confidence: [..., vocab]->[...,] probability of the sampled token.
        x0: [...,] sampled token ids.
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)

    probs = F.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    return confidence, x0


def _select_top1_position(
    logits: torch.Tensor,
    mask_in_block: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Select all masked positions tied for maximum sampled-token confidence.

    A rollout forward normally decodes one position. Exact confidence ties are
    decoded together so their tokens share the same pre-update block context.
    ``student_decode_steps`` remains the maximum number of rollout forwards
    for a block, not a top-k width.
    """
    block_logits = logits[mask_in_block]  # [num_masked, vocab]
    confidence, _ = _sample_tokens(
        block_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )
    masked_relative_positions = torch.nonzero(mask_in_block, as_tuple=True)[0]
    return masked_relative_positions[confidence == confidence.max()]


def _apply_assignments(
    decoded: torch.Tensor,
    assignments: List[_DecodeAssignment],
) -> torch.Tensor:
    """Apply all assignments to a tensor without in-place operations.

    Uses the out-of-place ``index_put`` operation to avoid autograd issues
    with gradient checkpointing.

    Args:
        decoded: [B, L] tensor to update.
        assignments: List of (batch_idx, position, token_id) tuples.

    Returns:
        New tensor with assignments applied.
    """
    if not assignments:
        return decoded

    batch_indices = torch.tensor(
        [a.batch_idx for a in assignments], device=decoded.device, dtype=torch.long
    )
    position_indices = torch.tensor(
        [a.position for a in assignments], device=decoded.device, dtype=torch.long
    )
    token_values = torch.stack([a.token_id for a in assignments]).to(decoded.device)

    return decoded.index_put((batch_indices, position_indices), token_values)


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

    Each block performs ``num_decode_steps`` rollout forwards. Every forward
    selects one sampled-token-confidence top-1 position; remaining masked
    positions are retained for the DMD remaining-position loss.

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

        # Each decode step runs one forward and fills one top-1 position.
        # Do not finish a block here: its remaining masks are supervised by loss.
        for _ in range(num_decode_steps):
            active_with_masks = [
                i for i in active
                if (student_decoded_sub[i, block_ranges[i][0]:block_ranges[i][1]] == mask_id).any()
            ]
            if not active_with_masks:
                break

            with torch.no_grad():
                outputs = model(student_decoded_sub, **attn_kw_sub)
                logits = outputs.logits
                if shift:
                    logits = shift_logits(logits)

            assignments: List[_DecodeAssignment] = []
            for i in active_with_masks:
                start_i, end_i = block_ranges[i]
                mask_in_block = student_decoded_sub[i, start_i:end_i] == mask_id
                decode_relative = _select_top1_position(
                    logits[i, start_i:end_i], mask_in_block,
                    temperature=temperature, top_p=top_p,
                )
                positions = start_i + decode_relative
                confidence, tokens = _sample_tokens(logits[i, positions, :], temperature, top_p)
                for pos, token, token_confidence in zip(positions.tolist(), tokens, confidence):
                    assignments.append(_DecodeAssignment(
                        batch_idx=i,
                        position=pos,
                        token_id=token,
                        confidence=token_confidence,
                    ))
                    decoded_positions[i, pos] = True

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
    is_llada: bool = False,
    shift: bool = True,
    lengths: Optional[torch.Tensor] = None,
    transition_csm: bool = False,
    transition_sample_ratio: float = 0.0,
    return_rollout_confidence: bool = False,
    decode_mode: str = 'sequential',
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Produce a detached rollout and sampled block transitions for CSM.

    ``transition_sample_ratio`` selects a uniform subset of rollout blocks.
    ``decode_mode='parallel_block'`` fills all remaining positions in each
    block from one shared forward pass; ``sequential`` preserves the legacy
    confidence-ordered rollout. At least one block is retained whenever the
    rollout has any block.
    """
    B, L = input_ids.shape
    if device is None:
        device = input_ids.device

    model = _unwrap(student_model)

    student_decoded = input_ids.clone()
    token_positions = torch.arange(L, device=device).expand(B, L)
    if lengths is None:
        valid_lengths = torch.full((B,), L, device=device, dtype=torch.long)
    else:
        valid_lengths = lengths.to(device=device, dtype=torch.long).clamp(0, L)
    prompt_lengths = question_length.to(device=device, dtype=torch.long).clamp(0, L)
    prompt_lengths = torch.minimum(prompt_lengths, valid_lengths)
    prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
    student_decoded[~prompt_mask] = mask_id
    decoded_positions = torch.zeros_like(input_ids, dtype=torch.bool)
    rollout_confidence = torch.full(
        input_ids.shape, float('inf'), dtype=torch.float32, device=device
    )

    prompt_lens = [int(prompt_lengths[i].item()) for i in range(B)]
    non_prompt_lens = [int(valid_lengths[i].item()) - pl for i, pl in enumerate(prompt_lens)]
    total_blocks = [(npl + block_size - 1) // block_size for npl in non_prompt_lens]
    max_blocks = max(total_blocks) if total_blocks else 0

    transitions: List[Dict[str, torch.Tensor]] = []
    if transition_csm and max_blocks:
        transition_sample_ratio = float(transition_sample_ratio)
        sampled_block_count = max(1, int(max_blocks * transition_sample_ratio))
        sampled_block_count = min(max_blocks, sampled_block_count)
        sampled_blocks = torch.randperm(max_blocks, device=device)[:sampled_block_count]
        sampled_blocks = set(sampled_blocks.cpu().tolist())
    else:
        sampled_blocks = set()

    answer_mask = (
        (token_positions >= prompt_lengths.unsqueeze(1))
        & (token_positions < valid_lengths.unsqueeze(1))
    )

    attention_mask_full = build_custom_float_attention_mask(
        student_decoded, prompt_lengths, block_size, device=device
    )
    valid_mask = token_positions < valid_lengths.unsqueeze(1)
    attention_mask_full = attention_mask_full.masked_fill(
        ~valid_mask[:, None, None, :], float('-inf')
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
            end = min(start + block_size, int(valid_lengths[i].item()))
            block_ranges[i] = (start, end)

        selected_block = transition_csm and block_idx in sampled_blocks
        if selected_block:
            predecessor_ids = student_decoded.detach().clone()

        if decode_mode not in {'sequential', 'parallel_block'}:
            raise ValueError(f"Unsupported rollout decode mode: {decode_mode}")
        decode_iterations = 1 if decode_mode == 'parallel_block' else num_decode_steps

        # Each sequential decode step fills the highest-confidence position;
        # parallel_block fills every remaining position from one forward.
        for _ in range(decode_iterations):
            active_with_masks = [
                i for i in active
                if (student_decoded_sub[i, block_ranges[i][0]:block_ranges[i][1]] == mask_id).any()
            ]
            if not active_with_masks:
                break

            # The rollout only determines the student trajectory. Its graph is
            # intentionally detached; gradients are computed on x_{t+1} below.
            with torch.no_grad():
                logits = _fwd(student_decoded_sub, attention_mask_sub).logits
                if shift:
                    logits = shift_logits(logits)

                assignments: List[_DecodeAssignment] = []
                for i in active_with_masks:
                    start_i, end_i = block_ranges[i]
                    mask_in_block = student_decoded_sub[i, start_i:end_i] == mask_id
                    if decode_mode == 'parallel_block':
                        decode_relative = torch.nonzero(mask_in_block, as_tuple=True)[0]
                    else:
                        decode_relative = _select_top1_position(
                            logits[i, start_i:end_i], mask_in_block,
                            temperature=temperature, top_p=top_p,
                        )
                    positions = start_i + decode_relative
                    confidence, token_ids = _sample_tokens(
                        logits[i, positions, :], temperature, top_p,
                    )
                    for pos, token_id, token_confidence in zip(
                        positions.tolist(), token_ids, confidence
                    ):
                        assignments.append(_DecodeAssignment(
                            batch_idx=i,
                            position=pos,
                            token_id=token_id,
                            confidence=token_confidence,
                        ))
                        decoded_positions[i, pos] = True

                student_decoded = _apply_assignments(student_decoded, assignments)
                if assignments:
                    batch_indices = torch.tensor(
                        [assignment.batch_idx for assignment in assignments],
                        dtype=torch.long,
                        device=device,
                    )
                    position_indices = torch.tensor(
                        [assignment.position for assignment in assignments],
                        dtype=torch.long,
                        device=device,
                    )
                    confidence_values = torch.stack(
                        [assignment.confidence for assignment in assignments]
                    ).float()
                    rollout_confidence = rollout_confidence.index_put(
                        (batch_indices, position_indices), confidence_values
                    )
                student_decoded_sub = student_decoded[:, :max_needed]

        if selected_block:
            advanced_mask = torch.zeros(B, dtype=torch.bool, device=device)
            advanced_mask[active] = True
            transitions.append({
                "predecessor_ids": predecessor_ids,
                "successor_ids": student_decoded.detach().clone(),
                "answer_mask": answer_mask,
                "advanced_mask": advanced_mask,
            })

    if not transition_csm:
        # Preserve the old DMD path: one final successor state after all blocks
        # have executed their configured student_decode_steps updates.
        remaining_mask = (student_decoded == mask_id) & (token_positions < valid_lengths.unsqueeze(1))
        transitions = [{
            "input_ids": student_decoded.clone(),
            "remaining_mask": remaining_mask,
        }]
    if return_rollout_confidence:
        return student_decoded, decoded_positions, transitions, rollout_confidence
    return student_decoded, decoded_positions, transitions
