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
    gumbel_tau: float = 1.0,
    use_grad_checkpoint: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DMD-style differentiable rollout using Gumbel-softmax straight-through.

    Unlike ``student_blockwise_rollout`` (no_grad, hard sampling), this
    function keeps gradients throughout the rollout chain:

    1. Forward passes use ``inputs_embeds`` with **soft embeddings** at
       previously-decoded positions (Gumbel-softmax output @ embed_weight),
       allowing gradient to flow across blocks.
    2. Sampling uses ``F.gumbel_softmax(hard=True)``: hard one-hot in
       forward, soft gradient in backward.
    3. Rollout logits at decoded positions are collected for the DMD loss.

    Returns:
        student_decoded: [B, L] decoded token IDs (hard).
        decoded_positions: [B, L] bool mask of decoded positions.
        rollout_logits: [B, L, vocab_size] logits at decoded positions
            (WITH grad).  Zero at non-decoded positions.
    """
    B, L = input_ids.shape
    if device is None:
        device = input_ids.device

    model = _unwrap(student_model)
    embed_layer = model.get_input_embeddings()  # nn.Embedding (callable)
    embed_weight = embed_layer.weight  # [vocab, dim] — for matmul with Gumbel-softmax

    student_decoded = input_ids.clone()
    token_positions = torch.arange(L, device=device).expand(B, L)
    prompt_mask = token_positions < question_length.unsqueeze(1)
    student_decoded[~prompt_mask] = mask_id
    decoded_positions = torch.zeros_like(input_ids, dtype=torch.bool)

    prompt_lens = [int(question_length[i].item()) for i in range(B)]
    non_prompt_lens = [L - pl for pl in prompt_lens]
    total_blocks = [(npl + block_size - 1) // block_size for npl in non_prompt_lens]
    max_blocks = max(total_blocks) if total_blocks else 0

    # Soft embeddings: [B, L, dim].  Initially zeros (no grad).  Updated
    # differentiably at each decode step via element-wise masking (no
    # in-place ops) so autograd can trace through the full chain.
    # Match model dtype (fp16 under mixed precision) to avoid mat1/mat2
    # dtype mismatch errors.
    dim = embed_weight.shape[1]
    model_dtype = embed_weight.dtype
    soft_embeds = torch.zeros(B, L, dim, device=device, dtype=model_dtype)

    # Rollout logits: collected as a list of (batch_idx, pos, logits) tuples.
    # In-place assignment on a zero tensor doesn't create a gradient link,
    # so we collect logits differentiably and build the final tensor at the end.
    rollout_logits_list = []  # [(batch_idx, pos, logits_tensor_with_grad)]

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

        # --- Build inputs_embeds for this subsequence ---------------------
        # Hard embeddings (detached — no grad on prompt/mask positions)
        hard_embeds_sub = embed_layer(student_decoded[:, :max_needed]).detach()  # [B, sub_L, dim]

        # Overlay soft embeddings at previously-decoded positions
        decode_mask_sub = decoded_positions[:, :max_needed].unsqueeze(-1).to(model_dtype)  # [B, sub_L, 1]
        soft_embeds_sub = soft_embeds[:, :max_needed, :]  # [B, sub_L, dim] — may have grad
        inputs_embeds_sub = hard_embeds_sub * (1 - decode_mask_sub) + soft_embeds_sub * decode_mask_sub

        attention_mask_sub = attention_mask_full[
            :, :, :max_needed, :max_needed
        ].contiguous()
        attn_kw_sub = _attn_kwargs(is_llada, attention_mask_sub)

        # --- Forward pass (WITH GRAD + optional gradient checkpointing) -------
        # Gradient checkpointing recomputes activations during backward,
        # reducing memory from O(num_blocks × activation) to O(1 × activation).
        # Can be disabled via ``use_grad_checkpoint=False`` for debugging
        # or when GPU memory is sufficient.
        def _fwd(embeds, attn_bias):
            kw = {"attention_bias": attn_bias} if is_llada else {"attention_mask": attn_bias}
            return model(input_ids=None, inputs_embeds=embeds, **kw)

        if use_grad_checkpoint:
            outputs = torch.utils.checkpoint.checkpoint(
                _fwd, inputs_embeds_sub, attention_mask_sub,
                use_reentrant=False,
            )
        else:
            outputs = _fwd(inputs_embeds_sub, attention_mask_sub)
        # accelerate's forward wrapper converts model outputs to fp32, but the
        # embedding matrix (embed_weight) is fp16 (base loaded in fp16).  Cast
        # logits back to the model dtype so the whole rollout chain —
        # gumbel-softmax, soft-embed matmul (soft @ embed_weight), and the
        # rollout_logits buffer — uses one consistent dtype and avoids
        # mat1/mat2 dtype-mismatch errors.  The cast is differentiable, so the
        # gradient still flows to the LoRA params.
        logits = outputs.logits.to(model_dtype)  # [B, sub_L, vocab] — WITH grad
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

                # Gumbel-softmax straight-through estimator.
                # ``gumbel_tau`` controls the sharpness of the soft gradient
                # (separate from ``temperature`` which controls sampling).
                soft = F.gumbel_softmax(
                    current_logits.unsqueeze(0), tau=gumbel_tau, hard=True
                )  # [1, vocab] — hard one-hot fwd, soft grad bwd
                token_id = soft.argmax(-1).squeeze()  # hard token ID

                # Soft embedding (WITH grad through Gumbel-softmax)
                soft_embed = (soft @ embed_weight).squeeze(0)  # [dim]

                # --- Update student_decoded (hard token ID, no grad) -------
                student_decoded[i, pos] = token_id
                decoded_positions[i, pos] = True

                # --- Store rollout logits (WITH grad) ----------------------
                # Collect into a list.  In-place assignment
                # (``preallocated[i, pos] = logits``) on a zero tensor does NOT
                # create an autograd link to ``logits``, so the DMD loss would
                # receive a zero gradient.  The full [B, L, vocab] tensor is
                # assembled differentiably after the loop (see below).
                rollout_logits_list.append((i, pos, current_logits))

                # --- Update soft_embeds differentiably ---------------------
                # Build [B, L, 1] update mask (1 at [i, pos], 0 elsewhere).
                # The in-place setitem on ``update_mask`` is safe: it is a
                # freshly-allocated tensor with no grad history.  The soft-embed
                # broadcast below is fully differentiable wrt ``soft_embed``
                # (no in-place setitem on a grad-bearing tensor — unlike the
                # previous ``soft_expanded[i, pos, :] = soft_embed`` which could
                # sever the grad link).
                update_mask = torch.zeros(B, L, 1, device=device, dtype=model_dtype)
                update_mask[i, pos, 0] = 1.0
                # Broadcast soft_embed [dim] -> [B, L, dim] via the mask, then
                # element-wise blend into soft_embeds (rebind, not in-place).
                soft_embeds = (
                    soft_embeds * (1 - update_mask)
                    + update_mask * soft_embed.view(1, 1, -1)
                )

    # --- Assemble rollout_logits [B, L, vocab] differentiably ------------
    # In-place assignment (``tensor[i, pos] = logits``) on a pre-allocated
    # zero tensor does NOT create an autograd link to ``logits``, so the DMD
    # loss would see a zero gradient.  Instead, accumulate each decoded
    # position's logits via a one-hot mask:
    #     rollout_logits = sum_k  onehot(b_k, p_k) * logits_k
    # Each term is differentiable wrt ``logits_k``; non-decoded positions
    # remain exactly zero.  With student_decode_steps=1 the number of terms
    # is small (~num_blocks), so the [B, L, vocab] intermediates are cheap.
    if rollout_logits_list:
        vocab = rollout_logits_list[0][2].shape[0]
        rollout_logits = torch.zeros(B, L, vocab, device=device, dtype=model_dtype)
        for bi, pos_i, lg in rollout_logits_list:
            mask = torch.zeros(B, L, 1, device=device, dtype=model_dtype)
            mask[bi, pos_i, 0] = 1.0
            rollout_logits = rollout_logits + mask * lg.view(1, 1, -1)
    else:
        rollout_logits = torch.zeros(
            B, L, vocab_size, device=device, dtype=model_dtype
        )

    return student_decoded, decoded_positions, rollout_logits
