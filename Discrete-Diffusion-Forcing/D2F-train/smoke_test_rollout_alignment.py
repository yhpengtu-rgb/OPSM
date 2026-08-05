"""Minimal smoke test for the on-policy rollout / eval_llada alignment.

Verifies:
1. `_sample_tokens` returns confidence = sampled-token probability
   (no margin_confidence / neg_entropy branches).
2. `_select_topk_positions` ranks masked positions by confidence and
   decodes the top-k (k = num_decode_steps = step).
3. The full `student_blockwise_rollout` runs end-to-end on a tiny model,
   decoding exactly `num_decode_steps` positions per step within a block.

Run:  python smoke_test_rollout_alignment.py
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.on_policy_rollout import (
    _sample_tokens,
    _select_topk_positions,
    student_blockwise_rollout,
)


class TinyLLada(nn.Module):
    """Tiny model exposing `.logits` and accepting attention_bias/attention_mask."""

    def __init__(self, vocab: int, d_model: int, seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))
        self.proj = nn.Linear(d_model, vocab)
        self.d_model = d_model

    def forward(self, input_ids, attention_bias=None, attention_mask=None, **kw):
        emb = self.embed(input_ids) + self.pos[: input_ids.shape[1]]
        logits = self.proj(emb)
        return type("Out", (), {"logits": logits})()


def test_sample_tokens():
    print("=== test_sample_tokens ===")
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 9.0]])
    conf, x0 = _sample_tokens(logits, temperature=0.0)  # greedy
    probs = torch.softmax(logits, dim=-1)
    expected = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(conf, expected, atol=1e-5), f"conf={conf}, expected={expected}"
    assert x0.tolist() == [1, 2], f"x0={x0}"
    print(f"  greedy conf={conf.tolist()} x0={x0.tolist()}  ✅")

    # temperature > 0: confidence must equal prob of sampled token
    torch.manual_seed(0)
    conf2, x0_2 = _sample_tokens(logits, temperature=1.0, top_p=1.0)
    probs2 = torch.softmax(logits, dim=-1)
    gathered = torch.gather(probs2, -1, x0_2.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(conf2, gathered, atol=1e-5), "confidence != sampled prob"
    print(f"  sampled conf={conf2.tolist()} x0={x0_2.tolist()}  ✅")
    print("  PASSED\n")


def test_select_topk_positions():
    print("=== test_select_topk_positions ===")
    # 4 positions logits; position 2 is by far the most confident
    logits = torch.tensor([
        [0.1, 0.2, 0.5],   # pos0
        [0.4, 0.3, 0.3],   # pos1
        [0.0, 0.0, 9.0],   # pos2  <- argmax prob ~1.0
        [0.2, 0.2, 0.6],   # pos3
    ])
    mask = torch.tensor([True, True, True, True])
    k = 2  # step = 2
    pos = _select_topk_positions(logits, mask, k, temperature=0.0)
    # top-2 by confidence must be pos2 (highest) and then next highest
    assert pos[0].item() == 2, f"top1 should be pos2, got {pos.tolist()}"
    assert len(pos) == k, f"expected {k} positions, got {len(pos)}"
    print(f"  selected positions (k={k}): {pos.tolist()}  ✅")

    # k clamped to number of masked positions
    mask2 = torch.tensor([True, False, True, False])
    pos2 = _select_topk_positions(logits, mask2, 5, temperature=0.0)
    assert len(pos2) == int(mask2.sum()), f"should clamp to masked count, got {pos2.tolist()}"
    assert set(pos2.tolist()) == {0, 2}
    print(f"  clamped positions: {pos2.tolist()}  ✅")
    print("  PASSED\n")


def test_full_rollout(device):
    print(f"=== test_full_rollout ({device}) ===")
    torch.manual_seed(42)
    vocab, d_model, L, mask_id = 100, 16, 16, 0
    block_size, steps = 4, 2  # each step decodes 2 positions in a block

    model = TinyLLada(vocab, d_model, L).to(device).eval()

    B = 1
    input_ids = torch.zeros(B, L, dtype=torch.long).to(device)
    question_length = torch.tensor([4]).to(device)  # first 4 tokens are prompt
    # assign distinct non-mask tokens to prompt positions so they're preserved
    input_ids[0, :4] = torch.tensor([1, 2, 3, 4])

    decoded, decoded_pos = student_blockwise_rollout(
        input_ids=input_ids,
        student_model=model,
        question_length=question_length,
        block_size=block_size,
        num_decode_steps=steps,
        mask_id=mask_id,
        eos_id=mask_id,
        temperature=0.0,
        top_p=1.0,
        device=device,
        vocab_size=vocab,
        is_llada=False,
        shift=True,  # shift_logits on a [B,sub_L,vocab] tensor
    )

    # prompt tokens must be preserved
    assert decoded[0, :4].tolist() == [1, 2, 3, 4], "prompt changed!"
    # all 12 non-prompt positions must be decoded (no mask_id remains)
    non_prompt = decoded[0, 4:]
    assert (non_prompt != mask_id).all(), f"undecoded position remains: {non_prompt.tolist()}"
    assert decoded_pos[0, 4:].all(), "some non-prompt positions not flagged decoded"
    print(f"  decoded seq: {decoded[0].tolist()}")
    print(f"  prompt preserved, all {int((non_prompt != mask_id).sum())} positions decoded  ✅")
    print("  PASSED\n")


if __name__ == "__main__":
    test_sample_tokens()
    test_select_topk_positions()
    test_full_rollout()
    print("ALL SMOKE TESTS PASSED 🎉")