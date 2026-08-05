"""Minimal smoke test for the on-policy rollout / eval_llada alignment.

Verifies:
1. `_sample_tokens` returns confidence = sampled-token probability
   (no margin_confidence / neg_entropy branches).
2. `_select_top1_position` selects the highest-confidence masked position.
3. The full `student_blockwise_rollout` runs end-to-end on a tiny model,
   performing `num_decode_steps` forwards and decoding one token per block
   per forward.

Run:  python smoke_test_rollout_alignment.py
"""

import sys
import os

import torch
import torch.nn as nn
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.on_policy_rollout import (
    _sample_tokens,
    _select_top1_position,
    student_blockwise_rollout,
    student_blockwise_rollout_dmd,
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

    @contextmanager
    def disable_adapter(self):
        yield


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


def test_select_top1_position():
    print("=== test_select_top1_position ===")
    # Position 2 has the highest sampled-token confidence.
    logits = torch.tensor([
        [0.1, 0.2, 0.5],
        [0.4, 0.3, 0.3],
        [0.0, 0.0, 9.0],
        [0.2, 0.2, 0.6],
    ])
    mask = torch.tensor([True, True, True, True])
    pos = _select_top1_position(logits, mask, temperature=0.0)
    assert pos.tolist() == [2], f"top-1 should be pos2, got {pos.tolist()}"
    print(f"  selected top-1 position: {pos.tolist()}  ✅")

    mask2 = torch.tensor([True, False, True, False])
    pos2 = _select_top1_position(logits, mask2, temperature=0.0)
    assert pos2.tolist() == [2], f"masked top-1 should be pos2, got {pos2.tolist()}"
    print(f"  masked top-1 position: {pos2.tolist()}  ✅")
    print("  PASSED\n")


def test_full_rollout(device):
    print(f"=== test_full_rollout ({device}) ===")
    torch.manual_seed(42)
    vocab, d_model, L, mask_id = 100, 16, 16, 0
    block_size, steps = 4, 2  # two forwards; one decoded position per forward/block

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
    # There are three 4-token blocks after the prompt. Each block executes
    # exactly `steps` forwards and decodes one top-1 token per forward.
    non_prompt_decoded = int(decoded_pos[0, 4:].sum())
    assert non_prompt_decoded == 3 * steps, (
        f"expected {3 * steps} decoded positions, got {non_prompt_decoded}"
    )
    assert (decoded[0, 4:][decoded_pos[0, 4:]] != mask_id).all()
    assert (decoded[0, 4:][~decoded_pos[0, 4:]] == mask_id).all()
    print(f"  decoded seq: {decoded[0].tolist()}")
    print(f"  prompt preserved, {non_prompt_decoded} positions decoded  ✅")
    print("  PASSED\n")


class TinyEMA:
    @contextmanager
    def swap(self, model):
        yield


def test_transition_csm_loss(device):
    print(f"=== test_transition_csm_loss ({device}) ===")
    from utils.loss import compute_transition_csm_loss

    torch.manual_seed(11)
    vocab, d_model, L, mask_id = 32, 8, 12, 0
    model = TinyLLada(vocab, d_model, L).to(device).train()
    with torch.no_grad():
        model.proj.bias[mask_id] = -100.0
    input_ids = torch.zeros(2, L, dtype=torch.long, device=device)
    input_ids[0, :2] = torch.tensor([1, 2], device=device)
    input_ids[1, :4] = torch.tensor([3, 4, 5, 6], device=device)
    question_length = torch.tensor([2, 4], device=device)
    lengths = torch.tensor([8, 12], device=device)

    class Config(dict):
        train = {"aux_remaining_weight": 0.1, "transition_sample_ratio": 0.67}

    config = Config(training_mode="dream")
    backward_calls = []

    def backward(loss):
        backward_calls.append(loss.detach())
        loss.backward()

    losses = compute_transition_csm_loss(
        input_ids=input_ids, denoiser=model, ema_lora=TinyEMA(),
        question_length=question_length, mask_id=mask_id, block_size=3,
        enable_shift=True, eos_id=mask_id, student_decode_steps=1,
        temperature=0.0, top_p=1.0, config=config, lengths=lengths,
        backward_callback=backward,
    )
    assert losses["transition_count"].item() == 2
    expected_backward_calls = 2 + int(losses["remaining_mask_length"].item() > 0)
    assert len(backward_calls) == expected_backward_calls
    assert any(parameter.grad is not None for parameter in model.parameters())
    print(f"  {len(backward_calls)} backward call(s), two sampled block transitions  ✅")
    print("  PASSED\n")


def test_final_draft_remask_loss(device):
    print(f"=== test_final_draft_remask_loss ({device}) ===")
    from utils.loss import compute_final_draft_remask_loss

    torch.manual_seed(13)
    vocab, d_model, L, mask_id = 32, 8, 12, 0
    model = TinyLLada(vocab, d_model, L).to(device).train()
    with torch.no_grad():
        model.proj.bias[mask_id] = -100.0
    input_ids = torch.zeros(2, L, dtype=torch.long, device=device)
    input_ids[0, :2] = torch.tensor([1, 2], device=device)
    input_ids[1, :4] = torch.tensor([3, 4, 5, 6], device=device)
    question_length = torch.tensor([2, 4], device=device)
    lengths = torch.tensor([8, 12], device=device)

    class Config(dict):
        train = {"final_draft_remask_ratio": 0.5}

    losses = compute_final_draft_remask_loss(
        input_ids=input_ids, denoiser=model, question_length=question_length,
        mask_id=mask_id, block_size=3, enable_shift=True, eos_id=mask_id,
        student_decode_steps=1, temperature=0.0, top_p=1.0,
        config=Config(training_mode="dream"), lengths=lengths,
        backward_callback=lambda loss: loss.backward(),
    )
    assert losses["remaining_mask_length"].item() > 0
    assert any(parameter.grad is not None for parameter in model.parameters())
    print("  low-confidence remask correction backward  ✅")
    print("  PASSED\n")


def test_transition_csm_rollout(device):
    print(f"=== test_transition_csm_rollout ({device}) ===")
    torch.manual_seed(7)
    vocab, d_model, L, mask_id = 32, 8, 12, 0
    model = TinyLLada(vocab, d_model, L).to(device).eval()
    with torch.no_grad():
        model.proj.bias[mask_id] = -100.0

    input_ids = torch.zeros(2, L, dtype=torch.long, device=device)
    input_ids[0, :2] = torch.tensor([1, 2], device=device)
    input_ids[1, :4] = torch.tensor([3, 4, 5, 6], device=device)
    question_length = torch.tensor([2, 4], device=device)
    lengths = torch.tensor([8, 12], device=device)

    decoded, _, transitions = student_blockwise_rollout_dmd(
        input_ids=input_ids,
        student_model=model,
        question_length=question_length,
        block_size=3,
        num_decode_steps=1,
        mask_id=mask_id,
        eos_id=mask_id,
        temperature=0.0,
        top_p=1.0,
        device=device,
        is_llada=False,
        shift=True,
        lengths=lengths,
        transition_csm=True,
        transition_sample_ratio=1.0,
    )

    assert transitions, "transition CSM rollout produced no transitions"
    positions = torch.arange(L, device=device).unsqueeze(0)
    expected_answer_mask = (
        (positions >= question_length.unsqueeze(1))
        & (positions < lengths.unsqueeze(1))
    )
    assert len(transitions) == 3, f"expected three sampled block transitions, got {len(transitions)}"
    for transition in transitions:
        predecessor_ids = transition["predecessor_ids"]
        successor_ids = transition["successor_ids"]
        assert not predecessor_ids.requires_grad and predecessor_ids.grad_fn is None
        assert not successor_ids.requires_grad and successor_ids.grad_fn is None
        assert torch.equal(transition["answer_mask"], expected_answer_mask)
        assert not transition["answer_mask"][positions < question_length.unsqueeze(1)].any()
        assert not transition["answer_mask"][positions >= lengths.unsqueeze(1)].any()
        expected_advanced_mask = predecessor_ids.ne(successor_ids).any(dim=1)
        assert torch.equal(transition["advanced_mask"], expected_advanced_mask)
        csm_mask = transition["answer_mask"] & transition["advanced_mask"][:, None]
        assert torch.equal(csm_mask, expected_answer_mask & expected_advanced_mask[:, None])

    assert (decoded[expected_answer_mask] != mask_id).any(), "no answer token was decoded"
    assert torch.equal(decoded[~expected_answer_mask], input_ids[~expected_answer_mask]), (
        "prompt or padding changed"
    )
    print(f"  {len(transitions)} detached transitions; answer mask excludes prompt/padding  ✅")
    print("  PASSED\n")


if __name__ == "__main__":
    test_sample_tokens()
    test_select_top1_position()
    test_full_rollout(torch.device("cpu"))
    test_transition_csm_rollout(torch.device("cpu"))
    test_transition_csm_loss(torch.device("cpu"))
    test_final_draft_remask_loss(torch.device("cpu"))
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        test_full_rollout(device)
        test_transition_csm_rollout(device)
        test_transition_csm_loss(device)
        test_final_draft_remask_loss(device)
    print("ALL SMOKE TESTS PASSED")