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

import contextlib
import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.on_policy_rollout import (
    _sample_tokens,
    _select_top1_position,
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
        self.logit_mode = None
        self.register_buffer("fake_logit_bias", torch.linspace(-0.2, 0.2, vocab))
        self.register_buffer("teacher_logit_bias", torch.linspace(0.15, -0.15, vocab))

    def forward(self, input_ids, attention_bias=None, attention_mask=None, **kw):
        emb = self.embed(input_ids) + self.pos[: input_ids.shape[1]]
        logits = self.proj(emb)
        position_bias = torch.outer(
            torch.linspace(-0.12, 0.18, input_ids.shape[1], device=logits.device),
            torch.linspace(0.3, -0.25, logits.shape[-1], device=logits.device),
        )
        if self.logit_mode == "fake":
            logits = logits + self.fake_logit_bias + position_bias
        elif self.logit_mode == "teacher":
            logits = logits + self.teacher_logit_bias - position_bias.flip(0)
        return type("Out", (), {"logits": logits})()

    @contextlib.contextmanager
    def disable_adapter(self):
        previous_mode = self.logit_mode
        self.logit_mode = "teacher"
        try:
            yield
        finally:
            self.logit_mode = previous_mode


class TinyEMA:
    @contextlib.contextmanager
    def swap(self, model):
        previous_mode = model.logit_mode
        model.logit_mode = "fake"
        try:
            yield
        finally:
            model.logit_mode = previous_mode


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


def test_joint_dmd_loss(device):
    print(f"=== test_joint_dmd_loss ({device}) ===")
    from utils.loss import compute_dmd_loss

    torch.manual_seed(7)
    vocab, d_model, L, mask_id = 32, 12, 10, 0
    model = TinyLLada(vocab, d_model, L).to(device)
    ema = TinyEMA()
    input_ids = torch.randint(1, vocab, (2, L), device=device)
    question_length = torch.tensor([2, 4], device=device)
    lengths = torch.tensor([6, 9], device=device)

    class TrainConfig(dict):
        def __init__(self, joint, position_temperature=1.0):
            super().__init__(training_mode="llada")
            self.train = ({"dmd_joint_action": True,
                           "dmd_position_temperature": position_temperature,
                           "dmd_token_temperature": 1.0} if joint else {})

    joint_transition_losses = []
    for joint, position_temperature in ((True, 1.0), (True, 0.5), (False, 1.0)):
        model.zero_grad()
        losses = compute_dmd_loss(
            input_ids=input_ids,
            denoiser=model,
            ema_lora=ema,
            question_length=question_length,
            mask_id=mask_id,
            block_size=8,
            enable_shift=False,
            share_steps=1,
            self_align=False,
            feature_align=False,
            self_step=False,
            eos_id=vocab - 1,
            student_decode_steps=1,
            temperature=0.0,
            top_p=1.0,
            config=TrainConfig(joint, position_temperature),
            lengths=lengths,
        )
        transition_loss = losses["transition_dmd_loss"]
        assert torch.isfinite(losses["loss"]), f"non-finite DMD loss (joint={joint})"
        assert torch.isfinite(transition_loss), f"non-finite transition DMD loss (joint={joint})"
        assert transition_loss.abs() > 1e-10, f"zero transition DMD loss (joint={joint})"
        if joint:
            joint_transition_losses.append(transition_loss)
        losses["loss"].backward()
        grad_abs_sum = sum(
            parameter.grad.abs().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        assert grad_abs_sum > 0, f"zero DMD gradient (joint={joint})"
        assert model.logit_mode is None, f"student forward mode leaked (joint={joint})"
        print(
            f"  joint={joint} position_temperature={position_temperature} loss={losses['loss'].item():.6f} "
            f"transition={transition_loss.item():.6f} grad_abs_sum={grad_abs_sum.item():.6f}  ✅"
        )
    assert not torch.allclose(*joint_transition_losses, atol=1e-6, rtol=1e-5), (
        "joint transition DMD loss should vary with dmd_position_temperature"
    )
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


if __name__ == "__main__":
    test_sample_tokens()
    test_select_top1_position()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_joint_dmd_loss(device)
    test_full_rollout(device)
    print("ALL SMOKE TESTS PASSED")