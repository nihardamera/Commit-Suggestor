"""The hand-written DPO loss against facts that follow from its formula.

The comparison with TRL's own loss on a real batch runs inside the DPO Kaggle
job (TRL computes it inline in its trainer, with no function to call here).
These tests need torch (`uv sync --extra train`); CI skips them.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from commit_suggestor.dpo import dpo_loss, sequence_logps  # noqa: E402


def fixed() -> tuple[torch.Tensor, ...]:
    pc = torch.tensor([-10.0, -12.0, -8.0])
    pr = torch.tensor([-11.0, -10.0, -9.0])  # pair 2: rejected gains more than chosen
    rc = torch.tensor([-10.5, -12.5, -8.0])
    rr = torch.tensor([-10.5, -11.0, -8.5])
    return pc, pr, rc, rr


def test_matches_the_formula_on_fixed_tensors():
    pc, pr, rc, rr = fixed()
    beta = 0.1
    margins = [(-10 + 10.5) - (-11 + 10.5), (-12 + 12.5) - (-10 + 11), (-8 + 8) - (-9 + 8.5)]
    expected = sum(-math.log(1 / (1 + math.exp(-beta * m))) for m in margins) / 3
    assert dpo_loss(pc, pr, rc, rr, beta).loss.item() == pytest.approx(expected, abs=1e-6)


def test_policy_equal_to_reference_gives_log_two():
    x = torch.tensor([-5.0, -7.0])
    assert dpo_loss(x, x, x, x, beta=0.3).loss.item() == pytest.approx(math.log(2))


def test_rewards_are_beta_times_the_log_ratio_and_accuracy_counts_wins():
    pc, pr, rc, rr = fixed()
    out = dpo_loss(pc, pr, rc, rr, beta=0.5)
    assert torch.allclose(out.chosen_rewards, 0.5 * (pc - rc))
    assert out.reward_accuracy.item() == pytest.approx(2 / 3)  # pair 2 rewards rejected more


def test_the_gradient_raises_chosen_and_lowers_rejected():
    pc, pr, rc, rr = (t.clone().requires_grad_(True) for t in fixed())
    dpo_loss(pc, pr, rc, rr, beta=0.1).loss.backward()
    assert (pc.grad < 0).all() and (pr.grad > 0).all()  # descent raises pc, lowers pr
    assert rc.grad is not None and (rc.grad > 0).all()


def test_larger_beta_penalises_a_wrong_margin_more():
    wrong = [
        torch.tensor([-12.0]),
        torch.tensor([-10.0]),
        torch.tensor([-11.0]),
        torch.tensor([-11.0]),
    ]
    assert dpo_loss(*wrong, beta=0.3).loss > dpo_loss(*wrong, beta=0.05).loss


def test_sequence_logps_sums_only_answer_tokens():
    vocab = 5
    logits = torch.zeros(1, 4, vocab)  # uniform: every token has log-prob -log 5
    labels = torch.tensor([[1, 2, 3, 4]])
    answer = torch.tensor([[0, 0, 1, 1]])  # the last two tokens are the answer
    assert sequence_logps(logits, labels, answer).item() == pytest.approx(-2 * math.log(vocab))
