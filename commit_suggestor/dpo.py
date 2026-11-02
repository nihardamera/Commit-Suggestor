"""The DPO loss, written out, to check TRL's against before trusting it.

DPO starts from the KL-constrained RL objective

    max_pi  E[r(x, y)] - beta * KL(pi(.|x) || pi_ref(.|x)),

whose optimum is pi*(y|x) = pi_ref(y|x) * exp(r(x, y) / beta) / Z(x). Solving
for r gives the implicit reward

    r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log Z(x).

Under the Bradley-Terry model, P(chosen > rejected) = sigmoid(r_c - r_r), and
the log Z(x) terms cancel because both answers share the prompt. Maximising
the likelihood of the preferred answers therefore gives

    loss = -log sigmoid(beta * [(log pi(y_c) - log pi_ref(y_c))
                                - (log pi(y_r) - log pi_ref(y_r))]).

beta sets how far the policy may move from the reference: small beta lets it
move far for a given margin, large beta keeps it close. Each log-prob is the
sum over the answer's tokens only (the prompt is shared).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DPOOutput:
    loss: torch.Tensor  # mean over the batch
    chosen_rewards: torch.Tensor  # beta * log-ratio, per pair, no gradient
    rejected_rewards: torch.Tensor
    reward_accuracy: torch.Tensor  # share of pairs where chosen gets the higher reward


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> DPOOutput:
    """Sigmoid DPO loss from summed answer log-probs, each of shape (batch,)."""
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    margin = chosen_logratios - rejected_logratios
    losses = -F.logsigmoid(beta * margin)
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    return DPOOutput(
        loss=losses.mean(),
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_accuracy=(chosen_rewards > rejected_rewards).float().mean(),
    )


def sequence_logps(
    logits: torch.Tensor, labels: torch.Tensor, answer_mask: torch.Tensor
) -> torch.Tensor:
    """Summed log-prob of each sequence's answer tokens.

    logits (batch, seq, vocab) predict the next token, so position t scores
    labels[:, t + 1]; answer_mask (batch, seq) is 1 on answer tokens.
    """
    logps = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = labels[:, 1:]
    per_token = torch.gather(logps, 2, targets.unsqueeze(-1)).squeeze(-1)
    return (per_token * answer_mask[:, 1:]).sum(dim=1)
