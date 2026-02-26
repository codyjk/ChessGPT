"""
Multi-task loss for chess transformer training.

Combines three losses:
1. Policy loss — cross-entropy on next-move prediction with outcome masking + checkmate weighting
2. Value loss — cross-entropy on game outcome (white/draw/black)
3. Checkmate loss — binary cross-entropy on checkmate detection

Total: policy_loss + alpha * value_loss + beta * checkmate_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Computes weighted sum of policy, value, and checkmate losses.

    Policy loss uses per-position masking (outcome-based) and per-position weighting
    (checkmate moves weighted higher). The mask and weights are provided by the dataset.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, pad_token_id: int = 0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pad_token_id = pad_token_id

    def forward(
        self,
        policy_logits: torch.Tensor,  # [batch, seq, vocab]
        value_logits: torch.Tensor,  # [batch, 3]
        checkmate_logits: torch.Tensor,  # [batch, 1]
        labels: torch.Tensor,  # [batch, seq]
        outcome: torch.Tensor,  # [batch, 3]
        checkmate_available: torch.Tensor,  # [batch]
        move_mask: torch.Tensor,  # [batch, seq]
        checkmate_weight: torch.Tensor,  # [batch, seq]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (total_loss, {policy_loss, value_loss, checkmate_loss})."""

        # --- Policy loss ---
        # Cross-entropy per position, then apply outcome mask and checkmate weight
        batch, seq, vocab = policy_logits.shape
        policy_ce = F.cross_entropy(
            policy_logits.reshape(-1, vocab),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction="none",
        ).reshape(batch, seq)

        # Apply outcome mask (zeros out losing player's moves) and checkmate weight.
        # Denominator uses move_mask.sum() (count of active positions), not weighted sum,
        # so checkmate_weight amplifies the checkmate move's contribution without
        # inflating the denominator.
        weighted_ce = policy_ce * move_mask * checkmate_weight
        mask_sum = move_mask.sum()
        policy_loss = weighted_ce.sum() / mask_sum.clamp(min=1.0)

        # --- Value loss ---
        value_loss = F.cross_entropy(value_logits, outcome)

        # --- Checkmate loss ---
        checkmate_loss = F.binary_cross_entropy_with_logits(
            checkmate_logits.squeeze(-1), checkmate_available
        )

        total = policy_loss + self.alpha * value_loss + self.beta * checkmate_loss

        details = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "checkmate_loss": checkmate_loss.item(),
            "total_loss": total.item(),
        }

        return total, details
