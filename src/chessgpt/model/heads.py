"""
Output heads for the chess transformer.

Three heads share the same backbone (transformer blocks) but produce different outputs:

PolicyHead:
    Predicts the next move at every position. This is the primary head — it's what
    actually plays chess. Output is a distribution over the vocabulary (all known moves).
    Trained with cross-entropy loss, outcome masking, and checkmate weighting.

ValueHead:
    Predicts game outcome (white wins / draw / black wins) from the last position.
    Teaches the model positional evaluation — positions near checkmate produce extreme
    values. Creates a "value gradient" the policy head can follow: each forcing move
    leads to a position with a more extreme value prediction.

CheckmateHead:
    Detects whether checkmate is available in the current position. This is a TRAINING
    SIGNAL ONLY — it improves the backbone's understanding of mating patterns, which
    indirectly helps the policy head find mates. It does NOT trigger any search or
    python-chess validation at inference time.
"""

import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """
    Next-move prediction at every sequence position.

    Simple linear projection from hidden state to vocabulary logits.
    Applied at every position so the model learns to predict the next move
    from any point in a game, not just the end.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model] → [batch, seq_len, vocab_size]"""
        return self.proj(x)


class ValueHead(nn.Module):
    """
    Game outcome prediction from the last hidden state.

    Two-layer MLP with SiLU activation, outputting 3 logits (white/draw/black).
    Only uses the last position's hidden state since game outcome is a property
    of the final position, not individual moves.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, 3, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model] → [batch, 3] (using last position only)"""
        return self.net(x[:, -1, :])


class CheckmateHead(nn.Module):
    """
    Checkmate availability detection from the last hidden state.

    Two-layer MLP with SiLU, outputting a single raw logit (NOT sigmoid).
    Predicts whether checkmate is deliverable in the current position.

    Returns raw logits so the loss function can use binary_cross_entropy_with_logits
    (numerically stable). Apply torch.sigmoid() at inference time for probabilities.

    This head exists purely as a training signal — forcing the backbone to develop
    representations that understand mating patterns. At inference, we display its
    output as a diagnostic but never use it to influence move selection.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model] → [batch, 1] raw logits (apply sigmoid for probs)"""
        return self.net(x[:, -1, :])
