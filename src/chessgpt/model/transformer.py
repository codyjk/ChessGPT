"""
ChessTransformer: full model assembly.

A decoder-only transformer purpose-built for chess move prediction. Takes a sequence
of tokenized chess moves and produces three outputs:

1. Policy logits — next move prediction at every position
2. Value logits — game outcome prediction (white/draw/black)
3. Checkmate probability — whether checkmate is available

Architecture:
    Token embedding (no position embedding — RoPE handles position inside attention)
    → N transformer blocks (pre-norm, causal self-attention, SwiGLU FFN)
    → Final RMSNorm
    → Three output heads (policy, value, checkmate)

This is essentially a small Llama-style model with chess-specific output heads.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .heads import CheckmateHead, PolicyHead, ValueHead
from .layers import RMSNorm, TransformerBlock


@dataclass
class TransformerConfig:
    """All hyperparameters needed to construct a ChessTransformer."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    dropout: float = 0.0

    @property
    def param_count_estimate(self) -> int:
        """Rough parameter count (for logging, not exact)."""
        # Embedding + n_layers * (attn + ffn) + heads
        d_ff = int(8 / 3 * self.d_model)
        d_ff = ((d_ff + 7) // 8) * 8
        per_block = 4 * self.d_model**2 + 3 * self.d_model * d_ff + 2 * self.d_model
        return self.vocab_size * self.d_model + self.n_layers * per_block


class ChessTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.n_heads, config.max_seq_len, dropout=config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.policy_head = PolicyHead(config.d_model, config.vocab_size)
        self.value_head = ValueHead(config.d_model)
        self.checkmate_head = CheckmateHead(config.d_model)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights. Residual-stream projections (out_proj, down_proj) are scaled
        by 1/sqrt(2*n_layers) to keep the residual stream variance stable as depth grows.
        This follows GPT-2 / Llama convention.
        """
        residual_scale = (2 * self.config.n_layers) ** -0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Scale down residual projections (attn out_proj + FFN down_proj)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * residual_scale)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=0.02 * residual_scale)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input_ids: [batch, seq_len] token IDs

        Returns:
            policy_logits:    [batch, seq_len, vocab_size]
            value_logits:     [batch, 3] (white/draw/black)
            checkmate_logits: [batch, 1] raw logits (apply sigmoid for probability)
        """
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)
        checkmate_logits = self.checkmate_head(x)

        return policy_logits, value_logits, checkmate_logits
