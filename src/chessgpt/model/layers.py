"""
Transformer building blocks: RMSNorm, SwiGLU FFN, and the full TransformerBlock.

RMSNorm (Root Mean Square Layer Normalization):
    Simpler and faster than LayerNorm. Instead of subtracting the mean and dividing
    by standard deviation, it just divides by the root-mean-square. Empirically works
    as well as LayerNorm for transformers (used in Llama, Gemma, etc.).

SwiGLU (Swish-Gated Linear Unit):
    A gated feed-forward network that's become standard since Llama 1. Uses three
    weight matrices instead of two: one for a "gate" that controls information flow.
    The gate uses SiLU (Swish) activation, which is smooth and avoids the dead neuron
    problem of ReLU. Learns to selectively amplify or suppress different features.

TransformerBlock:
    Pre-norm architecture: normalize BEFORE each sublayer, not after (more stable
    training). Each block is: attention → add residual → FFN → add residual.
    Residual connections are the "gradient highway" — they let gradients flow directly
    through the network without vanishing, enabling deep stacks.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadCausalSelfAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    For input x of dimension d:
        RMSNorm(x) = (x / RMS(x)) * gamma
        where RMS(x) = sqrt(mean(x^2) + eps)

    Gamma is a learnable scale parameter (initialized to 1).
    No bias term, no mean subtraction — just rescaling by magnitude.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Standard FFN: x → Linear → ReLU → Linear
    SwiGLU FFN:   x → (Linear_gate(x) * SiLU) ⊙ Linear_up(x) → Linear_down

    The gate projection learns WHICH features to amplify (via SiLU activation).
    The up projection provides the VALUES to amplify. Element-wise multiplication
    combines them. The down projection maps back to model dimension.

    Hidden dimension is typically ~2.67x model dimension (adjusted for the third
    weight matrix so total parameter count matches a standard 4x FFN).
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        # Default: 8/3 * d_model, rounded to nearest multiple of 8 for efficiency.
        # The 8/3 ratio compensates for SwiGLU having 3 weight matrices (gate, up, down)
        # vs the standard FFN's 2, keeping total parameter count ~equal to a 4x FFN.
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 7) // 8) * 8

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """
    A single transformer layer: attention + FFN with pre-norm and residual connections.

    Data flow:
        x → RMSNorm → Attention → + x (residual) → RMSNorm → SwiGLU FFN → + x (residual)

    Pre-norm (normalize before sublayer) is more stable than post-norm (GPT-2 style)
    because it keeps the residual stream unnormalized — the "main highway" of information
    flow. Each sublayer adds its contribution to this highway.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, max_seq_len, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with residual
        x = x + self.attn(self.attn_norm(x))
        # FFN sublayer with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x
