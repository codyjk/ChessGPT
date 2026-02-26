"""
Multi-head causal self-attention with Rotary Position Embeddings (RoPE).

Self-attention lets each position in a sequence "look at" other positions to gather
context. For chess, this means each move can attend to all previous moves to understand
the game state. Causal masking ensures moves can only look backward, not forward.

RoPE encodes position via rotation of query/key vectors rather than adding position
embeddings. Advantages over learned position embeddings (GPT-2 style):
- Relative position awareness: attention between two positions depends on their
  distance, not absolute position. "Nf3 after e4" means the same whether it's
  moves 1-2 or moves 15-16.
- Length generalization: can handle sequences longer than seen in training,
  since the rotation pattern extends naturally.
- No extra parameters: position information is baked into the attention computation.

Multi-head attention runs several attention computations in parallel with different
learned projections. Each head can specialize (e.g., one head might track piece
development, another might track king safety). Outputs are concatenated and projected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the complex-valued rotation frequencies for RoPE.

    For each pair of dimensions (2i, 2i+1), we compute a rotation angle that
    increases with position and decreases with dimension index. Lower dimensions
    rotate faster (capturing fine-grained local patterns), higher dimensions
    rotate slower (capturing broader patterns).

    Returns: [max_seq_len, dim // 2] complex tensor of rotation factors.
    """
    # Frequency for each dimension pair: theta^(-2i/d) for i in [0, dim//2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Position indices
    t = torch.arange(max_seq_len).float()
    # Outer product: [seq_len, dim//2] angles
    angles = torch.outer(t, freqs)
    # Convert to complex rotation factors: e^(i * angle)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to query or key tensor.

    Takes pairs of dimensions and rotates them by position-dependent angles.
    This is equivalent to multiplying pairs (x[2i], x[2i+1]) by complex
    rotation factors, which encodes relative position into the dot product.

    x: [batch, n_heads, seq_len, head_dim]
    freqs: [seq_len, head_dim // 2] complex tensor
    """
    # Reshape to pairs of dimensions and view as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Broadcast freqs to match batch and head dims: [1, 1, seq_len, head_dim//2]
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    # Rotate and convert back to real pairs
    rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return rotated.type_as(x)


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Each head independently computes:
        1. Project input to queries (Q), keys (K), values (V)
        2. Apply RoPE to Q and K (encodes position)
        3. Compute attention scores: softmax(Q @ K^T / sqrt(d_k))
        4. Mask future positions (causal)
        5. Weighted sum of values

    Multiple heads run in parallel, then concatenate and project.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Q, K, V projections — no bias (standard in modern transformers)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Output projection after concatenating heads
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies (not a parameter — deterministic)
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(self.head_dim, max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        Returns: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multi-head: [batch, n_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys (values don't get position encoding)
        q = apply_rope(q, self.rope_freqs[:seq_len])
        k = apply_rope(k, self.rope_freqs[:seq_len])

        # Scaled dot-product attention with causal mask
        # PyTorch's F.scaled_dot_product_attention handles the causal mask efficiently
        # dropout_p is only applied during training
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)

        # Concatenate heads and project: [batch, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.out_proj(out))
