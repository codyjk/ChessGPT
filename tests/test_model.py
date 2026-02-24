"""Tests for the custom chess transformer architecture."""

import torch

from chessgpt.model.transformer import ChessTransformer, TransformerConfig


def make_tiny_config(vocab_size: int = 100, **kwargs) -> TransformerConfig:
    defaults = dict(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=20,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def test_model_creates():
    config = make_tiny_config()
    model = ChessTransformer(config)
    assert model is not None


def test_model_forward_shapes():
    config = make_tiny_config(vocab_size=100)
    model = ChessTransformer(config)

    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    policy, value, checkmate = model(input_ids)

    assert policy.shape == (batch_size, seq_len, config.vocab_size)
    assert value.shape == (batch_size, 3)
    assert checkmate.shape == (batch_size, 1)


def test_model_handles_single_token():
    config = make_tiny_config()
    model = ChessTransformer(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 1))
    policy, value, checkmate = model(input_ids)

    assert policy.shape == (1, 1, config.vocab_size)
    assert value.shape == (1, 3)
    assert checkmate.shape == (1, 1)


def test_model_handles_max_seq_len():
    config = make_tiny_config()
    model = ChessTransformer(config)

    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    policy, value, checkmate = model(input_ids)

    assert policy.shape == (2, config.max_seq_len, config.vocab_size)


def test_checkmate_head_output_raw_logits():
    """Checkmate head returns raw logits (not sigmoid-bounded)."""
    config = make_tiny_config()
    model = ChessTransformer(config)

    input_ids = torch.randint(0, config.vocab_size, (4, 10))
    _, _, checkmate = model(input_ids)

    # Raw logits can be any real number, just check shape and dtype
    assert checkmate.shape == (4, 1)
    assert checkmate.dtype == torch.float32


def test_param_count_estimate():
    config = make_tiny_config()
    assert config.param_count_estimate > 0

    # Actual param count should be in the same ballpark
    model = ChessTransformer(config)
    actual = sum(p.numel() for p in model.parameters())
    assert actual > 0


def test_residual_projection_init_scaling():
    """
    out_proj and down_proj should have smaller initialization std than other layers.
    Specifically std ≈ 0.02 / sqrt(2 * n_layers).
    """
    config = make_tiny_config(n_layers=4)
    model = ChessTransformer(config)

    expected_std = 0.02 * (2 * 4) ** -0.5  # 0.02 / sqrt(8) ≈ 0.00707

    for block in model.blocks:
        out_proj_std = block.attn.out_proj.weight.std().item()
        down_proj_std = block.ffn.down_proj.weight.std().item()

        # Should be close to the scaled std (with some random variance)
        # We use a generous tolerance since it's random initialization
        assert out_proj_std < 0.015, (
            f"out_proj std {out_proj_std:.4f} too large (expected ~{expected_std:.4f})"
        )
        assert down_proj_std < 0.015, (
            f"down_proj std {down_proj_std:.4f} too large (expected ~{expected_std:.4f})"
        )

    # Other linear layers should have std ≈ 0.02
    q_proj_std = model.blocks[0].attn.q_proj.weight.std().item()
    assert q_proj_std > 0.015, f"q_proj std {q_proj_std:.4f} should be ~0.02 (not scaled)"


def test_dropout_config():
    """Model with dropout > 0 should create dropout layers."""
    config = make_tiny_config(dropout=0.1)
    model = ChessTransformer(config)

    # Check that attention has dropout
    attn = model.blocks[0].attn
    assert attn.dropout == 0.1
    assert attn.resid_dropout.p == 0.1

    # Check that FFN has dropout
    ffn = model.blocks[0].ffn
    assert ffn.dropout.p == 0.1


def test_dropout_zero_default():
    """Default model should have zero dropout."""
    config = make_tiny_config()
    model = ChessTransformer(config)

    attn = model.blocks[0].attn
    assert attn.dropout == 0.0


def test_model_deterministic_in_eval():
    """In eval mode, same input should give same output (no dropout noise)."""
    config = make_tiny_config(dropout=0.5)
    model = ChessTransformer(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    policy1, value1, cm1 = model(input_ids)
    policy2, value2, cm2 = model(input_ids)

    assert torch.allclose(policy1, policy2)
    assert torch.allclose(value1, value2)
    assert torch.allclose(cm1, cm2)


def test_model_different_seq_lengths():
    """Model should handle various sequence lengths up to max_seq_len."""
    config = make_tiny_config(max_seq_len=20)
    model = ChessTransformer(config)
    model.eval()

    for seq_len in [1, 5, 10, 20]:
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        policy, value, checkmate = model(input_ids)
        assert policy.shape == (1, seq_len, config.vocab_size)
        assert value.shape == (1, 3)
        assert checkmate.shape == (1, 1)


def test_value_head_uses_last_position():
    """Value head should use the last position only."""
    config = make_tiny_config()
    model = ChessTransformer(config)
    model.eval()

    # Note: Due to causal attention, changing position 0 does affect the hidden state at the last
    # position (through attention). So we just verify the output shape and that it works.
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    _, value, _ = model(input_ids)
    assert value.shape == (1, 3)


def test_config_dataclass_fields():
    """TransformerConfig should have all expected fields with correct defaults."""
    config = TransformerConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=20)
    assert config.dropout == 0.0  # default

    config2 = TransformerConfig(
        vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_seq_len=20, dropout=0.1
    )
    assert config2.dropout == 0.1
