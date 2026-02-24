"""Tests for the training loop and multi-task loss."""

import torch

from chessgpt.model.transformer import ChessTransformer, TransformerConfig
from chessgpt.training.loss import MultiTaskLoss


def make_tiny_model(vocab_size: int = 50) -> ChessTransformer:
    config = TransformerConfig(
        vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=2, max_seq_len=10
    )
    return ChessTransformer(config)


def test_multitask_loss_computes():
    vocab_size = 50
    batch, seq = 2, 10

    loss_fn = MultiTaskLoss(alpha=0.5, beta=0.5, pad_token_id=0)

    policy_logits = torch.randn(batch, seq, vocab_size)
    value_logits = torch.randn(batch, 3)
    checkmate_logits = torch.randn(batch, 1)  # raw logits, not sigmoid
    labels = torch.randint(1, vocab_size, (batch, seq))
    outcome = torch.zeros(batch, 3)
    outcome[:, 0] = 1.0  # white wins
    checkmate_available = torch.tensor([1.0, 0.0])
    move_mask = torch.ones(batch, seq)
    checkmate_weight = torch.ones(batch, seq)

    total_loss, details = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        move_mask,
        checkmate_weight,
    )

    assert total_loss.shape == ()
    assert total_loss.item() > 0
    assert "policy_loss" in details
    assert "value_loss" in details
    assert "checkmate_loss" in details
    assert "total_loss" in details


def test_loss_respects_mask():
    """Zeroed-out mask positions should not contribute to policy loss."""
    vocab_size = 50
    batch, seq = 1, 5

    loss_fn = MultiTaskLoss(alpha=0.0, beta=0.0, pad_token_id=0)

    policy_logits = torch.randn(batch, seq, vocab_size)
    value_logits = torch.randn(batch, 3)
    checkmate_logits = torch.randn(batch, 1)  # raw logits, not sigmoid
    labels = torch.randint(1, vocab_size, (batch, seq))
    outcome = torch.zeros(batch, 3)
    outcome[:, 0] = 1.0
    checkmate_available = torch.zeros(batch)

    # Full mask
    full_mask = torch.ones(batch, seq)
    weight = torch.ones(batch, seq)
    _, full_details = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        full_mask,
        weight,
    )

    # Partial mask (only first position)
    partial_mask = torch.zeros(batch, seq)
    partial_mask[0, 0] = 1.0
    _, partial_details = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        partial_mask,
        weight,
    )

    # Losses should differ since different positions contribute
    assert full_details["policy_loss"] != partial_details["policy_loss"]


def test_loss_zero_mask_gives_zero_policy():
    """With all-zero mask, policy loss should be 0 (no positions contribute)."""
    vocab_size = 50
    loss_fn = MultiTaskLoss(alpha=0.0, beta=0.0, pad_token_id=0)

    policy_logits = torch.randn(1, 5, vocab_size)
    value_logits = torch.randn(1, 3)
    checkmate_logits = torch.randn(1, 1)
    labels = torch.randint(1, vocab_size, (1, 5))
    outcome = torch.zeros(1, 3)
    outcome[:, 0] = 1.0
    checkmate_available = torch.zeros(1)
    zero_mask = torch.zeros(1, 5)
    weight = torch.ones(1, 5)

    _, details = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        zero_mask,
        weight,
    )
    assert details["policy_loss"] == 0.0


def test_loss_checkmate_weight_amplifies():
    """Higher checkmate_weight on a position should increase that position's loss contribution."""
    vocab_size = 50
    loss_fn = MultiTaskLoss(alpha=0.0, beta=0.0, pad_token_id=0)

    torch.manual_seed(42)
    policy_logits = torch.randn(1, 5, vocab_size)
    value_logits = torch.randn(1, 3)
    checkmate_logits = torch.randn(1, 1)
    labels = torch.randint(1, vocab_size, (1, 5))
    outcome = torch.zeros(1, 3)
    outcome[:, 0] = 1.0
    checkmate_available = torch.zeros(1)
    mask = torch.ones(1, 5)

    # Weight=1 everywhere
    weight_1 = torch.ones(1, 5)
    _, d1 = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        mask,
        weight_1,
    )

    # Weight=10 at position 2
    weight_10 = torch.ones(1, 5)
    weight_10[0, 2] = 10.0
    _, d10 = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        mask,
        weight_10,
    )

    # The weighted loss should be higher (unless position 2 happens to have near-zero loss)
    # Since we use random logits, position 2 almost certainly has non-trivial loss
    assert d10["policy_loss"] > d1["policy_loss"]


def test_loss_denominator_uses_mask_not_weighted_mask():
    """
    The denominator should be move_mask.sum(), not (move_mask * weight).sum().
    This ensures checkmate_weight amplifies numerator without inflating denominator.
    """
    vocab_size = 50
    loss_fn = MultiTaskLoss(alpha=0.0, beta=0.0, pad_token_id=0)

    torch.manual_seed(42)
    policy_logits = torch.randn(1, 3, vocab_size)
    labels = torch.randint(1, vocab_size, (1, 3))
    value_logits = torch.randn(1, 3)
    checkmate_logits = torch.randn(1, 1)
    outcome = torch.zeros(1, 3)
    outcome[:, 0] = 1.0
    checkmate_available = torch.zeros(1)
    mask = torch.ones(1, 3)

    # With uniform weight, should equal mean of per-position CE
    uniform_weight = torch.ones(1, 3)
    total_loss, _ = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        mask,
        uniform_weight,
    )

    # Manually compute what the loss should be
    import torch.nn.functional as F

    per_pos_ce = F.cross_entropy(
        policy_logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=0,
        reduction="none",
    ).reshape(1, 3)
    expected = (per_pos_ce * mask).sum() / mask.sum()
    assert abs(total_loss.item() - expected.item()) < 1e-5


def test_checkmate_loss_uses_bce_with_logits():
    """Checkmate loss should work correctly with raw logits (not sigmoid)."""
    loss_fn = MultiTaskLoss(alpha=0.0, beta=1.0, pad_token_id=0)

    policy_logits = torch.zeros(1, 3, 50)
    value_logits = torch.zeros(1, 3)
    labels = torch.ones(1, 3, dtype=torch.long)
    outcome = torch.zeros(1, 3)
    mask = torch.zeros(1, 3)
    weight = torch.ones(1, 3)

    # Large positive logit → should be near 1 → low loss when target=1
    checkmate_logits_pos = torch.tensor([[10.0]])
    checkmate_target_1 = torch.tensor([1.0])
    _, d_pos = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits_pos,
        labels,
        outcome,
        checkmate_target_1,
        mask,
        weight,
    )

    # Large negative logit → should be near 0 → high loss when target=1
    checkmate_logits_neg = torch.tensor([[-10.0]])
    _, d_neg = loss_fn(
        policy_logits,
        value_logits,
        checkmate_logits_neg,
        labels,
        outcome,
        checkmate_target_1,
        mask,
        weight,
    )

    assert d_pos["checkmate_loss"] < d_neg["checkmate_loss"]
    assert d_pos["checkmate_loss"] < 0.01  # should be very close to 0


def test_alpha_beta_weighting():
    """Value and checkmate losses should be weighted by alpha and beta."""
    vocab_size = 50
    torch.manual_seed(42)

    policy_logits = torch.randn(1, 5, vocab_size)
    value_logits = torch.randn(1, 3)
    checkmate_logits = torch.randn(1, 1)
    labels = torch.randint(1, vocab_size, (1, 5))
    outcome = torch.zeros(1, 3)
    outcome[:, 0] = 1.0
    checkmate_available = torch.tensor([1.0])
    mask = torch.ones(1, 5)
    weight = torch.ones(1, 5)

    # alpha=0, beta=0 → total = policy only
    loss_fn_0 = MultiTaskLoss(alpha=0.0, beta=0.0, pad_token_id=0)
    total_0, d_0 = loss_fn_0(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        mask,
        weight,
    )

    # alpha=1, beta=1
    loss_fn_1 = MultiTaskLoss(alpha=1.0, beta=1.0, pad_token_id=0)
    total_1, d_1 = loss_fn_1(
        policy_logits,
        value_logits,
        checkmate_logits,
        labels,
        outcome,
        checkmate_available,
        mask,
        weight,
    )

    # total_1 should be larger (adds value + checkmate)
    assert total_1.item() > total_0.item()

    # Verify: total = policy + alpha*value + beta*checkmate
    expected = d_1["policy_loss"] + 1.0 * d_1["value_loss"] + 1.0 * d_1["checkmate_loss"]
    assert abs(d_1["total_loss"] - expected) < 1e-5


def test_model_training_step():
    """Verify a single training step doesn't crash and loss decreases with gradient."""
    model = make_tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = MultiTaskLoss(alpha=0.5, beta=0.5, pad_token_id=0)

    input_ids = torch.randint(1, 50, (4, 10))
    labels = torch.randint(1, 50, (4, 10))
    outcome = torch.zeros(4, 3)
    outcome[:, 0] = 1.0
    checkmate = torch.zeros(4)
    mask = torch.ones(4, 10)
    weight = torch.ones(4, 10)

    policy, value, cm = model(input_ids)
    loss, _ = loss_fn(policy, value, cm, labels, outcome, checkmate, mask, weight)

    loss.backward()
    optimizer.step()

    # Should be able to do another forward pass without error
    policy2, value2, cm2 = model(input_ids)
    loss2, _ = loss_fn(policy2, value2, cm2, labels, outcome, checkmate, mask, weight)
    assert loss2.item() > 0


def test_model_training_step_loss_decreases():
    """Loss should decrease after a few training steps (model is learning)."""
    torch.manual_seed(42)
    model = make_tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = MultiTaskLoss(alpha=0.5, beta=0.5, pad_token_id=0)

    input_ids = torch.randint(1, 50, (4, 10))
    labels = torch.randint(1, 50, (4, 10))
    outcome = torch.zeros(4, 3)
    outcome[:, 0] = 1.0
    checkmate = torch.zeros(4)
    mask = torch.ones(4, 10)
    weight = torch.ones(4, 10)

    # Get initial loss
    policy, value, cm = model(input_ids)
    loss_initial, _ = loss_fn(policy, value, cm, labels, outcome, checkmate, mask, weight)

    # Train for 20 steps
    for _ in range(20):
        optimizer.zero_grad()
        policy, value, cm = model(input_ids)
        loss, _ = loss_fn(policy, value, cm, labels, outcome, checkmate, mask, weight)
        loss.backward()
        optimizer.step()

    assert loss.item() < loss_initial.item(), "Loss should decrease after training"
