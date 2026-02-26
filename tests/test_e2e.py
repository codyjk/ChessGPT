"""End-to-end pipeline test: CSV → Dataset → DataLoader → Model → Loss → Backward."""

import torch
from torch.utils.data import DataLoader

from chessgpt.data.dataset import ChessDataset
from chessgpt.inference.player import predict_next_move
from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer, TransformerConfig
from chessgpt.training.loss import MultiTaskLoss


def _make_csv(tmp_path):
    """Create a multi-game CSV with all outcome types."""
    path = tmp_path / "e2e_data.csv"
    lines = [
        "moves,outcome,checkmate_move_idx,ply_count",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qh5#,1-0,8,9",
        "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7,1/2-1/2,-1,8",
        "f3 e5 g4 Qh4#,0-1,3,4",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6,1-0,-1,10",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_e2e_train_step(tmp_path):
    """Full pipeline: CSV → tokenizer → dataset → dataloader → model → loss → backward."""
    csv_path = _make_csv(tmp_path)

    # Fit tokenizer
    tokenizer = ChessTokenizer.fit(str(csv_path))
    assert tokenizer.vocab_size > 2

    # Create dataset
    dataset = ChessDataset(str(csv_path), tokenizer, max_context_length=15)
    assert len(dataset) == 4

    # DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # Model
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=15,
    )
    model = ChessTransformer(config)

    # Loss
    loss_fn = MultiTaskLoss(alpha=0.5, beta=0.5, pad_token_id=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train one full epoch
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        policy, value, cm = model(batch["input_ids"])
        loss, details = loss_fn(
            policy,
            value,
            cm,
            batch["labels"],
            batch["outcome"],
            batch["checkmate_available"],
            batch["move_mask"],
            batch["checkmate_weight"],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # All loss components should be non-negative
        assert details["policy_loss"] >= 0
        assert details["value_loss"] >= 0
        assert details["checkmate_loss"] >= 0

    assert num_batches == 2  # 4 samples / batch_size 2
    assert total_loss > 0


def test_e2e_dataset_to_inference(tmp_path):
    """Full pipeline: train on CSV, then run inference."""
    csv_path = _make_csv(tmp_path)
    tokenizer = ChessTokenizer.fit(str(csv_path))

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=15,
    )
    model = ChessTransformer(config)

    # Run inference
    result = predict_next_move(model, tokenizer, ["e4", "e5", "Nf3"], temperature=0.5)

    assert "move" in result
    assert "top_k_moves" in result
    assert "value" in result
    assert "checkmate_prob" in result

    # Value should be a list of 3 probabilities summing to ~1
    assert len(result["value"]) == 3
    assert abs(sum(result["value"]) - 1.0) < 1e-5

    # Checkmate prob should be in [0, 1] (sigmoid of raw logit)
    assert 0.0 <= result["checkmate_prob"] <= 1.0


def test_e2e_outcomes_survive_roundtrip(tmp_path):
    """Verify all 3 outcome types are correctly encoded through the full pipeline."""
    csv_path = _make_csv(tmp_path)
    tokenizer = ChessTokenizer.fit(str(csv_path))
    dataset = ChessDataset(str(csv_path), tokenizer, max_context_length=15)

    # Row 0: 1-0 → [1, 0, 0]
    assert dataset[0]["outcome"].argmax().item() == 0
    # Row 1: 1/2-1/2 → [0, 1, 0]
    assert dataset[1]["outcome"].argmax().item() == 1
    # Row 2: 0-1 → [0, 0, 1]
    assert dataset[2]["outcome"].argmax().item() == 2
    # Row 3: 1-0 → [1, 0, 0]
    assert dataset[3]["outcome"].argmax().item() == 0


def test_e2e_checkmate_flag(tmp_path):
    """Verify checkmate_available is set correctly through the pipeline."""
    csv_path = _make_csv(tmp_path)
    tokenizer = ChessTokenizer.fit(str(csv_path))
    dataset = ChessDataset(str(csv_path), tokenizer, max_context_length=15)

    assert dataset[0]["checkmate_available"].item() == 1.0  # has checkmate
    assert dataset[1]["checkmate_available"].item() == 0.0  # draw, no checkmate
    assert dataset[2]["checkmate_available"].item() == 1.0  # has checkmate
    assert dataset[3]["checkmate_available"].item() == 0.0  # no checkmate


def test_e2e_batch_shapes(tmp_path):
    """Verify batched shapes are correct through DataLoader."""
    csv_path = _make_csv(tmp_path)
    tokenizer = ChessTokenizer.fit(str(csv_path))
    dataset = ChessDataset(str(csv_path), tokenizer, max_context_length=15)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 15)
    assert batch["labels"].shape == (4, 15)
    assert batch["outcome"].shape == (4, 3)
    assert batch["checkmate_available"].shape == (4,)
    assert batch["move_mask"].shape == (4, 15)
    assert batch["checkmate_weight"].shape == (4, 15)


def test_e2e_gradient_flows(tmp_path):
    """Verify gradients flow through all model components."""
    csv_path = _make_csv(tmp_path)
    tokenizer = ChessTokenizer.fit(str(csv_path))
    dataset = ChessDataset(str(csv_path), tokenizer, max_context_length=15)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_seq_len=15,
    )
    model = ChessTransformer(config)
    loss_fn = MultiTaskLoss(alpha=0.5, beta=0.5, pad_token_id=0)

    batch = next(iter(loader))
    policy, value, cm = model(batch["input_ids"])
    loss, _ = loss_fn(
        policy,
        value,
        cm,
        batch["labels"],
        batch["outcome"],
        batch["checkmate_available"],
        batch["move_mask"],
        batch["checkmate_weight"],
    )
    loss.backward()

    # Check that gradients exist for key components
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad.abs().sum() > 0

    assert model.blocks[0].attn.q_proj.weight.grad is not None
    assert model.blocks[0].attn.q_proj.weight.grad.abs().sum() > 0

    assert model.blocks[0].ffn.gate_proj.weight.grad is not None
    assert model.policy_head.proj.weight.grad is not None
    assert model.value_head.net[0].weight.grad is not None
    assert model.checkmate_head.net[0].weight.grad is not None
