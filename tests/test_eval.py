"""Tests for evaluation metrics."""

import torch

from chessgpt.evaluation.puzzles import evaluate_mate_in_1
from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer, TransformerConfig


def test_mate_in_1_runs():
    """Mate-in-1 eval should run without errors on untrained model."""
    config = TransformerConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=2, max_seq_len=20)
    model = ChessTransformer(config)

    tokenizer = ChessTokenizer()
    # Add some moves to the tokenizer so it can encode
    for i, move in enumerate(["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#", "f3", "g4", "Qh4#"]):
        tokenizer.move_to_id[move] = tokenizer.vocab_size
        tokenizer.id_to_move[tokenizer.vocab_size] = move
        tokenizer.vocab_size += 1

    result = evaluate_mate_in_1(model, tokenizer, torch.device("cpu"))
    assert "mate_in_1_top1" in result
    assert "mate_in_1_top3" in result
    assert "mate_in_1_top5" in result
    assert 0 <= result["mate_in_1_top1"] <= 1
