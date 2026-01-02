#!/usr/bin/env python3
"""Test if a trained model makes reasonable chess move predictions."""

import sys
import torch
import chess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_model.model import ChessTokenizer, ChessTransformer


def load_model(model_path, tokenizer_path, max_context=100, n_embd=1024, n_layer=24, n_head=16):
    """Load a trained model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = ChessTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size

    model = ChessTransformer(
        vocab_size=vocab_size,
        n_positions=max_context,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_move(model, tokenizer, device, move_history, top_k=5):
    """Predict next move given move history."""
    # Tokenize and pad
    input_ids = tokenizer.encode_and_pad(move_history, model.config.n_positions)
    input_tensor = torch.tensor([input_ids]).to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(input_tensor)
        # Get logits for last position
        last_logits = logits[0, -1, :]

        # Get top-k predictions
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            move = tokenizer.decode([int(idx)])
            # decode returns a list, get first element
            if isinstance(move, list):
                move = move[0] if move else "[UNK]"
            predictions.append((move, float(prob)))

        return predictions


def is_legal_move(board: chess.Board, move_str: str) -> bool:
    """Check if a move string is legal in the current position."""
    try:
        move = board.parse_san(move_str)
        return move in board.legal_moves
    except:
        return False


def test_model_on_positions():
    """Test model on various chess positions."""
    print("=" * 60)
    print("Testing ChessGPT Model Predictions")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model, tokenizer, device = load_model(
        "models/gpt2-test-5k/final_model/model.pth",
        "models/gpt2-test-5k/final_model/tokenizer.json",
    )
    print(f"   ✓ Model loaded on {device}")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Test positions
    test_cases = [
        {
            "name": "Opening position (e4 start)",
            "moves": [],
            "description": "Starting position, white to move"
        },
        {
            "name": "King's Pawn Game",
            "moves": ["e4", "e5"],
            "description": "After 1.e4 e5, white to move"
        },
        {
            "name": "Sicilian Defense",
            "moves": ["e4", "c5", "Nf3"],
            "description": "After 1.e4 c5 2.Nf3, black to move"
        },
        {
            "name": "Mid-game position",
            "moves": ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"],
            "description": "Ruy Lopez, white to move"
        },
    ]

    print("\n2. Testing predictions on various positions...")
    print("=" * 60)

    total_tests = 0
    legal_predictions = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"  {test_case['description']}")
        print(f"  Move history: {' '.join(test_case['moves']) if test_case['moves'] else '(empty)'}")

        # Get board position
        board = chess.Board()
        for move_str in test_case['moves']:
            move = board.parse_san(move_str)
            board.push(move)

        # Predict next move
        predictions = predict_move(model, tokenizer, device, test_case['moves'], top_k=5)

        print(f"\n  Top 5 predictions:")
        for j, (move, prob) in enumerate(predictions, 1):
            legal = is_legal_move(board, move)
            legal_str = "✓ LEGAL" if legal else "✗ illegal"
            print(f"    {j}. {move:6s} ({prob:.1%}) {legal_str}")

            total_tests += 1
            if legal:
                legal_predictions += 1

        print(f"\n  Legal moves in position: {len(list(board.legal_moves))}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total predictions tested: {total_tests}")
    print(f"Legal predictions: {legal_predictions} ({legal_predictions/total_tests*100:.1f}%)")
    print(f"Illegal predictions: {total_tests - legal_predictions} ({(total_tests-legal_predictions)/total_tests*100:.1f}%)")

    if legal_predictions / total_tests >= 0.5:
        print("\n✓ Model shows reasonable chess understanding (>50% legal moves)")
    else:
        print("\n✗ Model needs more training (<50% legal moves)")

    return legal_predictions / total_tests


if __name__ == "__main__":
    legal_rate = test_model_on_positions()
    sys.exit(0 if legal_rate >= 0.5 else 1)
