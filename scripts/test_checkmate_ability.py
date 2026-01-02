#!/usr/bin/env python3
"""Test model's ability to find checkmates - Quick validation."""

import sys
import json
import torch
import chess
from pathlib import Path
from typing import List, Tuple

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


def predict_moves(model, tokenizer, device, move_history: List[str], top_k=10) -> List[Tuple[str, float]]:
    """Predict next moves given move history."""
    input_ids = tokenizer.encode_and_pad(move_history, model.config.n_positions)
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        last_logits = logits[0, -1, :]

        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            move = tokenizer.decode([int(idx)])
            if isinstance(move, list):
                move = move[0] if move else "[UNK]"
            predictions.append((move, float(prob)))

        return predictions


def is_checkmate_move(board: chess.Board, move_str: str) -> bool:
    """Check if a move results in checkmate."""
    try:
        move = board.parse_san(move_str)
        board_copy = board.copy()
        board_copy.push(move)
        return board_copy.is_checkmate()
    except:
        return False


def is_legal_move(board: chess.Board, move_str: str) -> bool:
    """Check if a move is legal."""
    try:
        move = board.parse_san(move_str)
        return move in board.legal_moves
    except:
        return False


def test_checkmate_ability(model_name: str):
    """Test model's checkmate detection ability."""
    print("=" * 70)
    print(f"CHECKMATE ABILITY TEST: {model_name}")
    print("=" * 70)

    # Load model
    print("\n📦 Loading model...")
    model, tokenizer, device = load_model(
        f"models/{model_name}/final_model/model.pth",
        f"models/{model_name}/final_model/tokenizer.json",
    )
    print(f"   ✓ Loaded on {device}")

    # Load test puzzles
    with open("data/checkmate_test_puzzles.json") as f:
        puzzles = json.load(f)

    # Test mate-in-1 positions
    print("\n" + "=" * 70)
    print("TEST 1: MATE-IN-1 POSITIONS")
    print("=" * 70)

    mate_in_1_found = 0
    mate_in_1_total = 0

    for puzzle in puzzles["mate_in_1"]:
        if "Already checkmate" in puzzle["description"]:
            continue  # Skip positions that are already checkmate

        mate_in_1_total += 1
        print(f"\n🔍 {puzzle['id']}: {puzzle['description']}")
        print(f"   Position: {' '.join(puzzle['moves_leading_to_position'])}")
        print(f"   Solution: {puzzle['solution']}")

        # Get board position
        board = chess.Board()
        for move_str in puzzle["moves_leading_to_position"]:
            move = board.parse_san(move_str)
            board.push(move)

        # Predict moves
        predictions = predict_moves(model, tokenizer, device, puzzle["moves_leading_to_position"], top_k=10)

        # Check if checkmate is found
        checkmate_found = False
        checkmate_rank = None

        print(f"\n   Model's top predictions:")
        for i, (move, prob) in enumerate(predictions[:5], 1):
            is_legal = is_legal_move(board, move)
            is_mate = is_checkmate_move(board, move) if is_legal else False

            symbol = "🎯" if is_mate else ("✓" if is_legal else "✗")
            status = "MATE!" if is_mate else ("legal" if is_legal else "illegal")

            print(f"      {i}. {move:8s} ({prob:5.1%}) {symbol} {status}")

            if is_mate and not checkmate_found:
                checkmate_found = True
                checkmate_rank = i

        if checkmate_found:
            mate_in_1_found += 1
            print(f"\n   ✅ FOUND checkmate at rank {checkmate_rank}")
        else:
            print(f"\n   ❌ MISSED checkmate")

    # Test checkmate priority
    print("\n" + "=" * 70)
    print("TEST 2: CHECKMATE PRIORITY (when checkmate available)")
    print("=" * 70)

    checkmate_prioritized = 0
    priority_total = 0

    for puzzle in puzzles["checkmate_available"]:
        # Only test positions where checkmate is actually available
        if puzzle["best_move"].endswith("#"):
            priority_total += 1
            print(f"\n🔍 {puzzle['id']}: {puzzle['description']}")
            print(f"   Position: {' '.join(puzzle['moves_leading_to_position'])}")
            print(f"   Best move: {puzzle['best_move']} (CHECKMATE)")

            # Get board position
            board = chess.Board()
            for move_str in puzzle["moves_leading_to_position"]:
                move = board.parse_san(move_str)
                board.push(move)

            # Predict moves
            predictions = predict_moves(model, tokenizer, device, puzzle["moves_leading_to_position"], top_k=5)

            # Check if checkmate is in top 3
            top_3_moves = [move for move, _ in predictions[:3]]
            checkmate_in_top_3 = False

            print(f"\n   Model's top 3 predictions:")
            for i, (move, prob) in enumerate(predictions[:3], 1):
                is_legal = is_legal_move(board, move)
                is_mate = is_checkmate_move(board, move) if is_legal else False

                symbol = "🎯" if is_mate else ("✓" if is_legal else "✗")
                status = "MATE!" if is_mate else ("legal" if is_legal else "illegal")

                print(f"      {i}. {move:8s} ({prob:5.1%}) {symbol} {status}")

                if is_mate:
                    checkmate_in_top_3 = True

            if checkmate_in_top_3:
                checkmate_prioritized += 1
                print(f"\n   ✅ Checkmate in top 3")
            else:
                print(f"\n   ❌ Checkmate not prioritized")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mate_in_1_rate = (mate_in_1_found / mate_in_1_total * 100) if mate_in_1_total > 0 else 0
    priority_rate = (checkmate_prioritized / priority_total * 100) if priority_total > 0 else 0

    print(f"\n📊 Mate-in-1 Detection:")
    print(f"   Found: {mate_in_1_found}/{mate_in_1_total} ({mate_in_1_rate:.1f}%)")

    print(f"\n📊 Checkmate Priority:")
    print(f"   Prioritized: {checkmate_prioritized}/{priority_total} ({priority_rate:.1f}%)")

    print(f"\n🎯 Overall Checkmate Ability:")
    overall_score = (mate_in_1_rate + priority_rate) / 2
    print(f"   Score: {overall_score:.1f}%")

    if overall_score >= 40:
        print("\n   ✅ EXCELLENT - Strong checkmate understanding")
        verdict = "excellent"
    elif overall_score >= 25:
        print("\n   ✓ GOOD - Decent checkmate ability")
        verdict = "good"
    elif overall_score >= 15:
        print("\n   ⚠️  WEAK - Some checkmate awareness")
        verdict = "weak"
    else:
        print("\n   ❌ POOR - Minimal checkmate understanding")
        verdict = "poor"

    return {
        "mate_in_1_rate": mate_in_1_rate,
        "priority_rate": priority_rate,
        "overall_score": overall_score,
        "verdict": verdict,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt2-test-5k)")
    args = parser.parse_args()

    results = test_checkmate_ability(args.model)

    # Exit code based on verdict
    exit_codes = {"excellent": 0, "good": 0, "weak": 1, "poor": 2}
    sys.exit(exit_codes.get(results["verdict"], 2))
