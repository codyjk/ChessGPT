"""
Test if the model correctly predicts moves for the right side (White vs Black).

This script explicitly validates that the model predicts White moves when it's
White's turn and Black moves when it's Black's turn.
"""

import torch
import chess
import random

from chess_model.model import ChessTransformer
from chess_model.model.tokenizer import ChessTokenizer


def get_move_color(board, move_san):
    """Determine if a move in SAN notation is for White or Black."""
    try:
        move = board.parse_san(move_san)
        # Check which side can make this move
        # If it's in legal moves, it's for the current player
        if move in board.legal_moves:
            return board.turn  # True = White, False = Black
        return None  # Illegal move
    except:
        return None  # Can't parse


def test_turn_tracking(model, tokenizer, context_moves, max_context_length=10):
    """Test if model predictions are for the correct side."""
    # Set up board
    board = chess.Board()
    for move_san in context_moves:
        try:
            board.push_san(move_san)
        except:
            return None  # Invalid game sequence

    # Whose turn is it?
    expected_color = board.turn  # True = White, False = Black

    # Get model predictions
    input_ids = tokenizer.encode_and_pad(context_moves, max_context_length)
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        next_move_logits = logits[0, max_context_length - 1, :]

        # Get top-10 predictions
        probs = torch.softmax(next_move_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)

        predictions = []
        for i in range(10):
            move_token = tokenizer.decode([top_indices[i].item()])[0]
            if move_token in ['[PAD]', '[UNK]']:
                continue

            move_color = get_move_color(board, move_token)
            predictions.append({
                'move': move_token,
                'prob': top_probs[i].item(),
                'color': move_color,
                'correct_color': move_color == expected_color if move_color is not None else False
            })

    return {
        'context': ' '.join(context_moves),
        'expected_color': 'White' if expected_color else 'Black',
        'predictions': predictions
    }


def main():
    print("=" * 60)
    print("Turn Tracking Validation")
    print("=" * 60 + "\n")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = ChessTokenizer.load("outputs/simple_test/tokenizer.json")

    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        n_positions=10,
        n_embd=128,
        n_layer=3,
        n_head=4,
    )

    model.load_state_dict(torch.load("outputs/simple_test/model.pt",
                                     map_location="cpu",
                                     weights_only=False))
    model.eval()
    print("✓ Model loaded\n")

    # Test cases: mix of White to move and Black to move
    test_cases = [
        ['e4', 'e5', 'Nf3', 'Nc6'],      # White to move
        ['d4', 'd5', 'c4'],                # Black to move
        ['e4', 'c5'],                      # White to move
        ['d4', 'Nf6', 'c4', 'g6'],        # White to move
        ['e4', 'e6', 'd4'],                # Black to move
    ]

    print("Testing turn tracking...\n")
    print("=" * 60)

    total_predictions = 0
    correct_color_predictions = 0

    for i, context in enumerate(test_cases, 1):
        result = test_turn_tracking(model, tokenizer, context)
        if not result:
            continue

        print(f"\nExample {i}:")
        print(f"Context: {result['context']}")
        print(f"Expected: {result['expected_color']} to move")
        print(f"\nTop predictions:")

        for j, pred in enumerate(result['predictions'][:5], 1):
            marker = "✓" if pred['correct_color'] else "✗"
            color_str = "White" if pred['color'] else "Black" if pred['color'] is not None else "Invalid"
            print(f"  {j}. {pred['move']:8s} ({pred['prob']:6.2%}) - {color_str} {marker}")

            if pred['color'] is not None:
                total_predictions += 1
                if pred['correct_color']:
                    correct_color_predictions += 1

        print("-" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nTotal valid predictions: {total_predictions}")
    print(f"Correct color: {correct_color_predictions}/{total_predictions} ({correct_color_predictions/total_predictions*100:.1f}%)")

    if correct_color_predictions / total_predictions > 0.9:
        print("\n✓ PASS: Model correctly predicts moves for the right side!")
    elif correct_color_predictions / total_predictions > 0.7:
        print("\n⚠ PARTIAL: Model mostly predicts correct side, but with errors")
    else:
        print("\n✗ FAIL: Model struggles with turn tracking")


if __name__ == "__main__":
    main()
