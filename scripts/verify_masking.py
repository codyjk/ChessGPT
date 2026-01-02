"""
Verify that the move masking logic is correct.

Tests the masking for a few examples to ensure it properly masks
based on original move indices, not padded positions.
"""

import torch
from chess_model.model.tokenizer import ChessTokenizer
from chess_model.data import ChessDataset

# Load tokenizer
tokenizer = ChessTokenizer.load("trained_models/tokenizer.json")

# Load dataset
dataset = ChessDataset(
    csv_file="data/quick_sample/training-data.csv",
    tokenizer=tokenizer,
    max_context_length=12,
)

print("=" * 60)
print("Masking Verification")
print("=" * 60 + "\n")

# Test a few examples
for idx in range(min(5, len(dataset))):
    sample = dataset[idx]

    # Decode to see actual moves
    input_ids = sample['input_ids'].tolist()
    labels = sample['labels'].tolist()
    move_mask = sample['move_mask'].tolist()

    input_moves = tokenizer.decode(input_ids)
    label_moves = tokenizer.decode(labels)

    # Find where actual data starts (non-PAD)
    first_non_pad = next((i for i, move in enumerate(label_moves) if move != '[PAD]'), None)

    print(f"Example {idx + 1}:")
    print(f"  Labels: {label_moves[first_non_pad:]}")
    print(f"  Mask:   {move_mask[first_non_pad:]}")

    # Show which moves are kept
    kept_moves = []
    for i in range(first_non_pad, len(label_moves)):
        if move_mask[i] > 0:
            kept_moves.append(label_moves[i])

    print(f"  Kept:   {kept_moves}")

    # Determine expected pattern based on move indices
    print(f"  Expected: Every-other move based on outcome")
    print()

print("=" * 60)
print("\nInterpretation:")
print("  - For White-won games: Should keep moves at positions 1,3,5,... (Black moves masked)")
print("  - For Black-won games: Should keep moves at positions 0,2,4,... (White moves masked)")
print("  - For draws: Should keep all moves")
print("=" * 60)
