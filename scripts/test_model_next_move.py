"""
Test model by predicting the last move in actual training sequences.

This tests if the model can complete sequences it was trained on.
"""

import torch
import csv
import random

from chess_model.model import ChessTransformer
from chess_model.model.tokenizer import ChessTokenizer


def test_completion(model, tokenizer, moves, max_context_length=10):
    """Test if model can predict the last move given the first N-1 moves."""
    if len(moves) < 2:
        return None

    # Split into context and target
    context = moves[:-1]
    target = moves[-1]

    # Encode
    input_ids = tokenizer.encode_and_pad(context, max_context_length)
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        # Get logits for next move prediction
        # With left-padding, the last actual move is at position max_context_length - 1
        # (the rightmost position in the padded sequence)
        next_move_logits = logits[0, max_context_length - 1, :]

        # Get top-5 predictions
        top_k = 5
        probs = torch.softmax(next_move_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)

        # Decode
        predictions = []
        for i in range(top_k):
            move_token = tokenizer.decode([top_indices[i].item()])[0]
            prob = top_probs[i].item()
            predictions.append((move_token, prob))

        # Check if target is in top-k
        target_rank = None
        for i, (move, _) in enumerate(predictions, 1):
            if move == target:
                target_rank = i
                break

    return {
        'context': ' '.join(context),
        'target': target,
        'predictions': predictions,
        'target_rank': target_rank,
        'top1_correct': predictions[0][0] == target if predictions else False
    }


def main():
    """Test model on training data completions."""

    print("=" * 60)
    print("Testing Model: Next Move Prediction")
    print("=" * 60 + "\n")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = ChessTokenizer.load("outputs/simple_test/tokenizer.json")

    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        n_positions=10,  # Match training config
        n_embd=128,      # Match training config
        n_layer=3,       # Match training config
        n_head=4,        # Match training config
    )

    model.load_state_dict(torch.load("outputs/simple_test/model.pt",
                                     map_location="cpu",
                                     weights_only=False))
    model.eval()

    print(f"Model loaded (vocab_size: {tokenizer.vocab_size})\n")

    # Load some examples from validation data
    print("Loading validation examples...")
    examples = []
    with open("data/quick_sample/validation-data.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            moves = row['context'].split()
            if len(moves) >= 2:  # Need at least 2 moves
                examples.append(moves)

    # Sample 10 random examples
    test_examples = random.sample(examples, min(10, len(examples)))
    print(f"Testing on {len(test_examples)} validation examples\n")

    # Test each example
    print("=" * 60)
    results = []
    max_context_length = 10  # Match training config
    for i, moves in enumerate(test_examples, 1):
        result = test_completion(model, tokenizer, moves, max_context_length)
        if result:
            results.append(result)

            print(f"\nExample {i}:")
            print(f"Context: {result['context']}")
            print(f"Target:  {result['target']}")
            print(f"\nPredictions:")
            for j, (move, prob) in enumerate(result['predictions'], 1):
                marker = "✓" if move == result['target'] else " "
                print(f"  {j}. {move:8s} ({prob:6.2%}) {marker}")

            if result['target_rank']:
                print(f"\n→ Target found at rank {result['target_rank']}")
            else:
                print(f"\n→ Target not in top-5")

            print("-" * 60)

    # Statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    top1_correct = sum(1 for r in results if r['top1_correct'])
    in_top5 = sum(1 for r in results if r['target_rank'] is not None)

    print(f"\nTotal examples tested: {len(results)}")
    print(f"Top-1 accuracy:        {top1_correct}/{len(results)} ({top1_correct/len(results)*100:.1f}%)")
    print(f"Top-5 accuracy:        {in_top5}/{len(results)} ({in_top5/len(results)*100:.1f}%)")

    # Check for [PAD] predictions
    pad_predictions = sum(1 for r in results if r['predictions'][0][0] == '[PAD]')
    print(f"\n[PAD] as top prediction: {pad_predictions}/{len(results)} ({pad_predictions/len(results)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("\nWith only 3K training examples and 2 epochs:")
    print("  • >50% top-1 accuracy = model is learning patterns")
    print("  • 30-50% top-1 accuracy = weak learning, needs more data/epochs")
    print("  • <30% top-1 accuracy = model struggling, likely outputting [PAD]")
    print("\nTo improve:")
    print("  • Use more training data (10K-100K games)")
    print("  • Train for more epochs (5-10)")
    print("  • Use larger model or better hyperparameters")


if __name__ == "__main__":
    main()
