import argparse
import sys

import torch
import torch.nn.functional as F

from chess_model import get_device
from chess_model.models.transformer import ChessTransformer
from chess_model.utils.tokenizer import ChessTokenizer


def load_model(
    model_path, tokenizer_path, device, n_positions, n_embd, n_layer, n_head
):
    tokenizer = ChessTokenizer.load(tokenizer_path)
    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer


def preprocess_input(move_sequence, tokenizer, max_length=50):
    input_ids = tokenizer.encode(move_sequence.split())
    input_ids = input_ids[-max_length:]  # Keep only the last max_length tokens
    input_ids = [0] * (max_length - len(input_ids)) + input_ids  # Pad from the left
    return torch.tensor([input_ids], dtype=torch.long)


def predict_next_move(
    model, tokenizer, move_sequence, device, temperature=1.0, top_k=5
):
    input_ids = preprocess_input(move_sequence, tokenizer).to(device)

    with torch.no_grad():
        move_logits = model(input_ids)

    move_logits = move_logits / temperature
    move_probs = F.softmax(move_logits, dim=-1)
    move_probs = move_probs / move_probs.sum()

    top_k_probs, top_k_indices = torch.topk(move_probs, top_k)

    sampled_index = torch.multinomial(top_k_probs.squeeze(), 1).item()
    predicted_move_id = top_k_indices.squeeze()[sampled_index].item()
    predicted_move = tokenizer.decode([predicted_move_id])[0]

    return predicted_move, move_probs.squeeze()


def print_debug_info(tokenizer, move_probs, top_k=5):
    top_k_probs, top_k_indices = torch.topk(move_probs, top_k)
    print("\nTop {} candidate moves:".format(top_k))
    for prob, idx in zip(top_k_probs, top_k_indices):
        move = tokenizer.decode([idx.item()])[0]
        print(f"{move}: {prob.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Chess CLI using trained model")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--tokenizer", required=True, help="Path to the tokenizer")
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Player's color (default: white)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show debugging information"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top candidate moves to show (default: 5)",
    )

    # Add ChessTransformer hyperparameters
    parser.add_argument(
        "--n_positions", type=int, default=50, help="Number of positions (default: 50)"
    )
    parser.add_argument(
        "--n_embd", type=int, default=256, help="Embedding dimension (default: 256)"
    )
    parser.add_argument(
        "--n_layer", type=int, default=4, help="Number of layers (default: 4)"
    )
    parser.add_argument(
        "--n_head", type=int, default=4, help="Number of attention heads (default: 4)"
    )

    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model(
        args.model,
        args.tokenizer,
        device,
        args.n_positions,
        args.n_embd,
        args.n_layer,
        args.n_head,
    )

    moves = []
    player_turn = args.color == "white"

    print("Chess game started. Type your moves or 'quit' to exit.")
    print("Current position: <start>")

    while True:
        if player_turn:
            move = input("Your move: ").strip()
            if move.lower() == "quit":
                break
            moves.append(move)
        else:
            move_sequence = " ".join(moves)
            predicted_move, move_probs = predict_next_move(
                model, tokenizer, move_sequence, device, top_k=args.top_k
            )
            print(f"Model's move: {predicted_move}")
            moves.append(predicted_move)

            if args.debug:
                print(f"Number of moves played: {len(moves)}")
                print_debug_info(tokenizer, move_probs, args.top_k)

        print(f"Current position: {' '.join(moves)}")
        player_turn = not player_turn

    print("Game ended.")


if __name__ == "__main__":
    main()
