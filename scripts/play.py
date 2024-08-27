import argparse
import sys

import chess
import torch
import torch.nn.functional as F
from blessed import Terminal

from chess_model.models.transformer import ChessTransformer
from chess_model.utils.tokenizer import ChessTokenizer


def load_model(
    model_path, tokenizer_path, n_positions, n_embd, n_layer, n_head, device
):
    tokenizer = ChessTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size

    model = ChessTransformer(vocab_size, n_positions, n_embd, n_layer, n_head)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


def preprocess_input(move_sequence, tokenizer, max_length):
    input_ids = tokenizer.encode(move_sequence)
    input_ids = input_ids[-max_length:]
    input_ids = [0] * (max_length - len(input_ids)) + input_ids
    return torch.tensor(input_ids).unsqueeze(0)


def predict_next_move(model, tokenizer, move_sequence, device, top_k=5):
    input_ids = preprocess_input(move_sequence, tokenizer, model.config.n_positions)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        move_logits = model(input_ids)

    move_probs = F.softmax(move_logits, dim=-1)
    move_probs = move_probs / move_probs.sum()

    top_k_probs, top_k_indices = torch.topk(move_probs, top_k)
    sampled_index = torch.multinomial(top_k_probs.squeeze(), 1).item()
    predicted_move_id = top_k_indices.squeeze()[sampled_index].item()
    predicted_move = tokenizer.decode([predicted_move_id])[0]
    return predicted_move, move_probs.squeeze()


def render_board(term, board, player_color):
    output = term.home + term.clear
    board_str = str(board)
    if player_color == chess.BLACK:
        board_str = "\n".join(reversed(board_str.split("\n")))
    output += term.white_on_black(board_str)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Chess CLI for playing against a trained model"
    )
    parser.add_argument("--model", required=True, help="Path to the trained model file")
    parser.add_argument("--tokenizer", required=True, help="Path to the tokenizer file")
    parser.add_argument(
        "--n-positions", type=int, default=50, help="Number of positions"
    )
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--n-head", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--color", default="white", choices=["white", "black"], help="Player's color"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sampling parameter")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(
        args.model,
        args.tokenizer,
        args.n_positions,
        args.n_embd,
        args.n_layer,
        args.n_head,
        device,
    )

    term = Terminal()
    board = chess.Board()
    move_sequence = []
    player_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK

    with term.fullscreen(), term.hidden_cursor():
        while not board.is_game_over():
            print(render_board(term, board, player_color))
            print(f"Moves played: {' '.join(move_sequence)}")

            if board.turn == player_color:
                move = input("Enter your move: ")
                try:
                    board.push_san(move)
                    move_sequence.append(move)
                except ValueError:
                    print("Invalid move. Try again.")
                    continue
            else:
                valid_move_found = False
                attempts = 0
                top_k = args.top_k

                while not valid_move_found and attempts < 200:
                    predicted_move, move_probs = predict_next_move(
                        model, tokenizer, move_sequence, device, top_k
                    )
                    try:
                        board.push_san(predicted_move)
                        move_sequence.append(predicted_move)
                        valid_move_found = True
                        print(f"Model's move: {predicted_move}")
                    except ValueError:
                        attempts += 1
                        if attempts % 10 == 0:
                            top_k *= 2

                    if args.debug:
                        top_k_moves = torch.topk(move_probs, args.top_k)
                        print("Top-k predictions:")
                        for i, (prob, move_id) in enumerate(
                            zip(top_k_moves.values, top_k_moves.indices)
                        ):
                            move = tokenizer.decode([move_id.item()])[0]
                            print(f"{i+1}. {move}: {prob.item():.4f}")

                if not valid_move_found:
                    print("Failed to find a valid move. Exiting.")
                    sys.exit(1)

        print(render_board(term, board, player_color))
        print(f"Final position: {' '.join(move_sequence)}")
        print("Game over. Result:", board.result())


if __name__ == "__main__":
    main()
