import argparse
import sys
from typing import Dict, List, Tuple

import chess
import torch
import torch.nn.functional as F
from blessed import Terminal

from chess_model.model import ChessTokenizer, ChessTransformer


def load_model(args: argparse.Namespace) -> Tuple[ChessTransformer, ChessTokenizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChessTokenizer.load(args.input_tokenizer_file)
    vocab_size = tokenizer.vocab_size

    model = ChessTransformer(
        vocab_size,
        args.max_context_length,
        args.num_embeddings,
        args.num_layers,
        args.num_heads,
    )
    model.load_state_dict(torch.load(args.input_model_file, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


def preprocess_input(
    move_sequence: List[str], tokenizer: ChessTokenizer, max_context_length: int
) -> torch.Tensor:
    input_ids = tokenizer.encode(move_sequence)
    input_ids = input_ids[-max_context_length:]
    input_ids = [0] * (max_context_length - len(input_ids)) + input_ids
    return torch.tensor(input_ids).unsqueeze(0)


def predict_next_move(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    move_sequence: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    input_ids = preprocess_input(move_sequence, tokenizer, args.max_context_length).to(
        device
    )

    with torch.no_grad():
        move_logits = model(input_ids)

    last_move_logits = move_logits[0, -1, :] / args.temperature
    move_probs = F.softmax(last_move_logits, dim=-1)
    move_probs = move_probs / move_probs.sum()

    top_k_probs, top_k_indices = torch.topk(move_probs, args.top_k)
    sampled_index = torch.multinomial(top_k_probs, 1).item()
    predicted_move_id = top_k_indices[sampled_index].item()
    predicted_move = tokenizer.decode([predicted_move_id])[0]
    return predicted_move, move_probs


def get_unicode_piece(piece: str) -> str:
    piece_unicode = {
        "R": "♜",
        "N": "♞",
        "B": "♝",
        "Q": "♛",
        "K": "♚",
        "P": "♟",
        "r": "♖",
        "n": "♘",
        "b": "♗",
        "q": "♕",
        "k": "♔",
        "p": "♙",
    }
    return piece_unicode.get(piece, " ")


def render_file_labels(files: str, player_color: chess.Color) -> str:
    files = files if player_color == chess.WHITE else files[::-1]
    return "   " + " ".join(f"{f:^3}" for f in files) + "\n"


def render_board_border(left: str, mid: str, joint: str, right: str) -> str:
    return f"  {left}{(mid * 3 + joint) * 7}{mid * 3}{right}\n"


def render_rank(
    rank: str, files: str, v_line: str, board: chess.Board, player_color: chess.Color
) -> str:
    output = f"{rank} {v_line}"
    for file in files if player_color == chess.WHITE else files[::-1]:
        square = chess.parse_square(file + rank)
        piece = board.piece_at(square)
        if piece:
            output += f" {get_unicode_piece(piece.symbol())} {v_line}"
        else:
            output += f" {'·' if (ord(rank) + ord(file)) % 2 == 0 else ' '} {v_line}"
    return f"{output} {rank}\n"


def render_board(term: Terminal, board: chess.Board, player_color: chess.Color) -> str:
    output = term.home + term.clear

    files = "abcdefgh"
    ranks = "87654321" if player_color == chess.WHITE else "12345678"

    # Unicode box-drawing characters
    h_line, v_line = "─", "│"
    tl_corner, tr_corner, bl_corner, br_corner = "┌", "┐", "└", "┘"
    t_joint, b_joint, l_joint, r_joint, cross = "┬", "┴", "├", "┤", "┼"

    output += render_file_labels(files, player_color)
    output += render_board_border(tl_corner, h_line, t_joint, tr_corner)

    for i, rank in enumerate(ranks):
        output += render_rank(rank, files, v_line, board, player_color)
        if i < 7:
            output += render_board_border(l_joint, h_line, cross, r_joint)

    output += render_board_border(bl_corner, h_line, b_joint, br_corner)
    output += render_file_labels(files, player_color)

    return output


def print_debug_info(
    prev_move_probs: torch.Tensor, tokenizer: ChessTokenizer, top_k: int
):
    if prev_move_probs is not None:
        top_k_moves = torch.topk(prev_move_probs, top_k)
        print("Top-k predictions:")
        for i, (prob, move_id) in enumerate(
            zip(top_k_moves.values, top_k_moves.indices)
        ):
            move = tokenizer.decode([move_id.item()])[0]
            print(f"{i+1}. {move}: {prob.item():.4f}")


def handle_player_move(board: chess.Board, move_sequence: List[str]) -> None:
    while True:
        move = input("Enter your move: ")
        try:
            board.push_san(move)
            move_sequence.append(move)
            break
        except ValueError:
            print("Invalid move. Try again.")


def handle_model_move(
    board: chess.Board,
    move_sequence: List[str],
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    valid_move_found = False
    attempts = 0
    top_k = args.top_k
    move_probs = None

    while not valid_move_found and attempts < 200:
        predicted_move, move_probs = predict_next_move(
            model, tokenizer, move_sequence, args, device
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

    if not valid_move_found:
        print("Failed to find a valid move. Exiting.")
        sys.exit(1)

    return move_probs


def print_game_result(board: chess.Board, move_sequence: List[str]):
    print(f"Final position: {' '.join(move_sequence)}")
    print("Game over. Result:", board.result())
    print("Exit by typing 'exit' or pressing Ctrl+C.")
    exit_input = input()
    if exit_input.lower() == "exit":
        sys.exit(0)


def play_game(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args)
    term = Terminal()
    board = chess.Board()
    move_sequence = []
    player_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK
    prev_move_probs = None

    with term.fullscreen(), term.hidden_cursor():
        while not board.is_game_over():
            print(render_board(term, board, player_color))
            print(f"Moves played: {' '.join(move_sequence)}")

            if args.debug:
                print_debug_info(prev_move_probs, tokenizer, args.top_k)

            if board.turn == player_color:
                handle_player_move(board, move_sequence)
                prev_move_probs = None
            else:
                prev_move_probs = handle_model_move(
                    board, move_sequence, model, tokenizer, args, device
                )

        print(render_board(term, board, player_color))
        print_game_result(board, move_sequence)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chess CLI for playing against a trained model"
    )
    parser.add_argument(
        "--input-model-file", required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--input-tokenizer-file", required=True, help="Path to the tokenizer file"
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        required=True,
        help="The maximum context length (number of moves) that the model was trained against.",
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        required=True,
        help="The number of embeddings that the model was trained with.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="The number of layers that the model was trained with.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        required=True,
        help="The number of heads that the model was trained with.",
    )
    parser.add_argument(
        "--color", default="white", choices=["white", "black"], help="Player's color"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sampling parameter")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def main():
    args = parse_arguments()
    play_game(args)


if __name__ == "__main__":
    main()
