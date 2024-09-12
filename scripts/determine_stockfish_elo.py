import argparse
import os
import subprocess
import sys
import time

import chess
import chess.engine
import torch
from blessed import Terminal
from tqdm import tqdm

from chess_model.model import ChessTokenizer, ChessTransformer
from chess_model.util import get_device


def load_model(args):
    device = get_device()
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

    return model, tokenizer, device


def predict_move(model, tokenizer, move_sequence, args, device):
    input_ids = tokenizer.encode_and_pad(move_sequence, args.max_context_length)
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        move_logits = model(input_tensor)

    last_move_logits = move_logits[0, -1, :]
    predicted_move_id = torch.argmax(last_move_logits).item()
    predicted_move = tokenizer.decode([predicted_move_id])[0]
    return predicted_move


def get_unicode_piece(piece: chess.Piece) -> str:
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
    return piece_unicode.get(piece.symbol(), " ")


def render_board(term: Terminal, board: chess.Board, player_color: chess.Color) -> str:
    output = term.home + term.clear

    files = "abcdefgh"
    ranks = "87654321" if player_color == chess.WHITE else "12345678"

    output += "   " + " ".join(f"{f:^3}" for f in files) + "\n"
    output += "  ┌" + "───┬" * 7 + "───┐\n"

    for rank in ranks:
        output += f"{rank} │"
        for file in files:
            square = chess.parse_square(file + rank)
            piece = board.piece_at(square)
            if piece:
                output += f" {get_unicode_piece(piece)} │"
            else:
                output += f" {'·' if (ord(rank) + ord(file)) % 2 == 0 else ' '} │"
        output += f" {rank}\n"
        if rank != ranks[-1]:
            output += "  ├" + "───┼" * 7 + "───┤\n"

    output += "  └" + "───┴" * 7 + "───┘\n"
    output += "   " + " ".join(f"{f:^3}" for f in files) + "\n"

    return output


import chess


def coordinate_to_algebraic(board: chess.Board, move: chess.Move) -> str:
    return board.san(move)


def play_game(model, tokenizer, engine, args, device, model_color, term):
    board = chess.Board()
    move_sequence = []

    while not board.is_game_over():
        print(render_board(term, board, model_color))
        print(f"Moves played: {' '.join(move_sequence)}")

        if board.turn == model_color:
            print("Model is thinking...")
            for _ in range(100):  # Try up to 100 times to make a valid move
                predicted_move = predict_move(
                    model, tokenizer, move_sequence, args, device
                )
                try:
                    move = board.parse_san(predicted_move)
                    board.push(move)
                    move_sequence.append(predicted_move)
                    print(f"Model's move: {predicted_move}")
                    break
                except ValueError:
                    continue
            else:
                print("Model couldn't make a valid move")
                return 0  # Model couldn't make a valid move, count as a loss
        else:
            print("Stockfish is thinking...")
            result = engine.play(board, chess.engine.Limit(time=0.1))
            algebraic_move = coordinate_to_algebraic(board, result.move)
            board.push(result.move)
            move_sequence.append(algebraic_move)
            print(f"Stockfish's move: {algebraic_move}")

        time.sleep(0.1)  # Add a small delay to make the game progression visible

    print(render_board(term, board, model_color))
    print(f"Game over. Result: {board.result()}")
    time.sleep(0.5)  # Pause

    result = board.result()
    if result == "1-0":
        return 1 if model_color == chess.WHITE else 0
    elif result == "0-1":
        return 1 if model_color == chess.BLACK else 0
    else:
        return 0.5


def estimate_elo(model, tokenizer, engine, args, device, term):
    elo_levels = list(range(1320, 3001, 50))
    games_per_level = 10
    results = []

    for elo in tqdm(elo_levels, desc="Testing ELO levels"):
        engine.configure({"UCI_Elo": elo})
        score = 0
        print(f"\nTesting against Stockfish ELO {elo}")
        for game in range(games_per_level):
            print(f"\nGame {game + 1} as White")
            score += play_game(
                model, tokenizer, engine, args, device, chess.WHITE, term
            )
            print(f"\nGame {game + 1} as Black")
            score += play_game(
                model, tokenizer, engine, args, device, chess.BLACK, term
            )
        average_score = score / (2 * games_per_level)
        results.append((elo, average_score))
        print(
            f"Score against ELO {elo}: {score}/{2*games_per_level} (Average: {average_score:.2f})"
        )
        if average_score < 0.1:  # If winning less than 10%, stop testing higher levels
            break

    # Find the ELO level where the model scores closest to 0.5
    closest_elo = min(results, key=lambda x: abs(x[1] - 0.5))
    return closest_elo[0]


def find_stockfish():
    if sys.platform.startswith("win"):
        # On Windows, search for stockfish.exe in PATH
        try:
            result = subprocess.run(
                ["where", "stockfish"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip().split("\n")[0]
        except subprocess.CalledProcessError:
            return None
    else:
        # On Unix-like systems, use 'which' command
        try:
            result = subprocess.run(
                ["which", "stockfish"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Determine the ELO rating of a chess model using Stockfish"
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
        help="Maximum context length for the model",
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        required=True,
        help="Number of embeddings in the model",
    )
    parser.add_argument(
        "--num-layers", type=int, required=True, help="Number of layers in the model"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        required=True,
        help="Number of attention heads in the model",
    )
    parser.add_argument("--stockfish-path", help="Path to the Stockfish executable")
    args = parser.parse_args()

    if args.stockfish_path is None:
        args.stockfish_path = find_stockfish()
        if args.stockfish_path is None:
            print(
                "Error: Stockfish not found. Please install Stockfish and make sure it's in your PATH, or specify the path using --stockfish-path."
            )
            sys.exit(1)

    if not os.path.exists(args.stockfish_path):
        print(f"Error: Stockfish executable not found at {args.stockfish_path}")
        sys.exit(1)

    print(f"Using Stockfish at: {args.stockfish_path}")

    model, tokenizer, device = load_model(args)
    term = Terminal()

    try:
        with chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as engine:
            with term.fullscreen(), term.hidden_cursor():
                estimated_elo = estimate_elo(
                    model, tokenizer, engine, args, device, term
                )
    except chess.engine.EngineTerminatedError:
        print(
            "Error: Stockfish engine terminated unexpectedly. Please check if the correct Stockfish version is installed."
        )
        sys.exit(1)

    print(f"Estimated ELO rating of the model: {estimated_elo}")


if __name__ == "__main__":
    main()
