"""
CLI: Play against the chess model interactively.

Pure AI inference — no python-chess at runtime. The model plays on its own.
Board rendering uses python-chess for display only (showing the board to the human).
If the model outputs an illegal move, it's displayed as-is.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from chessgpt.cli.eval import load_model
from chessgpt.inference.player import predict_next_move
from chessgpt.training.trainer import get_device


def render_board(board: chess.Board, perspective: chess.Color) -> str:
    """Simple text-based board rendering."""
    files = "abcdefgh"
    ranks = "87654321" if perspective == chess.WHITE else "12345678"

    ordered_files = files if perspective == chess.WHITE else files[::-1]
    lines = []
    lines.append("   " + " ".join(f" {f} " for f in ordered_files))
    lines.append("  +" + "---+" * 8)

    piece_map = {
        "R": "\u265c",
        "N": "\u265e",
        "B": "\u265d",
        "Q": "\u265b",
        "K": "\u265a",
        "P": "\u265f",
        "r": "\u2656",
        "n": "\u2658",
        "b": "\u2657",
        "q": "\u2655",
        "k": "\u2654",
        "p": "\u2659",
    }

    for rank in ranks:
        row = f"{rank} |"
        for file in files if perspective == chess.WHITE else files[::-1]:
            sq = chess.parse_square(file + rank)
            piece = board.piece_at(sq)
            symbol = piece_map.get(piece.symbol(), " ") if piece else " "
            row += f" {symbol} |"
        lines.append(row)
        lines.append("  +" + "---+" * 8)

    lines.append("   " + " ".join(f" {f} " for f in ordered_files))
    return "\n".join(lines)


def save_pgn(
    board: chess.Board,
    model_path: str,
    model_config,
    player_color: chess.Color,
    temperature: float,
    top_k: int,
) -> Path:
    """Build a PGN from the board's move stack and save it to disk.

    Returns the path to the saved PGN file.
    """
    game = chess.pgn.Game.from_board(board)

    model_dir = Path(model_path).parent
    model_name = model_dir.name
    config_summary = f"d{model_config.d_model}_L{model_config.n_layers}_H{model_config.n_heads}"

    white_name = "Human" if player_color == chess.WHITE else f"ChessGPT {model_name}"
    black_name = "Human" if player_color == chess.BLACK else f"ChessGPT {model_name}"

    now = datetime.now()
    game.headers["Event"] = "ChessGPT Interactive"
    game.headers["Date"] = now.strftime("%Y.%m.%d")
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = board.result() if board.is_game_over() else "*"
    game.headers["ChessGPTModel"] = str(model_path)
    game.headers["ChessGPTConfig"] = config_summary
    game.headers["ChessGPTTemperature"] = str(temperature)
    game.headers["ChessGPTTopK"] = str(top_k)

    games_dir = model_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    filename = now.strftime("%Y%m%d_%H%M%S") + ".pgn"
    output_path = games_dir / filename

    output_path.write_text(str(game) + "\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Play chess against ChessGPT")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Your color")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sampling")
    parser.add_argument("--no-save", action="store_true", help="Don't save PGN after the game")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model(args.model, device)
    print(f"Loaded model (d_model={model.config.d_model}, {model.config.n_layers} layers)")
    print(f"Playing as {args.color}. Type 'quit' to exit.\n")

    board = chess.Board()
    move_history: list[str] = []
    player_color = chess.WHITE if args.color == "white" else chess.BLACK

    def _finish_game():
        """Save PGN and print final state."""
        print(render_board(board, player_color))
        print(f"\nMoves: {' '.join(move_history)}")
        if board.is_game_over():
            print(f"Game over! Result: {board.result()}")
        if not args.no_save and move_history:
            pgn_path = save_pgn(
                board,
                args.model,
                model.config,
                player_color,
                args.temperature,
                args.top_k,
            )
            print(f"\nPGN saved to: {pgn_path}")
            print(pgn_path.read_text())

    while not board.is_game_over():
        print(render_board(board, player_color))
        print(f"\nMoves: {' '.join(move_history)}")

        if board.turn == player_color:
            # Human's turn
            while True:
                move_str = input("\nYour move: ").strip()
                if move_str.lower() == "quit":
                    print("Thanks for playing!")
                    _finish_game()
                    sys.exit(0)
                try:
                    board.push_san(move_str)
                    move_history.append(move_str)
                    break
                except (ValueError, chess.InvalidMoveError):
                    print(f"Invalid move: {move_str}. Try again.")
        else:
            # Model's turn
            result = predict_next_move(
                model,
                tokenizer,
                move_history,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            predicted = result["move"]
            print(f"\nModel plays: {predicted}")

            # Show diagnostics
            value = result["value"]
            print(f"  Value: White {value[0]:.0%} | Draw {value[1]:.0%} | Black {value[2]:.0%}")
            print(f"  Checkmate prob: {result['checkmate_prob']:.1%}")
            top = ", ".join(f"{m} ({p:.1%})" for m, p in result["top_k_moves"])
            print(f"  Top predictions: {top}")

            # Try to push the move — if illegal, display and let human decide
            try:
                board.push_san(predicted)
                move_history.append(predicted)
            except (ValueError, chess.InvalidMoveError):
                print(f"  ** ILLEGAL MOVE: {predicted} **")
                print("  Enter the move to play for the model, or 'quit':")
                while True:
                    fix = input("  Override: ").strip()
                    if fix.lower() == "quit":
                        _finish_game()
                        sys.exit(0)
                    try:
                        board.push_san(fix)
                        move_history.append(fix)
                        break
                    except (ValueError, chess.InvalidMoveError):
                        print(f"  Invalid: {fix}. Try again.")

    _finish_game()
