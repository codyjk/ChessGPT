"""CLI: Analyze a saved PGN with Stockfish to evaluate move quality."""

import argparse
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

# Centipawn thresholds for move classification
INACCURACY_CP = 20
MISTAKE_CP = 50
BLUNDER_CP = 100


def classify_move(cp_loss: int) -> str:
    """Classify a move by centipawn loss relative to the engine's best."""
    if cp_loss <= 0:
        return "Best"
    elif cp_loss < INACCURACY_CP:
        return "Good"
    elif cp_loss < MISTAKE_CP:
        return "Inaccuracy"
    elif cp_loss < BLUNDER_CP:
        return "Mistake"
    else:
        return "Blunder"


def score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int | None:
    """Convert a PovScore to centipawns from the given side's perspective.

    Returns None for mate scores (handled separately).
    """
    relative = score.pov(turn)
    cp = relative.score()
    if cp is not None:
        return cp
    # Mate score -- convert to large cp value for comparison
    mate = relative.mate()
    if mate is not None:
        return 30000 if mate > 0 else -30000
    return None


def game_phase(board: chess.Board) -> str:
    """Classify position as opening, middlegame, or endgame."""
    piece_count = len(board.piece_map())
    if piece_count > 28:
        return "opening"
    elif piece_count > 12:
        return "middlegame"
    else:
        return "endgame"


def analyze_game(pgn_path: str, depth: int, stockfish_path: str, model_color: chess.Color | None):
    """Run Stockfish analysis on each move and print a quality report."""
    path = Path(pgn_path)
    if not path.exists():
        print(f"Error: PGN file not found: {pgn_path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        game = chess.pgn.read_game(f)
    if game is None:
        print(f"Error: Could not parse PGN from {pgn_path}", file=sys.stderr)
        sys.exit(1)

    # Determine which side is the model
    if model_color is None:
        # Auto-detect from ChessGPT headers
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        if "ChessGPT" in black:
            model_color = chess.BLACK
        elif "ChessGPT" in white:
            model_color = chess.WHITE
        else:
            print("Warning: Could not detect model color from headers, analyzing both sides.")

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except FileNotFoundError:
        print(
            f"Error: Stockfish not found at '{stockfish_path}'. "
            "Install Stockfish or pass --stockfish /path/to/stockfish",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Analyzing: {pgn_path}")
    print(f"Depth: {depth}")
    print(f"White: {game.headers.get('White', '?')}")
    print(f"Black: {game.headers.get('Black', '?')}")
    if model_color is not None:
        print(f"Model plays: {'White' if model_color == chess.WHITE else 'Black'}")
    print()

    # Table header
    header = (
        f"{'#':>3}  {'Move':<10} {'Side':<6} {'Eval':>7}  "
        f"{'Best':>10}  {'CP Loss':>7}  {'Class':<11} {'Phase'}"
    )
    print(header)
    print("-" * 75)

    board = game.board()
    moves = list(game.mainline_moves())
    move_stats: list[dict] = []

    for i, move in enumerate(moves):
        turn = board.turn
        move_num = board.fullmove_number
        san = board.san(move)
        phase = game_phase(board)
        is_model = model_color is not None and turn == model_color

        # Evaluate position before the move
        info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = info_before.get("pv", [None])[0]
        best_san = board.san(best_move) if best_move else "?"
        score_before = info_before["score"]

        # Play the actual move
        board.push(move)

        # Evaluate position after
        info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_after = info_after["score"]

        # Compute centipawn loss from the moving side's perspective
        cp_before = score_to_cp(score_before, turn)
        cp_after = score_to_cp(score_after, turn)

        if cp_before is not None and cp_after is not None:
            cp_loss = max(0, cp_before - cp_after)
        else:
            cp_loss = 0

        classification = classify_move(cp_loss)

        # Format eval for display (from white's perspective)
        eval_cp = score_to_cp(score_after, chess.WHITE)
        eval_str = f"{eval_cp / 100:+.2f}" if eval_cp is not None else "?"

        side_str = "W" if turn == chess.WHITE else "B"
        if is_model:
            side_str += "*"
        move_label = f"{move_num}.{san}" if turn == chess.WHITE else f"{move_num}...{san}"

        print(
            f"{i + 1:>3}  {move_label:<10} {side_str:<6} {eval_str:>7}  "
            f"{best_san:>10}  {cp_loss:>5}cp  {classification:<11} {phase}"
        )

        move_stats.append(
            {
                "is_model": is_model,
                "cp_loss": cp_loss,
                "classification": classification,
                "phase": phase,
            }
        )

    engine.quit()

    # Summary
    model_moves = [m for m in move_stats if m["is_model"]]
    if not model_moves:
        # If no model color detected, summarize all moves
        model_moves = move_stats

    print(f"\n{'=' * 40}")
    print(f"Model moves: {len(model_moves)}")

    if model_moves:
        classifications = [m["classification"] for m in model_moves]
        total = len(classifications)
        for label in ["Best", "Good", "Inaccuracy", "Mistake", "Blunder"]:
            count = classifications.count(label)
            print(f"  {label:<12} {count:>3}  ({count / total:.0%})")

        avg_cp = sum(m["cp_loss"] for m in model_moves) / len(model_moves)
        print(f"\n  Avg CP loss: {avg_cp:.1f}")

        # Per-phase breakdown
        phases = ["opening", "middlegame", "endgame"]
        for phase in phases:
            phase_moves = [m for m in model_moves if m["phase"] == phase]
            if phase_moves:
                phase_avg = sum(m["cp_loss"] for m in phase_moves) / len(phase_moves)
                phase_blunders = sum(1 for m in phase_moves if m["classification"] == "Blunder")
                print(f"  {phase:<12} avg CP loss: {phase_avg:.1f}  blunders: {phase_blunders}")


def main():
    parser = argparse.ArgumentParser(description="Analyze a ChessGPT game with Stockfish")
    parser.add_argument("--pgn", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish search depth")
    parser.add_argument(
        "--stockfish", type=str, default="stockfish", help="Path to Stockfish binary"
    )
    parser.add_argument(
        "--model-color",
        choices=["white", "black"],
        default=None,
        help="Which side is the model (auto-detected from PGN headers if omitted)",
    )
    args = parser.parse_args()

    color = None
    if args.model_color == "white":
        color = chess.WHITE
    elif args.model_color == "black":
        color = chess.BLACK

    analyze_game(args.pgn, args.depth, args.stockfish, color)
