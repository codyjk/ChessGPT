"""
Automated model-vs-engine play via UCI protocol.

Drives any UCI-compatible engine (Stockfish, RustChess, etc.) and pits it against
the ChessGPT model. Alternates colors across games and reports W/L/D + illegal rate.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chessgpt.inference.player import predict_next_move
from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer


@dataclass
class MatchResult:
    """Results from a single game."""

    model_color: chess.Color
    outcome: str  # "model_win", "engine_win", "draw", "model_forfeit"
    moves: list[str]
    num_moves: int
    illegal_moves: int
    pgn_path: Path | None = None


@dataclass
class MatchSummary:
    """Aggregate results across all games."""

    games: list[MatchResult] = field(default_factory=list)

    @property
    def model_wins(self) -> int:
        return sum(1 for g in self.games if g.outcome == "model_win")

    @property
    def engine_wins(self) -> int:
        return sum(1 for g in self.games if g.outcome in ("engine_win", "model_forfeit"))

    @property
    def draws(self) -> int:
        return sum(1 for g in self.games if g.outcome == "draw")

    @property
    def forfeits(self) -> int:
        return sum(1 for g in self.games if g.outcome == "model_forfeit")

    @property
    def total_illegal(self) -> int:
        return sum(g.illegal_moves for g in self.games)

    @property
    def total_moves(self) -> int:
        return sum(g.num_moves for g in self.games)

    @property
    def avg_game_length(self) -> float:
        return self.total_moves / max(len(self.games), 1)

    def __str__(self) -> str:
        n = len(self.games)
        lines = [
            f"Match results ({n} games):",
            f"  Model wins:  {self.model_wins}",
            f"  Engine wins: {self.engine_wins} ({self.forfeits} by forfeit)",
            f"  Draws:       {self.draws}",
            f"  Illegal moves: {self.total_illegal}/{self.total_moves}"
            f" ({self.total_illegal / max(self.total_moves, 1):.1%})",
            f"  Avg game length: {self.avg_game_length:.1f} moves",
        ]
        return "\n".join(lines)


def play_game(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    engine: chess.engine.SimpleEngine,
    model_color: chess.Color,
    engine_time: float = 0.5,
    temperature: float = 0.3,
    top_k: int = 5,
    retry_illegal: bool = False,
    max_moves: int = 300,
) -> MatchResult:
    """Play a single game between the model and a UCI engine.

    Args:
        model: The chess transformer model.
        tokenizer: The tokenizer for encoding/decoding moves.
        engine: An open UCI engine instance.
        model_color: Which color the model plays.
        engine_time: Time limit per engine move in seconds.
        temperature: Sampling temperature for the model.
        top_k: Top-k sampling for the model.
        retry_illegal: If True, try next-best move on illegal; if False, forfeit.
        max_moves: Maximum total moves before declaring a draw.

    Returns:
        MatchResult for this game.
    """
    board = chess.Board()
    move_history: list[str] = []
    illegal_count = 0

    while not board.is_game_over() and len(move_history) < max_moves:
        if board.turn == model_color:
            # Model's turn
            result = predict_next_move(
                model,
                tokenizer,
                move_history,
                temperature=temperature,
                top_k=top_k,
            )

            predicted = result["move"]
            try:
                board.push_san(predicted)
                move_history.append(predicted)
            except (ValueError, chess.InvalidMoveError):
                illegal_count += 1
                if retry_illegal:
                    # Try remaining top-k moves (skip the one already attempted)
                    pushed = False
                    for move_str, _ in result["top_k_moves"]:
                        if move_str == predicted:
                            continue
                        try:
                            board.push_san(move_str)
                            move_history.append(move_str)
                            pushed = True
                            break
                        except (ValueError, chess.InvalidMoveError):
                            illegal_count += 1
                    if not pushed:
                        return MatchResult(
                            model_color=model_color,
                            outcome="model_forfeit",
                            moves=move_history,
                            num_moves=len(move_history),
                            illegal_moves=illegal_count,
                        )
                else:
                    return MatchResult(
                        model_color=model_color,
                        outcome="model_forfeit",
                        moves=move_history,
                        num_moves=len(move_history),
                        illegal_moves=illegal_count,
                    )
        else:
            # Engine's turn
            engine_result = engine.play(board, chess.engine.Limit(time=engine_time))
            if engine_result.move is None:
                # Engine resigned or errored -- model wins
                return MatchResult(
                    model_color=model_color,
                    outcome="model_win",
                    moves=move_history,
                    num_moves=len(move_history),
                    illegal_moves=illegal_count,
                )
            san = board.san(engine_result.move)
            board.push(engine_result.move)
            move_history.append(san)

    # Determine outcome -- check checkmate first (handles edge case where
    # checkmate lands on exactly the max_moves-th ply)
    if board.is_checkmate():
        winner = not board.turn  # board.turn is the loser (it's their turn but they're mated)
        outcome = "model_win" if winner == model_color else "engine_win"
    else:
        outcome = "draw"

    return MatchResult(
        model_color=model_color,
        outcome=outcome,
        moves=move_history,
        num_moves=len(move_history),
        illegal_moves=illegal_count,
    )


def save_game_pgn(
    result: MatchResult,
    model_name: str,
    engine_name: str,
    output_dir: Path,
    game_num: int,
) -> Path:
    """Save a game result as PGN."""
    board = chess.Board()
    game = chess.pgn.Game()

    white = model_name if result.model_color == chess.WHITE else engine_name
    black = model_name if result.model_color == chess.BLACK else engine_name

    game.headers["Event"] = "ChessGPT Engine Match"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Round"] = str(game_num)

    # Replay moves onto the game node
    node = game
    for san in result.moves:
        try:
            move = board.parse_san(san)
            node = node.add_variation(move)
            board.push(move)
        except (ValueError, chess.InvalidMoveError):
            break

    # Set result
    if result.outcome == "model_win":
        pgn_result = "1-0" if result.model_color == chess.WHITE else "0-1"
    elif result.outcome in ("engine_win", "model_forfeit"):
        pgn_result = "0-1" if result.model_color == chess.WHITE else "1-0"
    else:
        pgn_result = "1/2-1/2"
    game.headers["Result"] = pgn_result

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"game_{game_num:03d}.pgn"
    path.write_text(str(game) + "\n")
    return path


def run_match(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    engine_path: str,
    num_games: int = 20,
    engine_time: float = 0.5,
    engine_elo: int | None = None,
    temperature: float = 0.3,
    top_k: int = 5,
    retry_illegal: bool = False,
    output_dir: Path | None = None,
    model_name: str = "ChessGPT",
) -> MatchSummary:
    """Run a match between the model and a UCI engine.

    Alternates colors each game. Saves PGNs if output_dir is provided.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    if engine_elo is not None:
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": engine_elo})
        except chess.engine.EngineError:
            print(f"Warning: engine does not support UCI_Elo, ignoring --engine-elo {engine_elo}")

    engine_name = Path(engine_path).stem
    summary = MatchSummary()

    try:
        for i in range(num_games):
            model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_str = "white" if model_color == chess.WHITE else "black"
            print(f"Game {i + 1}/{num_games} (model plays {color_str})...", end=" ", flush=True)

            result = play_game(
                model,
                tokenizer,
                engine,
                model_color=model_color,
                engine_time=engine_time,
                temperature=temperature,
                top_k=top_k,
                retry_illegal=retry_illegal,
            )

            if output_dir is not None:
                matches_dir = output_dir / "matches"
                result.pgn_path = save_game_pgn(
                    result,
                    model_name,
                    engine_name,
                    matches_dir,
                    i + 1,
                )

            summary.games.append(result)
            print(f"{result.outcome} ({result.num_moves} moves, {result.illegal_moves} illegal)")
    finally:
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    return summary
