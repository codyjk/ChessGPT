"""Tests for engine match evaluation."""

from unittest.mock import MagicMock, patch

import chess
import chess.engine

from chessgpt.evaluation.engine_match import (
    MatchResult,
    MatchSummary,
    play_game,
    run_match,
    save_game_pgn,
)
from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer, TransformerConfig


def make_tiny_model() -> tuple[ChessTransformer, ChessTokenizer]:
    """Create a tiny model + tokenizer for testing."""
    tokenizer = ChessTokenizer()
    moves = ["e4", "e5", "d4", "d5", "Nf3", "Nc6", "Bc4", "Bc5", "Qh5", "Qxf7#"]
    for move in moves:
        tokenizer.move_to_id[move] = tokenizer.vocab_size
        tokenizer.id_to_move[tokenizer.vocab_size] = move
        tokenizer.vocab_size += 1

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=20,
    )
    model = ChessTransformer(config)
    return model, tokenizer


def test_match_result_fields():
    """MatchResult stores game data correctly."""
    result = MatchResult(
        model_color=chess.WHITE,
        outcome="model_win",
        moves=["e4", "e5", "Qh5"],
        num_moves=3,
        illegal_moves=0,
    )
    assert result.outcome == "model_win"
    assert result.num_moves == 3
    assert result.illegal_moves == 0
    assert result.model_color == chess.WHITE


def test_match_summary_aggregation():
    """MatchSummary correctly aggregates game results."""
    summary = MatchSummary(
        games=[
            MatchResult(chess.WHITE, "model_win", ["e4"], 10, 0),
            MatchResult(chess.BLACK, "engine_win", ["d4"], 20, 1),
            MatchResult(chess.WHITE, "draw", ["c4"], 30, 0),
            MatchResult(chess.BLACK, "model_forfeit", ["Nf3"], 5, 2),
        ]
    )

    assert summary.model_wins == 1
    assert summary.engine_wins == 2  # engine_win + model_forfeit
    assert summary.draws == 1
    assert summary.forfeits == 1
    assert summary.total_illegal == 3
    assert summary.total_moves == 65
    assert summary.avg_game_length == 65 / 4


def test_match_summary_str():
    """MatchSummary __str__ produces readable output."""
    summary = MatchSummary(
        games=[
            MatchResult(chess.WHITE, "model_win", [], 10, 0),
            MatchResult(chess.BLACK, "draw", [], 20, 1),
        ]
    )
    text = str(summary)
    assert "2 games" in text
    assert "Model wins:" in text
    assert "Engine wins:" in text


def test_play_game_forfeit_on_illegal():
    """Model forfeits when it plays an illegal move (retry_illegal=False)."""
    model, tokenizer = make_tiny_model()

    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)

    def mock_predict(model, tokenizer, history, temperature=0.3, top_k=5):
        return {
            "move": "ZZZZZ",  # Always illegal
            "top_k_moves": [("ZZZZZ", 0.9)],
            "value": [0.5, 0.3, 0.2],
            "checkmate_prob": 0.01,
        }

    with patch("chessgpt.evaluation.engine_match.predict_next_move", side_effect=mock_predict):
        result = play_game(
            model,
            tokenizer,
            mock_engine,
            model_color=chess.WHITE,
            retry_illegal=False,
            max_moves=10,
        )

    assert result.outcome == "model_forfeit"
    assert result.illegal_moves >= 1
    assert result.num_moves == 0  # Forfeited on first move, nothing pushed


def test_play_game_respects_max_moves():
    """Game ends as draw when max_moves is reached."""
    model, tokenizer = make_tiny_model()

    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)

    # We'll mock predict_next_move to always return a legal move
    moves_white = iter(["e4", "d4", "Nf3", "Bc4"])
    moves_black = [
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("d7d5"),
        chess.Move.from_uci("b8c6"),
        chess.Move.from_uci("f8c5"),
    ]
    black_iter = iter(moves_black)

    def mock_engine_play(board, limit):
        result = MagicMock()
        result.move = next(black_iter)
        return result

    mock_engine.play.side_effect = mock_engine_play

    def mock_predict(model, tokenizer, history, temperature=0.3, top_k=5):
        move = next(moves_white)
        return {
            "move": move,
            "top_k_moves": [(move, 0.9)],
            "value": [0.5, 0.3, 0.2],
            "checkmate_prob": 0.01,
        }

    with patch("chessgpt.evaluation.engine_match.predict_next_move", side_effect=mock_predict):
        result = play_game(
            model,
            tokenizer,
            mock_engine,
            model_color=chess.WHITE,
            max_moves=4,
        )

    assert result.outcome == "draw"
    assert result.num_moves == 4


def test_play_game_retry_illegal_finds_legal():
    """With retry_illegal=True, model retries top-k and finds a legal move."""
    model, tokenizer = make_tiny_model()

    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)
    mock_engine_result = MagicMock()
    mock_engine_result.move = chess.Move.from_uci("e7e5")
    mock_engine.play.return_value = mock_engine_result

    def mock_predict(model, tokenizer, history, temperature=0.3, top_k=5):
        return {
            "move": "ZZZZZ",  # Sampled move is illegal
            "top_k_moves": [("ZZZZZ", 0.5), ("e4", 0.3), ("d4", 0.2)],
            "value": [0.5, 0.3, 0.2],
            "checkmate_prob": 0.01,
        }

    with patch("chessgpt.evaluation.engine_match.predict_next_move", side_effect=mock_predict):
        result = play_game(
            model,
            tokenizer,
            mock_engine,
            model_color=chess.WHITE,
            retry_illegal=True,
            max_moves=2,  # Model plays e4 (retry), engine plays e5, then stop
        )

    assert result.outcome == "draw"  # Reached max_moves
    assert result.num_moves == 2
    assert result.illegal_moves == 1  # First attempt was illegal


def test_play_game_retry_illegal_all_fail():
    """With retry_illegal=True, model forfeits when all top-k moves are illegal."""
    model, tokenizer = make_tiny_model()

    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)

    def mock_predict(model, tokenizer, history, temperature=0.3, top_k=5):
        return {
            "move": "ZZZZZ",
            "top_k_moves": [("ZZZZZ", 0.5), ("YYYYY", 0.3), ("XXXXX", 0.2)],
            "value": [0.5, 0.3, 0.2],
            "checkmate_prob": 0.01,
        }

    with patch("chessgpt.evaluation.engine_match.predict_next_move", side_effect=mock_predict):
        result = play_game(
            model,
            tokenizer,
            mock_engine,
            model_color=chess.WHITE,
            retry_illegal=True,
            max_moves=10,
        )

    assert result.outcome == "model_forfeit"
    assert result.illegal_moves == 3  # ZZZZZ + YYYYY + XXXXX


def test_save_game_pgn(tmp_path):
    """PGN files are saved correctly."""
    result = MatchResult(
        model_color=chess.WHITE,
        outcome="model_win",
        moves=["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"],
        num_moves=7,
        illegal_moves=0,
    )

    path = save_game_pgn(result, "ChessGPT", "Stockfish", tmp_path, 1)

    assert path.exists()
    assert path.name == "game_001.pgn"

    content = path.read_text()
    assert "ChessGPT" in content
    assert "Stockfish" in content
    assert "1-0" in content  # model (white) won


def test_save_game_pgn_model_black_win(tmp_path):
    """PGN result is 0-1 when model plays black and wins."""
    result = MatchResult(
        model_color=chess.BLACK,
        outcome="model_win",
        moves=["f3", "e5", "g4", "Qh4#"],
        num_moves=4,
        illegal_moves=0,
    )

    path = save_game_pgn(result, "ChessGPT", "Stockfish", tmp_path, 2)
    content = path.read_text()
    assert "0-1" in content


def test_save_game_pgn_engine_win(tmp_path):
    """PGN result is correct when engine wins."""
    result = MatchResult(
        model_color=chess.WHITE,
        outcome="engine_win",
        moves=["f3", "e5", "g4", "Qh4#"],
        num_moves=4,
        illegal_moves=0,
    )

    path = save_game_pgn(result, "ChessGPT", "Stockfish", tmp_path, 3)
    content = path.read_text()
    assert "0-1" in content  # model was white and lost


def test_run_match_orchestration(tmp_path):
    """run_match alternates colors, saves PGNs, and cleans up the engine."""
    model, tokenizer = make_tiny_model()

    move_num = 0

    def mock_predict(model, tokenizer, history, temperature=0.3, top_k=5):
        nonlocal move_num
        # Alternate between a few legal opening moves
        moves = ["e4", "d4", "Nf3", "Bc4", "Nc3"]
        move = moves[move_num % len(moves)]
        move_num += 1
        return {
            "move": move,
            "top_k_moves": [(move, 0.9)],
            "value": [0.5, 0.3, 0.2],
            "checkmate_prob": 0.01,
        }

    mock_engine = MagicMock(spec=chess.engine.SimpleEngine)

    def mock_engine_play(board, limit):
        # Pick any legal move
        legal = list(board.legal_moves)
        result = MagicMock()
        result.move = legal[0] if legal else None
        return result

    mock_engine.play.side_effect = mock_engine_play
    mock_engine.configure = MagicMock()
    mock_engine.quit = MagicMock()

    with (
        patch("chessgpt.evaluation.engine_match.predict_next_move", side_effect=mock_predict),
        patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine),
    ):
        summary = run_match(
            model=model,
            tokenizer=tokenizer,
            engine_path="/fake/engine",
            num_games=4,
            engine_time=0.1,
            output_dir=tmp_path,
            model_name="TestModel",
        )

    # All 4 games completed
    assert len(summary.games) == 4

    # Colors alternate: game 1=white, 2=black, 3=white, 4=black
    assert summary.games[0].model_color == chess.WHITE
    assert summary.games[1].model_color == chess.BLACK
    assert summary.games[2].model_color == chess.WHITE
    assert summary.games[3].model_color == chess.BLACK

    # Engine was cleaned up
    mock_engine.quit.assert_called_once()

    # PGNs were saved
    matches_dir = tmp_path / "matches"
    assert matches_dir.exists()
    pgn_files = list(matches_dir.glob("*.pgn"))
    assert len(pgn_files) == 4
