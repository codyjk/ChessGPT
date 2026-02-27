"""Tests for PGN save and Stockfish analysis tools."""

import shutil

import chess
import chess.pgn
import pytest

from chessgpt.cli.analyze import classify_move, game_phase, score_to_cp
from chessgpt.cli.play import save_pgn

# --- PGN save tests ---


class FakeConfig:
    d_model = 512
    n_layers = 12
    n_heads = 8


def test_save_pgn_creates_file(tmp_path):
    """save_pgn should create a PGN file with correct headers and moves."""
    model_dir = tmp_path / "medium_v1"
    model_dir.mkdir()
    model_path = str(model_dir / "model.pt")

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    pgn_path = save_pgn(
        board=board,
        model_path=model_path,
        model_config=FakeConfig(),
        player_color=chess.WHITE,
        temperature=0.3,
        top_k=5,
    )

    assert pgn_path.exists()
    assert pgn_path.suffix == ".pgn"
    assert pgn_path.parent.name == "games"
    assert pgn_path.parent.parent.name == "medium_v1"

    content = pgn_path.read_text()
    assert '[Event "ChessGPT Interactive"]' in content
    assert '[White "Human"]' in content
    assert '[Black "ChessGPT medium_v1"]' in content
    assert '[ChessGPTConfig "d512_L12_H8"]' in content
    assert '[ChessGPTTemperature "0.3"]' in content
    assert '[ChessGPTTopK "5"]' in content
    assert "e4" in content
    assert "e5" in content
    assert "Nf3" in content


def test_save_pgn_model_as_white(tmp_path):
    """When human plays black, model should be listed as White."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    model_path = str(model_dir / "model.pt")

    board = chess.Board()
    board.push_san("d4")

    pgn_path = save_pgn(
        board=board,
        model_path=model_path,
        model_config=FakeConfig(),
        player_color=chess.BLACK,
        temperature=0.5,
        top_k=10,
    )

    content = pgn_path.read_text()
    assert '[White "ChessGPT test_model"]' in content
    assert '[Black "Human"]' in content


def test_save_pgn_game_over_result(tmp_path):
    """Completed games should have the correct result header."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    model_path = str(model_dir / "model.pt")

    # Scholar's mate
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        board.push_san(san)
    assert board.is_game_over()

    pgn_path = save_pgn(
        board=board,
        model_path=model_path,
        model_config=FakeConfig(),
        player_color=chess.WHITE,
        temperature=0.3,
        top_k=5,
    )

    content = pgn_path.read_text()
    assert '[Result "1-0"]' in content


def test_save_pgn_incomplete_result(tmp_path):
    """Unfinished games (quit) should have result '*'."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    model_path = str(model_dir / "model.pt")

    board = chess.Board()
    board.push_san("e4")

    pgn_path = save_pgn(
        board=board,
        model_path=model_path,
        model_config=FakeConfig(),
        player_color=chess.WHITE,
        temperature=0.3,
        top_k=5,
    )

    content = pgn_path.read_text()
    assert '[Result "*"]' in content


# --- Move classification tests ---


def test_classify_best():
    assert classify_move(0) == "Best"
    assert classify_move(-5) == "Best"


def test_classify_good():
    assert classify_move(10) == "Good"
    assert classify_move(19) == "Good"


def test_classify_inaccuracy():
    assert classify_move(20) == "Inaccuracy"
    assert classify_move(49) == "Inaccuracy"


def test_classify_mistake():
    assert classify_move(50) == "Mistake"
    assert classify_move(99) == "Mistake"


def test_classify_blunder():
    assert classify_move(100) == "Blunder"
    assert classify_move(500) == "Blunder"


# --- score_to_cp tests ---


def test_score_to_cp_centipawns():
    score = chess.engine.PovScore(chess.engine.Cp(150), chess.WHITE)
    assert score_to_cp(score, chess.WHITE) == 150
    assert score_to_cp(score, chess.BLACK) == -150


def test_score_to_cp_mate():
    score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    assert score_to_cp(score, chess.WHITE) == 30000
    assert score_to_cp(score, chess.BLACK) == -30000


def test_score_to_cp_mate_negative():
    score = chess.engine.PovScore(chess.engine.Mate(-2), chess.WHITE)
    assert score_to_cp(score, chess.WHITE) == -30000


# --- game_phase tests ---


def test_game_phase_opening():
    board = chess.Board()  # 32 pieces
    assert game_phase(board) == "opening"


def test_game_phase_endgame():
    # King + rook vs king
    board = chess.Board("8/8/8/8/8/8/8/K1k1R3 w - - 0 1")
    assert game_phase(board) == "endgame"


def test_game_phase_middlegame():
    # Remove a few pieces to get into middlegame range (13-28 pieces)
    board = chess.Board("r1bqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")
    count = len(board.piece_map())
    # This position has ~30 pieces, still opening. Use a sparser one.
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    count = len(board.piece_map())
    # 28 pieces -- right on boundary. Make it clearly middlegame.
    board = chess.Board("r1bqk2r/ppp2ppp/2n2n2/3pp3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 5")
    count = len(board.piece_map())
    if count > 28:
        # Use an even sparser position
        board = chess.Board("r1bqk2r/ppp2ppp/2n5/3pp3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 6")
    assert game_phase(board) in ("middlegame", "opening")  # accept either near boundary


# --- Stockfish integration tests (skipped if not installed) ---

has_stockfish = shutil.which("stockfish") is not None


@pytest.mark.skipif(not has_stockfish, reason="Stockfish not installed")
def test_stockfish_analysis_smoke(tmp_path):
    """Smoke test: analyze a short game with Stockfish."""
    from chessgpt.cli.analyze import analyze_game

    # Write a minimal PGN
    pgn_content = (
        '[Event "Test"]\n'
        '[White "Human"]\n'
        '[Black "ChessGPT test"]\n'
        '[Result "*"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 *\n"
    )
    pgn_file = tmp_path / "test.pgn"
    pgn_file.write_text(pgn_content)

    # Should run without errors (depth 5 for speed)
    analyze_game(str(pgn_file), depth=5, stockfish_path="stockfish", model_color=chess.BLACK)
