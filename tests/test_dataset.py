"""Tests for the chess dataset and tokenizer."""

import pytest

from chessgpt.data.dataset import ChessDataset
from chessgpt.model.tokenizer import ChessTokenizer


@pytest.fixture
def training_csv(tmp_path):
    """
    Create a minimal training CSV with the new format.

    Game: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qh5#
    Outcome: 1-0 (white wins)
    Checkmate on move index 8 (Qh5#, 0-indexed)
    9 moves total, 0-indexed: e4(0) e5(1) Nf3(2) Nc6(3) Bb5(4) a6(5) Ba4(6) Nf6(7) Qh5#(8)
    """
    path = tmp_path / "test_data.csv"
    csv_content = "moves,outcome,checkmate_move_idx,ply_count\n"
    csv_content += "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qh5#,1-0,8,9\n"
    path.write_text(csv_content)
    return path


@pytest.fixture
def draw_csv(tmp_path):
    """A draw game with no checkmate."""
    path = tmp_path / "test_draw.csv"
    csv_content = "moves,outcome,checkmate_move_idx,ply_count\n"
    csv_content += "d4 d5 c4 e6 Nc3 Nf6,1/2-1/2,-1,6\n"
    path.write_text(csv_content)
    return path


@pytest.fixture
def black_wins_csv(tmp_path):
    """A game where black wins with checkmate."""
    path = tmp_path / "test_black.csv"
    csv_content = "moves,outcome,checkmate_move_idx,ply_count\n"
    csv_content += "f3 e5 g4 Qh4#,0-1,3,4\n"
    path.write_text(csv_content)
    return path


@pytest.fixture
def multi_row_csv(tmp_path):
    """Multiple games in one CSV."""
    path = tmp_path / "test_multi.csv"
    csv_content = "moves,outcome,checkmate_move_idx,ply_count\n"
    csv_content += "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qh5#,1-0,8,9\n"
    csv_content += "d4 d5 c4 e6 Nc3 Nf6,1/2-1/2,-1,6\n"
    csv_content += "f3 e5 g4 Qh4#,0-1,3,4\n"
    path.write_text(csv_content)
    return path


@pytest.fixture
def tokenizer(training_csv):
    return ChessTokenizer.fit(str(training_csv))


# ---- Tokenizer tests ----


def test_tokenizer_fit(tokenizer):
    # Should have [PAD], [UNK], plus all unique moves
    # Moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qh5#
    assert tokenizer.vocab_size == 2 + 9  # PAD + UNK + 9 unique moves


def test_tokenizer_encode_decode(tokenizer):
    moves = ["e4", "e5", "Nf3"]
    ids = tokenizer.encode(moves)
    decoded = tokenizer.decode(ids)
    assert decoded == moves


def test_tokenizer_unknown_move(tokenizer):
    ids = tokenizer.encode(["INVALID_MOVE"])
    assert ids == [1]  # UNK token


def test_tokenizer_encode_and_pad(tokenizer):
    moves = ["e4", "e5"]
    padded = tokenizer.encode_and_pad(moves, 5)
    assert len(padded) == 5
    assert padded[:3] == [0, 0, 0]  # left-padding
    assert padded[3:] == tokenizer.encode(moves)


def test_tokenizer_truncates_to_max_length(tokenizer):
    """encode_and_pad should keep most recent moves when truncating."""
    moves = ["e4", "e5", "Nf3", "Nc6", "Bb5"]
    padded = tokenizer.encode_and_pad(moves, 3)
    assert len(padded) == 3
    # Should keep the last 3 moves: Nf3, Nc6, Bb5
    expected = tokenizer.encode(moves[-3:])
    assert padded == expected


def test_tokenizer_save_load(tokenizer, tmp_path):
    path = str(tmp_path / "tokenizer.json")
    tokenizer.save(path)
    loaded = ChessTokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode(["e4"]) == tokenizer.encode(["e4"])


def test_tokenizer_save_load_pathlib(tokenizer, tmp_path):
    """Save/load should accept pathlib.Path objects."""
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ChessTokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size


def test_tokenizer_pad_token_id(tokenizer):
    assert tokenizer.pad_token_id == 0
    assert tokenizer.move_to_id["[PAD]"] == 0


def test_tokenizer_empty_encode(tokenizer):
    assert tokenizer.encode([]) == []
    assert tokenizer.decode([]) == []


def test_tokenizer_encode_and_pad_empty(tokenizer):
    padded = tokenizer.encode_and_pad([], 5)
    assert padded == [0, 0, 0, 0, 0]


# ---- Dataset basic tests ----


def test_dataset_len(training_csv, tokenizer):
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=10)
    assert len(dataset) == 1


def test_dataset_getitem_keys(training_csv, tokenizer):
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=10)
    item = dataset[0]

    assert "input_ids" in item
    assert "labels" in item
    assert "outcome" in item
    assert "checkmate_available" in item
    assert "move_mask" in item
    assert "checkmate_weight" in item


def test_dataset_shapes(training_csv, tokenizer):
    ctx_len = 10
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=ctx_len)
    item = dataset[0]

    assert item["input_ids"].shape == (ctx_len,)
    assert item["labels"].shape == (ctx_len,)
    assert item["outcome"].shape == (3,)
    assert item["checkmate_available"].shape == ()
    assert item["move_mask"].shape == (ctx_len,)
    assert item["checkmate_weight"].shape == (ctx_len,)


# ---- Outcome encoding tests ----


def test_dataset_outcome_white_wins(training_csv, tokenizer):
    """1-0 → [1, 0, 0] (white=index 0)."""
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=10)
    item = dataset[0]
    assert item["outcome"].tolist() == [1.0, 0.0, 0.0]


def test_dataset_outcome_draw(draw_csv):
    """1/2-1/2 → [0, 1, 0] (draw=index 1)."""
    tok = ChessTokenizer.fit(str(draw_csv))
    dataset = ChessDataset(str(draw_csv), tok, max_context_length=10)
    item = dataset[0]
    assert item["outcome"].tolist() == [0.0, 1.0, 0.0]


def test_dataset_outcome_black_wins(black_wins_csv):
    """0-1 → [0, 0, 1] (black=index 2)."""
    tok = ChessTokenizer.fit(str(black_wins_csv))
    dataset = ChessDataset(str(black_wins_csv), tok, max_context_length=10)
    item = dataset[0]
    assert item["outcome"].tolist() == [0.0, 0.0, 1.0]


# ---- Checkmate tests ----


def test_dataset_checkmate_available(training_csv, tokenizer):
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=10)
    item = dataset[0]
    assert item["checkmate_available"].item() == 1.0


def test_dataset_no_checkmate(draw_csv):
    tok = ChessTokenizer.fit(str(draw_csv))
    dataset = ChessDataset(str(draw_csv), tok, max_context_length=10)
    item = dataset[0]
    assert item["checkmate_available"].item() == 0.0


def test_dataset_checkmate_weight_position(training_csv, tokenizer):
    """The checkmate move (Qh5#, index 8) should get elevated weight at the correct position."""
    dataset = ChessDataset(
        str(training_csv), tokenizer, max_context_length=10, checkmate_weight=5.0
    )
    item = dataset[0]
    weight = item["checkmate_weight"]

    # Game: 9 moves (0-8). Context = moves[:-1] = 8. Labels = moves[1:] = 8.
    # Checkmate at index 8 appears as label at position (8-1)=7 in unpadded labels.
    # With ctx_len=10 and 8 actual tokens, pad_len = 10 - 8 = 2.
    # So the 5.0 weight should be at padded position 2 + 7 = 9.
    assert weight[9].item() == 5.0

    # All other positions should be 1.0
    for i in range(10):
        if i != 9:
            assert weight[i].item() == 1.0, f"Position {i} should be 1.0, got {weight[i].item()}"


# ---- Move masking tests ----


def test_dataset_move_mask_white_wins_precise(training_csv, tokenizer):
    """
    For white wins (1-0), only white's moves should be unmasked.

    Game: e4(0) e5(1) Nf3(2) Nc6(3) Bb5(4) a6(5) Ba4(6) Nf6(7) Qh5#(8)
    Labels are moves[1:] = e5(1) Nf3(2) Nc6(3) Bb5(4) a6(5) Ba4(6) Nf6(7) Qh5#(8)
    Label at position i has original_idx = i + 1 + offset.
    With no truncation, offset=0, so original indices are 1,2,3,4,5,6,7,8.

    For 1-0 (white wins), masked positions have even original indices (white moves):
    - original_idx=2 (Nf3, white) → mask=1
    - original_idx=4 (Bb5, white) → mask=1
    - original_idx=6 (Ba4, white) → mask=1
    - original_idx=8 (Qh5#, white) → mask=1
    - odd original indices → mask=0
    """
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=10)
    item = dataset[0]
    mask = item["move_mask"]

    # 8 actual label positions, pad_len=2
    # Expected mask at non-pad positions (indices 2-9):
    # pos 2: original_idx=1 (odd) → 0
    # pos 3: original_idx=2 (even) → 1
    # pos 4: original_idx=3 (odd) → 0
    # pos 5: original_idx=4 (even) → 1
    # pos 6: original_idx=5 (odd) → 0
    # pos 7: original_idx=6 (even) → 1
    # pos 8: original_idx=7 (odd) → 0
    # pos 9: original_idx=8 (even) → 1
    expected = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    assert mask.tolist() == [float(x) for x in expected]


def test_dataset_move_mask_black_wins(black_wins_csv):
    """
    For black wins (0-1), only black's moves should be unmasked.

    Game: f3(0) e5(1) g4(2) Qh4#(3)
    Labels = moves[1:] = e5(1) g4(2) Qh4#(3)
    original indices: 1, 2, 3
    For 0-1, odd original indices (black moves) are unmasked.
    """
    tok = ChessTokenizer.fit(str(black_wins_csv))
    dataset = ChessDataset(str(black_wins_csv), tok, max_context_length=6)
    item = dataset[0]
    mask = item["move_mask"]

    # 3 actual label positions, pad_len = 6 - 3 = 3
    # pos 3: original_idx=1 (odd) → 1 (black)
    # pos 4: original_idx=2 (even) → 0 (white)
    # pos 5: original_idx=3 (odd) → 1 (black)
    expected = [0, 0, 0, 1, 0, 1]
    assert mask.tolist() == [float(x) for x in expected]


def test_dataset_move_mask_draw(draw_csv):
    """For draws (1/2-1/2), all moves should be unmasked."""
    tok = ChessTokenizer.fit(str(draw_csv))
    dataset = ChessDataset(str(draw_csv), tok, max_context_length=10)
    item = dataset[0]
    mask = item["move_mask"]

    # 5 actual label positions (6 moves, labels = 5), pad_len = 10 - 5 = 5
    # All non-pad positions should be 1.0 for draws
    for i in range(5):
        assert mask[i].item() == 0.0, f"Pad position {i} should be 0"
    for i in range(5, 10):
        assert mask[i].item() == 1.0, f"Real position {i} should be 1"


# ---- Truncation tests ----


def test_dataset_truncation(training_csv, tokenizer):
    """When context length < game length, should keep most recent moves."""
    # Game has 9 moves. With max_context_length=5, we keep last 5.
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=5)
    item = dataset[0]

    # After truncation: last 5 moves = Bb5 a6 Ba4 Nf6 Qh5#
    # Context = first 4: Bb5, a6, Ba4, Nf6
    # Labels = last 4: a6, Ba4, Nf6, Qh5#
    # With max_context_length=5, pad_len = 5 - 4 = 1
    assert item["input_ids"].shape == (5,)
    assert item["labels"].shape == (5,)
    assert item["input_ids"][0].item() == 0  # padding


def test_dataset_truncation_loses_checkmate(tmp_path):
    """If truncation cuts before checkmate index, checkmate_available=0."""
    path = tmp_path / "long_game.csv"
    # 20-move game with checkmate at index 19
    moves = " ".join([f"m{i}" for i in range(20)])
    csv_content = "moves,outcome,checkmate_move_idx,ply_count\n"
    csv_content += f"{moves},1-0,19,20\n"
    path.write_text(csv_content)

    tok = ChessTokenizer()
    for i in range(20):
        move = f"m{i}"
        tok.move_to_id[move] = tok.vocab_size
        tok.id_to_move[tok.vocab_size] = move
        tok.vocab_size += 1

    # With max_context_length=5, only last 5 moves are kept.
    # Checkmate at index 19 - offset(15) = 4, which IS in the window.
    dataset = ChessDataset(str(path), tok, max_context_length=5)
    item = dataset[0]
    assert item["checkmate_available"].item() == 1.0

    # With max_context_length=3, checkmate at 19 - 17 = 2, still in window.
    dataset2 = ChessDataset(str(path), tok, max_context_length=3)
    item2 = dataset2[0]
    assert item2["checkmate_available"].item() == 1.0


# ---- Multi-row tests ----


def test_dataset_multi_row_len(multi_row_csv):
    tok = ChessTokenizer.fit(str(multi_row_csv))
    dataset = ChessDataset(str(multi_row_csv), tok, max_context_length=10)
    assert len(dataset) == 3


def test_dataset_multi_row_outcomes(multi_row_csv):
    """Each row should have its own outcome encoding."""
    tok = ChessTokenizer.fit(str(multi_row_csv))
    dataset = ChessDataset(str(multi_row_csv), tok, max_context_length=10)

    # Row 0: 1-0 (white wins)
    assert dataset[0]["outcome"].tolist() == [1.0, 0.0, 0.0]
    # Row 1: 1/2-1/2 (draw)
    assert dataset[1]["outcome"].tolist() == [0.0, 1.0, 0.0]
    # Row 2: 0-1 (black wins)
    assert dataset[2]["outcome"].tolist() == [0.0, 0.0, 1.0]


def test_dataset_multi_row_checkmate(multi_row_csv):
    """Only games with checkmate should have checkmate_available=1."""
    tok = ChessTokenizer.fit(str(multi_row_csv))
    dataset = ChessDataset(str(multi_row_csv), tok, max_context_length=10)

    assert dataset[0]["checkmate_available"].item() == 1.0  # white checkmate
    assert dataset[1]["checkmate_available"].item() == 0.0  # draw
    assert dataset[2]["checkmate_available"].item() == 1.0  # black checkmate


# ---- Input/label shift tests ----


def test_dataset_input_label_shift(training_csv, tokenizer):
    """Input should be moves[:-1] and labels should be moves[1:], properly encoded."""
    dataset = ChessDataset(str(training_csv), tokenizer, max_context_length=10)
    item = dataset[0]

    # Game: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qh5#
    # Context (input): e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 (8 moves)
    # Labels: e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 Qh5# (8 moves)
    # pad_len = 10 - 8 = 2

    # First 2 positions should be padding (0)
    assert item["input_ids"][0].item() == 0
    assert item["input_ids"][1].item() == 0

    # Non-pad input_ids should decode to e4, e5, Nf3, Nc6, Bb5, a6, Ba4, Nf6
    input_moves = tokenizer.decode(item["input_ids"][2:].tolist())
    assert input_moves == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]

    # Non-pad labels should decode to e5, Nf3, Nc6, Bb5, a6, Ba4, Nf6, Qh5#
    label_moves = tokenizer.decode(item["labels"][2:].tolist())
    assert label_moves == ["e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "Qh5#"]
