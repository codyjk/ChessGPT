import pytest

from chess_model import ChessDataset, ChessTokenizer


@pytest.fixture
def chess_tokenizer():
    tokenizer = ChessTokenizer()
    moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4"]
    tokenizer.fit(moves)
    return tokenizer


@pytest.fixture
def chess_dataset(tmp_path, chess_tokenizer):
    # Create a temporary CSV file
    csv_file = tmp_path / "test_chess_data.csv"
    csv_content = """context,next_move,is_checkmate,outcome
,e4,0,
e4,e5,0,
e4 e5 Nf3,Nc6,0,
e4 e5 Nf3 Nc6 Bb5,a6,0,
e4 e5 Nf3 Nc6 Bb5 a6,Ba4,0,
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6,O-O,0,1-0
"""
    csv_file.write_text(csv_content)

    return ChessDataset(str(csv_file), chess_tokenizer, max_length=10)


def test_chess_dataset_len(chess_dataset):
    assert len(chess_dataset) == 6  # 6 examples in the CSV file


def test_chess_dataset_getitem(chess_dataset):
    item = chess_dataset[2]

    assert "input_ids" in item
    assert "labels" in item
    assert "is_checkmate" in item
    assert "outcome" in item

    assert item["input_ids"].shape == (10,)  # max_length is 10
    assert item["labels"].shape == ()  # Single label
    assert item["is_checkmate"].shape == ()  # Single value
    assert item["outcome"].shape == (3,)  # One-hot encoding for 3 possible outcomes

    # Check if the input_ids are correct for the context "e4 e5 Nf3"
    expected_input = [0] * 7 + [2, 3, 4]  # 7 padding tokens + 3 move tokens
    assert item["input_ids"].tolist() == expected_input

    # Check if the label is correct for the next_move "Nc6"
    assert item["labels"].item() == chess_dataset.tokenizer.move_to_id["Nc6"]

    assert item["is_checkmate"].item() == 0.0
    assert item["outcome"].tolist() == [0.0, 0.0, 0.0]  # No outcome specified


def test_chess_dataset_padding(chess_dataset):
    item = chess_dataset[4]  # Second to last item in fixture

    # The tokenizer is initialized with `[UNK]` and `[PAD]` tokens, so any new tokens
    # passed in start at index 2. Therefore, `e4` is at index 2, `e5` at index 3, etc.
    expected_input = [0] * 4 + [2, 3, 4, 5, 6, 7]  # 4 padding tokens + 6 move tokens
    assert item["input_ids"].tolist() == expected_input


def test_chess_dataset_outcome(chess_dataset):
    item = chess_dataset[len(chess_dataset) - 1]  # The last item has the outcome
    assert item["outcome"].tolist() == [1.0, 0.0, 0.0]  # White win
