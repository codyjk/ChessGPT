import pytest

from chess_model.data import ChessDataset
from chess_model.model import ChessTokenizer


@pytest.fixture
def training_data_file(tmp_path):
    """
    The tokenizer sorts the moves before assigning them ids.

    Mapping for this test (starting at 2 to account for [PAD] and [UNK]):
    Ba4 Bb5 Nc6 Nf3 Nf6 O-O a6 e4 e5
    2   3   4   5   6   7   8  9  10
    """
    path = tmp_path / "test_chess_data.csv"
    csv_content = """context,is_checkmate,outcome
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O,0,1-0
"""
    path.write_text(csv_content)
    return path


@pytest.fixture
def chess_dataset(training_data_file, chess_tokenizer):
    return ChessDataset(str(training_data_file), chess_tokenizer, max_context_length=10)


@pytest.fixture
def chess_tokenizer(training_data_file):
    return ChessTokenizer.fit(str(training_data_file))


def test_chess_dataset_len(chess_dataset):
    assert len(chess_dataset) == 1  # 1 examples in the CSV file


def test_chess_dataset_getitem(chess_dataset):
    item = chess_dataset[0]

    assert "input_ids" in item
    assert "labels" in item
    assert "is_checkmate" in item
    assert "outcome" in item

    assert item["input_ids"].shape == (10,)  # max_context_length is 10
    assert item["labels"].shape == (10,)  # same size as input_ids
    assert item["is_checkmate"].shape == ()  # Single value
    assert item["outcome"].shape == (3,)  # One-hot encoding for 3 possible outcomes

    # Check if the input_ids are correct for the context.
    # 2 padding tokens + 8 move tokens.
    # The input should not include the final move
    # The mapping of ids is described at the top in the test fixture.
    expected_input = [0] * 2 + [9, 10, 5, 4, 3, 8, 2, 6]
    assert item["input_ids"].tolist() == expected_input

    # Check if the label is correct for the next_move "Nc6"
    expected_labels = [0] * 2 + [10, 5, 4, 3, 8, 2, 6, 7]
    assert item["labels"].tolist() == expected_labels

    assert item["is_checkmate"].item() == 0.0
    # Outcome is specified as white win
    assert item["outcome"].tolist() == [1.0, 0.0, 0.0]
