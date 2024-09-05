import csv
import mmap

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ChessDataset(Dataset):
    """
    Reads a CSV file with chess game examples in the following format:

    ```
    context,is_checkmate,outcome
    d4 e6 Nf3 b6 c4 Bb7 Nc3 Bb4 g3 f5,0,
    ...
    ```

    Each example is composed of a context (a list of moves), a boolean
    indicating if the game is a checkmate, and an outcome (1-0, 0-1, or
    1/2-1/2).

    The implementation does not load all data into memory at once, but rather
    caches the line offsets of the CSV file. This allows for efficient random
    access to the data while minimizing memory usage.
    """

    def __init__(self, csv_file, tokenizer, max_context_length=50):
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.line_offsets = []
        self.file = open(self.csv_file, "r")

        # Create an index of line offsets for random access
        with open(self.csv_file, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            total_size = mm.size()
            self.line_offsets.append(0)

            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Indexing CSV file"
            ) as pbar:
                while mm.readline():
                    current_pos = mm.tell()
                    self.line_offsets.append(current_pos)
                    pbar.update(current_pos - pbar.n)

            mm.close()

        # Remove the last offset (empty line at the end of file)
        self.line_offsets.pop()

    def __len__(self):
        return len(self.line_offsets) - 1  # Subtract 1 to account for header

    def __getitem__(self, idx):
        # Add 1 to idx to skip the header
        self.file.seek(self.line_offsets[idx + 1])
        line = self.file.readline().strip()

        # Parse the CSV line
        row = next(csv.reader([line]))
        context, is_checkmate, outcome = row

        context = context.split() if context else []
        context, last_move = context[:-1], context[-1]
        is_checkmate = float(is_checkmate)

        input_ids = self.tokenizer.encode_and_pad(context, self.max_context_length)

        # Shift context to the left to create labels
        # The next move prediction for input_ids[n] is labels[n]
        labels = context[1:] + [last_move]
        labels = self.tokenizer.encode_and_pad(labels, self.max_context_length)

        # If white won, we want the model to learn from white's moves, not black's.
        # Conversely, if black won, we want the model to learn from black's moves.
        # For draws, we want the model to learn from both moves.
        # We will produce a mask that masks out the moves for the losing player,
        # and the model will learn from the remaining moves.
        move_mask = torch.ones(self.max_context_length, dtype=torch.float)

        if outcome == "1-0":  # White won
            # Mask out odd-indexed moves (Black's moves)
            move_mask[1::2] = 0
        elif outcome == "0-1":  # Black won
            # Mask out even-indexed moves (White's moves)
            move_mask[::2] = 0
        # For draws (1/2-1/2), keep all moves (mask stays 1)

        # If the context is shorter than max_context_length, zero-out that part of the mask
        if len(context) < self.max_context_length:
            move_mask[len(context) :] = 0

        # Convert outcome to one-hot encoding (as float)
        outcome_label = torch.zeros(3, dtype=torch.float)
        if outcome == "1-0":
            outcome_label[0] = 1.0
        elif outcome == "0-1":
            outcome_label[1] = 1.0
        elif outcome == "1/2-1/2":
            outcome_label[2] = 1.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_checkmate": torch.tensor(is_checkmate, dtype=torch.float),
            "outcome": outcome_label,
            "move_mask": move_mask,
        }

    def __del__(self):
        # Close the file when the dataset object is destroyed
        if hasattr(self, "file"):
            self.file.close()
