import csv
import mmap

import torch
from torch.utils.data import Dataset

"""
Reads a CSV file with chess game examples in the following format:

```
context,next_move,is_checkmate,outcome
,d4,0,
d4,e6,0,
d4 e6 Nf3,b6,0,
d4 e6 Nf3 b6,c4,0,
d4 e6 Nf3 b6 c4,Bb7,0,
d4 e6 Nf3 b6 c4 Bb7,Nc3,0,
d4 e6 Nf3 b6 c4 Bb7 Nc3,Bb4,0,
d4 e6 Nf3 b6 c4 Bb7 Nc3 Bb4,g3,0,
d4 e6 Nf3 b6 c4 Bb7 Nc3 Bb4 g3,f5,0,
...
```

Each example is composed of a context (a list of previous moves), a next move,
a boolean indicating if the game is a checkmate, and an outcome (1-0, 0-1, or
1/2-1/2).

The implementation does not load all data into memory at once, but rather
caches the line offsets of the CSV file. This allows for efficient random
access to the data while minimizing memory usage.
"""


class ChessDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=50):
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_offsets = []

        # Open the file and keep it open
        self.file = open(self.csv_file, "r")

        # Create an index of line offsets for random access
        with open(self.csv_file, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.line_offsets.append(0)
            while mm.readline():
                self.line_offsets.append(mm.tell())
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
        context, next_move, is_checkmate, outcome = row

        context = context.split() if context else []
        is_checkmate = float(is_checkmate)

        # Tokenize input (context)
        input_ids = self.tokenizer.encode(context)
        input_ids = input_ids[
            -self.max_length :
        ]  # Keep only the last max_length tokens
        input_ids = [0] * (
            self.max_length - len(input_ids)
        ) + input_ids  # Pad from the left

        # Create labels (next_move)
        labels = self.tokenizer.encode([next_move])[0]

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
        }

    def __del__(self):
        # Close the file when the dataset object is destroyed
        if hasattr(self, "file"):
            self.file.close()
