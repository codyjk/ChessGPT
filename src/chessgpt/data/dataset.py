"""
Chess dataset for multi-task training.

Reads the enriched CSV format:
    moves,outcome,checkmate_move_idx,ply_count

Returns batches with:
- input_ids: tokenized, left-padded move sequence
- labels: shifted sequence (next-move targets)
- outcome: one-hot [white, draw, black]
- checkmate_available: 1 if context window includes checkmate position
- move_mask: outcome-based masking (winner's moves only for decisive games)
- checkmate_weight: elevated weight for the checkmate-delivering move

Uses mmap-based random access from v1 for memory efficiency on large files.
num_workers=0 recommended for DataLoader on macOS/MPS.
"""

import csv
import mmap

import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer,
        max_context_length: int = 60,
        checkmate_weight: float = 5.0,
    ):
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.checkmate_weight = checkmate_weight
        self.line_offsets: list[int] = []

        # Build line offset index for mmap-based random access.
        # Byte offsets allow seeking directly to any row without reading the whole file.
        with open(self.csv_file, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.line_offsets.append(0)
            while mm.readline():
                self.line_offsets.append(mm.tell())
            mm.close()

        # Remove trailing empty line offset
        self.line_offsets.pop()

    def __len__(self) -> int:
        return len(self.line_offsets) - 1  # subtract header

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Open file per-call so this is safe with DataLoader num_workers>0
        # (each worker gets its own file handle via fork/spawn).
        with open(self.csv_file) as f:
            f.seek(self.line_offsets[idx + 1])  # +1 to skip header
            line = f.readline().strip()

        row = next(csv.reader([line]))
        moves_str, outcome, checkmate_move_idx_str, ply_count_str = row

        moves = moves_str.split() if moves_str else []
        checkmate_move_idx = int(checkmate_move_idx_str)
        int(ply_count_str)  # validate field exists

        # Truncate to max_context_length (keep most recent moves)
        if len(moves) > self.max_context_length:
            offset = len(moves) - self.max_context_length
            moves = moves[-self.max_context_length :]
            # Adjust checkmate index for truncation
            if checkmate_move_idx >= 0:
                checkmate_move_idx -= offset
                if checkmate_move_idx < 0:
                    checkmate_move_idx = -1
        else:
            offset = 0

        # Split into input (all but last) and label (all but first)
        context = moves[:-1]
        labels_moves = moves[1:]

        input_ids = self.tokenizer.encode_and_pad(context, self.max_context_length)
        labels = self.tokenizer.encode_and_pad(labels_moves, self.max_context_length)

        num_actual = len(context)
        pad_len = self.max_context_length - num_actual

        # Outcome-based move masking: train on winner's moves only
        move_mask = torch.zeros(self.max_context_length, dtype=torch.float)
        for i in range(num_actual):
            padded_pos = pad_len + i
            # labels[padded_pos] corresponds to move at original index (i + 1 + offset)
            original_idx = i + 1 + offset
            if outcome == "1-0":
                if original_idx % 2 == 0:  # white moves (0-indexed even = white's 1st, 3rd, ...)
                    move_mask[padded_pos] = 1.0
            elif outcome == "0-1":
                if original_idx % 2 == 1:  # black moves
                    move_mask[padded_pos] = 1.0
            elif outcome == "1/2-1/2":
                move_mask[padded_pos] = 1.0

        # Checkmate weighting: elevate loss for the checkmate-delivering move
        weight_mask = torch.ones(self.max_context_length, dtype=torch.float)
        if checkmate_move_idx >= 0:
            # The checkmate move appears as a label at position (checkmate_move_idx - 1)
            # after accounting for the context/label shift
            cm_label_pos = checkmate_move_idx - 1
            if 0 <= cm_label_pos < num_actual:
                weight_mask[pad_len + cm_label_pos] = self.checkmate_weight

        # Checkmate available: 1 if the context window includes the checkmate move
        checkmate_available = 1.0 if checkmate_move_idx >= 0 else 0.0

        # Outcome one-hot
        outcome_label = torch.zeros(3, dtype=torch.float)
        if outcome == "1-0":
            outcome_label[0] = 1.0
        elif outcome == "0-1":
            outcome_label[2] = 1.0  # black wins → index 2
        elif outcome == "1/2-1/2":
            outcome_label[1] = 1.0  # draw → index 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "outcome": outcome_label,
            "checkmate_available": torch.tensor(checkmate_available, dtype=torch.float),
            "move_mask": move_mask,
            "checkmate_weight": weight_mask,
        }
