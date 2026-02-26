"""
Chess move tokenizer.

Maps chess move strings (e.g. "Nf3", "O-O", "e4") to integer token IDs and back.
Uses a custom vocabulary fitted from training data — not BPE or any subword method,
since each chess move is an atomic unit. Includes [PAD] and [UNK] special tokens.

The vocabulary is small (~300 unique moves in standard chess), so each move gets
its own ID. This is fitted once from training data and saved/loaded as JSON.
"""

import json
from pathlib import Path

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


class ChessTokenizer:
    def __init__(self):
        self.move_to_id: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_move: dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.vocab_size: int = 2

    @property
    def pad_token_id(self) -> int:
        return 0

    def encode(self, moves: list[str]) -> list[int]:
        return [self.move_to_id.get(move, self.move_to_id[UNK_TOKEN]) for move in moves]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_move.get(token_id, UNK_TOKEN) for token_id in ids]

    def encode_and_pad(self, moves: list[str], max_context_length: int) -> list[int]:
        """Encode moves, truncate to max_context_length (keeping most recent), left-pad."""
        ids = self.encode(moves)
        ids = ids[-max_context_length:]
        pad_length = max_context_length - len(ids)
        return [0] * pad_length + ids

    def save(self, file_path: str | Path) -> None:
        state = {
            "move_to_id": self.move_to_id,
            "id_to_move": {int(k): v for k, v in self.id_to_move.items()},
            "vocab_size": self.vocab_size,
        }
        with open(file_path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "ChessTokenizer":
        with open(file_path) as f:
            state = json.load(f)
        tokenizer = cls()
        tokenizer.move_to_id = state["move_to_id"]
        tokenizer.id_to_move = {int(k): v for k, v in state["id_to_move"].items()}
        tokenizer.vocab_size = state["vocab_size"]
        return tokenizer

    @classmethod
    def fit(cls, csv_file: str | Path) -> "ChessTokenizer":
        """Build vocabulary from a training CSV file."""
        unique_moves: set[str] = set()
        with open(csv_file) as data:
            for i, row in enumerate(data):
                if i == 0:
                    continue  # skip header
                moves_str = row.strip().split(",")[0]
                for move in moves_str.split():
                    unique_moves.add(move)

        tokenizer = cls()
        for move in sorted(unique_moves):
            if move not in tokenizer.move_to_id:
                tokenizer.move_to_id[move] = tokenizer.vocab_size
                tokenizer.id_to_move[tokenizer.vocab_size] = move
                tokenizer.vocab_size += 1
        return tokenizer
