import json
import mmap
import os

from tqdm import tqdm

from .count_lines import count_lines_fast

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


class ChessTokenizer:
    """
    A tokenizer for chess moves.
    """

    def __init__(self):
        self.move_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_move = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.vocab_size = 2

    def encode(self, moves):
        return [self.move_to_id.get(move, self.move_to_id[UNK_TOKEN]) for move in moves]

    def decode(self, ids):
        return [self.id_to_move.get(id, UNK_TOKEN) for id in ids]

    def save(self, file_path):
        """Save the tokenizer state to a JSON file."""
        state = {
            "move_to_id": self.move_to_id,
            "id_to_move": {
                int(k): v for k, v in self.id_to_move.items()
            },  # Ensure keys are serializable
            "vocab_size": self.vocab_size,
        }
        with open(file_path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, file_path):
        """Load the tokenizer state from a JSON file."""
        with open(file_path, "r") as f:
            state = json.load(f)

        tokenizer = cls()
        tokenizer.move_to_id = state["move_to_id"]
        tokenizer.id_to_move = {int(k): v for k, v in state["id_to_move"].items()}
        tokenizer.vocab_size = state["vocab_size"]
        return tokenizer

    @classmethod
    def fit(cls, csv_file):
        unique_moves = set()

        # Count total lines so we can show progress in the actual fit step
        total_lines = count_lines_fast(csv_file)

        with open(csv_file, "r") as data:
            for i, row in enumerate(
                tqdm(data, total=total_lines, desc="Processing moves")
            ):
                if i == 0:
                    # Skip header row
                    continue

                context, next_move, _is_checkmate, _outcome = row.strip().split(",")
                context = context.strip().split()
                unique_moves.add(next_move)
                for move in context:
                    unique_moves.add(move)

        tokenizer = cls()
        moves = sorted(list(unique_moves))
        for move in moves:
            if move not in tokenizer.move_to_id:
                tokenizer.move_to_id[move] = tokenizer.vocab_size
                tokenizer.id_to_move[tokenizer.vocab_size] = move
                tokenizer.vocab_size += 1
        return tokenizer
