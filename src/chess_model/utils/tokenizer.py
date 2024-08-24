import json

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
        with open(csv_file, "r") as data:
            for row_number, row in enumerate(data):
                if row_number == 0:
                    # Skip header
                    continue

                context, next_move, _is_checkmate, _outcome = row.split(",")
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
