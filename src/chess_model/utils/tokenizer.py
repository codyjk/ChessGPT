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

    def fit(self, moves):
        for move in moves:
            if move not in self.move_to_id:
                self.move_to_id[move] = self.vocab_size
                self.id_to_move[self.vocab_size] = move
                self.vocab_size += 1

    def encode(self, moves):
        return [self.move_to_id.get(move, self.move_to_id[UNK_TOKEN]) for move in moves]

    def decode(self, ids):
        return [self.id_to_move.get(id, UNK_TOKEN) for id in ids]
