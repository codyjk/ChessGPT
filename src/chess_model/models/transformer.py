import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class ChessTransformer(nn.Module):
    # The defaults here are a relatively small and easy-to-train model
    def __init__(self, vocab_size, n_positions=10, n_embd=128, n_layer=2, n_head=2):
        super(ChessTransformer, self).__init__()

        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )

        self.transformer = GPT2Model(self.config)
        self.move_head = nn.Linear(n_embd, vocab_size)
        self.checkmate_head = nn.Linear(n_embd, 1)
        self.outcome_head = nn.Linear(n_embd, 3)  # Win, Loss, Draw

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        prediction_hidden_state = hidden_states[:, -1, :]

        move_logits = self.move_head(prediction_hidden_state)
        checkmate_logits = self.checkmate_head(prediction_hidden_state)
        outcome_logits = self.outcome_head(prediction_hidden_state)

        return move_logits, checkmate_logits, outcome_logits
