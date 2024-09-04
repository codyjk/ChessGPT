import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class ChessTransformer(nn.Module):
    # The defaults here are a relatively small and easy-to-train model
    def __init__(self, vocab_size, n_positions=10, n_embd=256, n_layer=4, n_head=4):
        super(ChessTransformer, self).__init__()

        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )

        self.transformer = GPT2Model(self.config)
        self.next_move_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        prediction_hidden_state = hidden_states[:, -1, :]

        next_move_logits = self.next_move_head(prediction_hidden_state)
        return next_move_logits
