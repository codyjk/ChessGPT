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

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for loss computation (ignored, handled by Trainer)

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        next_move_logits = self.next_move_head(hidden_states)
        return next_move_logits
