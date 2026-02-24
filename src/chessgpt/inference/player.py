"""
Pure AI move selection — no python-chess at runtime.

The model plays entirely on its own. If it outputs an illegal move, that's displayed
as-is. Legal move rate is a training metric, not a runtime filter.

Value and checkmate heads are available as diagnostics only during play
(displayed to user: "Model thinks white is 73% to win", "Checkmate probability: 2%").
They do NOT influence move selection.
"""

import torch
import torch.nn.functional as F

from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer


def predict_next_move(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    move_history: list[str],
    temperature: float = 0.3,
    top_k: int = 5,
) -> dict:
    """
    Predict the next move given game history.

    Returns dict with:
        - move: predicted move string
        - top_k_moves: list of (move, probability) tuples
        - value: [white_prob, draw_prob, black_prob]
        - checkmate_prob: float
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode_and_pad(move_history, model.config.max_seq_len)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        policy_logits, value_logits, checkmate_logits = model(input_tensor)

    # Policy: sample from top-k with temperature
    last_logits = policy_logits[0, -1, :] / temperature
    probs = F.softmax(last_logits, dim=-1)

    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[0]))
    sampled_idx = torch.multinomial(top_k_probs, 1).item()
    predicted_id = top_k_indices[sampled_idx].item()
    predicted_move = tokenizer.decode([predicted_id])[0]

    # Diagnostics
    top_k_moves = [
        (tokenizer.decode([idx.item()])[0], prob.item())
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]

    value_probs = F.softmax(value_logits[0], dim=-1).cpu().tolist()
    checkmate_prob = torch.sigmoid(checkmate_logits[0, 0]).item()

    return {
        "move": predicted_move,
        "top_k_moves": top_k_moves,
        "value": value_probs,
        "checkmate_prob": checkmate_prob,
    }
