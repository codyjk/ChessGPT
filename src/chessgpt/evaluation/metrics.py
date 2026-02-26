"""
Evaluation metrics for chess transformer models.

Three categories:
1. Move prediction accuracy (top-1, top-5) on held-out games
2. Legal move rate — % of predictions that are legal chess moves
3. Mate-in-1 accuracy — does the model find checkmate when it's available?

Legal move validation uses python-chess, but only during evaluation (never at inference).
"""

import chess
import torch
from torch.utils.data import DataLoader

from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer


def evaluate_move_accuracy(
    model: ChessTransformer,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute top-1 and top-5 move prediction accuracy on a dataset."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            move_mask = batch["move_mask"].to(device)

            policy_logits, _, _ = model(input_ids)

            # Only evaluate on non-masked, non-padding positions
            mask = (move_mask > 0) & (labels != 0)

            if mask.sum() == 0:
                continue

            logits_flat = policy_logits[mask]  # [N, vocab]
            labels_flat = labels[mask]  # [N]

            # Top-1
            top1 = logits_flat.argmax(dim=-1)
            correct_top1 += (top1 == labels_flat).sum().item()

            # Top-5
            top5 = logits_flat.topk(5, dim=-1).indices
            correct_top5 += (top5 == labels_flat.unsqueeze(-1)).any(dim=-1).sum().item()

            total += labels_flat.shape[0]

    return {
        "move_accuracy_top1": correct_top1 / max(total, 1),
        "move_accuracy_top5": correct_top5 / max(total, 1),
    }


def evaluate_legal_move_rate(
    model: ChessTransformer,
    data_loader: DataLoader,
    tokenizer: ChessTokenizer,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate what % of model's top-1 predictions are legal moves.

    Replays each game position with python-chess, checks if the model's
    predicted move is legal. Broken down by game phase.
    """
    model.eval()
    phase_counts = {"opening": [0, 0], "middlegame": [0, 0], "endgame": [0, 0]}
    total_legal = 0
    total_moves = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            move_mask = batch["move_mask"].to(device)

            policy_logits, _, _ = model(input_ids)
            predictions = policy_logits.argmax(dim=-1)  # [batch, seq]

            for b in range(input_ids.shape[0]):
                ids = input_ids[b].cpu().tolist()
                label_ids = labels[b].cpu().tolist()

                # Find where non-pad tokens start (left-padded sequences)
                first_real = 0
                while first_real < len(ids) and ids[first_real] == 0:
                    first_real += 1

                # Replay the game from the beginning, checking predictions at each position.
                # input_ids[pos] is the move played at this position.
                # labels[pos] is the next move (what we predict).
                # We need the board AFTER pushing input_ids[pos] to check if labels[pos] is legal.
                board = chess.Board()
                for pos in range(first_real, len(ids)):
                    move_str = tokenizer.decode([ids[pos]])[0]

                    # Try to push the input move to advance board state
                    try:
                        board.push_san(move_str)
                    except (ValueError, chess.InvalidMoveError):
                        break  # board state is broken, stop this sample

                    # If this position is masked out or label is padding, skip evaluation
                    if move_mask[b, pos] == 0 or label_ids[pos] == 0:
                        continue

                    # Check if the model's prediction at this position is legal
                    pred_id = predictions[b, pos].item()
                    pred_move = tokenizer.decode([pred_id])[0]

                    is_legal = False
                    try:
                        board.parse_san(pred_move)
                        is_legal = True
                    except (ValueError, chess.InvalidMoveError):
                        pass

                    # Determine game phase by ply count
                    ply = board.ply()
                    if ply < 20:
                        phase = "opening"
                    elif ply < 60:
                        phase = "middlegame"
                    else:
                        phase = "endgame"

                    phase_counts[phase][0] += 1 if is_legal else 0
                    phase_counts[phase][1] += 1
                    total_legal += 1 if is_legal else 0
                    total_moves += 1

    results = {
        "legal_move_rate": total_legal / max(total_moves, 1),
    }
    for phase, (legal, total) in phase_counts.items():
        results[f"legal_move_rate_{phase}"] = legal / max(total, 1)

    return results


def evaluate_value_accuracy(
    model: ChessTransformer,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate game outcome prediction accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            outcome = batch["outcome"].to(device)

            _, value_logits, _ = model(input_ids)
            preds = value_logits.argmax(dim=-1)
            targets = outcome.argmax(dim=-1)

            correct += (preds == targets).sum().item()
            total += targets.shape[0]

    return {"value_accuracy": correct / max(total, 1)}


def evaluate_checkmate_detection(
    model: ChessTransformer,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate checkmate detection accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            checkmate_available = batch["checkmate_available"].to(device)

            _, _, checkmate_logits = model(input_ids)
            # Threshold on raw logits: >0 corresponds to sigmoid>0.5
            preds = (checkmate_logits.squeeze(-1) > 0.0).float()

            correct += (preds == checkmate_available).sum().item()
            total += checkmate_available.shape[0]

    return {"checkmate_detection_accuracy": correct / max(total, 1)}
