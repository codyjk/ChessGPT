"""Chess-specific evaluation metrics."""

from typing import List
import torch
import chess


class ChessMetrics:
    """Collection of chess-specific evaluation metrics."""

    @staticmethod
    def move_accuracy(
        predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> float:
        """
        Calculate top-1 move accuracy with masking.

        Args:
            predictions: Predicted token IDs [batch_size, seq_len] or logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            mask: Binary mask [batch_size, seq_len]

        Returns:
            Accuracy as a float between 0 and 1
        """
        if predictions.dim() == 3:  # Logits
            predictions = predictions.argmax(dim=-1)

        correct = (predictions == targets) & mask.bool()
        return (correct.sum() / mask.sum()).item()

    @staticmethod
    def top_k_accuracy(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, k: int = 5
    ) -> float:
        """
        Calculate top-k move accuracy with masking.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            mask: Binary mask [batch_size, seq_len]
            k: Top-k value

        Returns:
            Top-k accuracy as a float between 0 and 1
        """
        topk_preds = logits.topk(k, dim=-1).indices  # [batch_size, seq_len, k]
        targets_expanded = targets.unsqueeze(-1).expand_as(
            topk_preds
        )  # [batch_size, seq_len, k]

        # Check if target is in top-k predictions
        in_topk = (topk_preds == targets_expanded).any(dim=-1)  # [batch_size, seq_len]
        correct = in_topk & mask.bool()

        return (correct.sum() / mask.sum()).item()

    @staticmethod
    def legal_move_rate(
        predictions: List[str], board_states: List[chess.Board]
    ) -> float:
        """
        Calculate percentage of predicted moves that are legal.

        Args:
            predictions: List of predicted moves in SAN notation
            board_states: List of corresponding board states

        Returns:
            Legal move rate as a float between 0 and 1
        """
        if len(predictions) == 0:
            return 0.0

        legal_count = 0
        for move, board in zip(predictions, board_states):
            try:
                # Try to parse the move
                parsed_move = board.parse_san(move)
                # Check if it's in the legal moves
                if parsed_move in board.legal_moves:
                    legal_count += 1
            except (ValueError, AssertionError):
                # Move is illegal or malformed
                pass

        return legal_count / len(predictions)

    @staticmethod
    def checkmate_accuracy(
        is_checkmate_pred: torch.Tensor, is_checkmate_target: torch.Tensor
    ) -> float:
        """
        Calculate accuracy of checkmate prediction.

        Args:
            is_checkmate_pred: Predicted checkmate flags [batch_size]
            is_checkmate_target: Target checkmate flags [batch_size]

        Returns:
            Accuracy as a float between 0 and 1
        """
        # Round predictions to 0 or 1
        pred_binary = (is_checkmate_pred > 0.5).float()
        correct = (pred_binary == is_checkmate_target).float()
        return correct.mean().item()

    @staticmethod
    def perplexity(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Calculate perplexity.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            mask: Binary mask [batch_size, seq_len]

        Returns:
            Perplexity value
        """
        # Calculate cross-entropy loss
        batch_size, seq_len, vocab_size = logits.size()
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Cross-entropy per token
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(logits_flat, targets_flat)

        # Apply mask and average
        masked_loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        # Perplexity is exp(loss)
        return torch.exp(masked_loss).item()

    @staticmethod
    def batch_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        is_checkmate_pred: torch.Tensor = None,
        is_checkmate_target: torch.Tensor = None,
    ) -> dict:
        """
        Calculate all metrics for a batch.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            mask: Binary mask [batch_size, seq_len]
            is_checkmate_pred: Optional checkmate predictions [batch_size]
            is_checkmate_target: Optional checkmate targets [batch_size]

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "move_accuracy": ChessMetrics.move_accuracy(logits, targets, mask),
            "top5_accuracy": ChessMetrics.top_k_accuracy(logits, targets, mask, k=5),
            "perplexity": ChessMetrics.perplexity(logits, targets, mask),
        }

        if is_checkmate_pred is not None and is_checkmate_target is not None:
            metrics["checkmate_accuracy"] = ChessMetrics.checkmate_accuracy(
                is_checkmate_pred, is_checkmate_target
            )

        return metrics
