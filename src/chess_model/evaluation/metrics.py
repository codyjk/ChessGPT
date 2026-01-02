"""Chess-specific evaluation metrics."""

from typing import List, Optional, Tuple
import torch
import chess

from .checkmate_analysis import find_mate_in_one, find_mate_in_two, analyze_position


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
    def checkmate_delivery_rate(
        predictions: List[str],
        board_states: List[chess.Board],
        mate_types: Optional[List[int]] = None,
    ) -> dict:
        """
        Calculate checkmate delivery rate - percentage of mate puzzles solved.

        Args:
            predictions: List of predicted moves in SAN notation
            board_states: List of corresponding board states
            mate_types: Optional list of mate types (1 for mate-in-1, 2 for mate-in-2, etc.)

        Returns:
            Dictionary with delivery rates by mate type:
            {
                'overall_rate': float,
                'mate_in_1_rate': float,
                'mate_in_2_rate': float,
                'mate_in_1_count': int,
                'mate_in_2_count': int,
                'total_positions': int,
            }
        """
        if len(predictions) == 0 or len(board_states) == 0:
            return {
                'overall_rate': 0.0,
                'mate_in_1_rate': 0.0,
                'mate_in_2_rate': 0.0,
                'mate_in_1_count': 0,
                'mate_in_2_count': 0,
                'total_positions': 0,
            }

        mate_in_1_correct = 0
        mate_in_1_total = 0
        mate_in_2_correct = 0
        mate_in_2_total = 0
        overall_correct = 0

        for i, (pred_move, board) in enumerate(zip(predictions, board_states)):
            # Analyze position to determine mate type
            analysis = analyze_position(board)

            if analysis['mate_in_1']:
                mate_in_1_total += 1
                # Check if prediction matches mate-in-1 move
                try:
                    pred_parsed = board.parse_san(pred_move)
                    if pred_parsed == analysis['mate_in_1']:
                        mate_in_1_correct += 1
                        overall_correct += 1
                except (ValueError, AssertionError):
                    pass

            elif analysis['mate_in_2']:
                mate_in_2_total += 1
                # Check if prediction matches mate-in-2 first move
                try:
                    pred_parsed = board.parse_san(pred_move)
                    if pred_parsed == analysis['mate_in_2']:
                        mate_in_2_correct += 1
                        overall_correct += 1
                except (ValueError, AssertionError):
                    pass

        # Calculate rates
        overall_rate = overall_correct / len(predictions) if len(predictions) > 0 else 0.0
        mate_in_1_rate = mate_in_1_correct / mate_in_1_total if mate_in_1_total > 0 else 0.0
        mate_in_2_rate = mate_in_2_correct / mate_in_2_total if mate_in_2_total > 0 else 0.0

        return {
            'overall_rate': overall_rate,
            'mate_in_1_rate': mate_in_1_rate,
            'mate_in_2_rate': mate_in_2_rate,
            'mate_in_1_correct': mate_in_1_correct,
            'mate_in_1_total': mate_in_1_total,
            'mate_in_2_correct': mate_in_2_correct,
            'mate_in_2_total': mate_in_2_total,
            'total_positions': len(predictions),
        }

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
