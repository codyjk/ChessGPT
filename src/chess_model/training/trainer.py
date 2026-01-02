"""Custom HuggingFace Trainer for ChessGPT with outcome-based masking."""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from ..evaluation.metrics import ChessMetrics


class ChessTrainer(Trainer):
    """
    Custom Trainer that implements outcome-based masking for chess move prediction.

    Key features:
    - Outcome-based masking: Only learns from winning player's moves
    - Sample-level loss weighting: Higher weight for checkmate examples
    - Chess-specific metrics: Move accuracy, legal move rate, etc.
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,  # New parameter in transformers 4.46+
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss with outcome-based masking and sample weighting.

        Args:
            model: The model to train
            inputs: Dictionary containing:
                - input_ids: [batch_size, seq_len]
                - labels: [batch_size, seq_len]
                - move_mask: [batch_size, seq_len] - outcome-based mask
                - loss_weight: [batch_size] - optional sample weights
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        move_mask = inputs.get("move_mask")
        sample_weights = inputs.get("loss_weight", None)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
        )

        # Get logits (handle different model output types)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs[0]

        # Calculate masked loss
        loss = self._calculate_masked_loss(
            logits, labels, move_mask, sample_weights
        )

        return (loss, outputs) if return_outputs else loss

    def _calculate_masked_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        move_mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate cross-entropy loss with masking and weighting.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            move_mask: [batch_size, seq_len] - binary mask
            sample_weights: [batch_size] - optional per-sample weights

        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, vocab_size = logits.size()

        # Reshape for loss calculation
        logits_flat = logits.view(-1, vocab_size)  # [batch*seq, vocab]
        labels_flat = labels.view(-1)  # [batch*seq]

        # Compute per-token loss (no reduction)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss_per_token = loss_fn(logits_flat, labels_flat)  # [batch*seq]

        # Reshape back
        loss_per_token = loss_per_token.view(batch_size, seq_len)  # [batch, seq]

        # Apply move mask (outcome-based masking)
        if move_mask is not None:
            loss_per_token = loss_per_token * move_mask

        # Apply sample weights (e.g., higher weight for checkmate examples)
        if sample_weights is not None:
            # Expand sample weights to match sequence dimension
            sample_weights = sample_weights.view(-1, 1).expand_as(
                loss_per_token
            )  # [batch, seq]
            loss_per_token = loss_per_token * sample_weights

        # Calculate final loss
        if move_mask is not None:
            # Average over non-masked positions
            mask_sum = move_mask.sum()
            if mask_sum > 0:
                loss = loss_per_token.sum() / mask_sum
            else:
                loss = loss_per_token.sum()  # Fallback (shouldn't happen)
        else:
            # Average over all positions
            loss = loss_per_token.mean()

        return loss

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute chess-specific evaluation metrics.

        Args:
            eval_pred: EvalPrediction containing:
                - predictions: Model logits [num_samples, seq_len, vocab_size]
                - label_ids: Target labels [num_samples, seq_len]
                - inputs: Dictionary with move_mask, etc.

        Returns:
            Dictionary of metric names and values
        """
        logits = torch.from_numpy(eval_pred.predictions)
        labels = torch.from_numpy(eval_pred.label_ids)

        # Extract move_mask from inputs if available
        # Note: This requires passing inputs through evaluation
        # For now, create a simple all-ones mask
        mask = torch.ones_like(labels, dtype=torch.float)

        # Compute metrics
        metrics = ChessMetrics.batch_metrics(logits, labels, mask)

        return metrics

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Override prediction step to handle custom outputs.

        This ensures move_mask and other custom fields are properly handled
        during evaluation.
        """
        # Store move_mask for metrics computation
        self._current_move_mask = inputs.get("move_mask")

        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

    def log(self, logs: Dict[str, float]) -> None:
        """
        Override logging to add custom metrics and WandB integration.

        Args:
            logs: Dictionary of metrics to log
        """
        # Call parent logging
        super().log(logs)

        # Add gradient norm if available
        if hasattr(self.model, "parameters"):
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logs["grad_norm"] = total_norm

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create optimizer and learning rate scheduler.

        Supports warmup and different LR schedules as specified in TrainingArguments.
        """
        # Use parent implementation which respects TrainingArguments
        super().create_optimizer_and_scheduler(num_training_steps)


def create_training_args(config) -> TrainingArguments:
    """
    Create HuggingFace TrainingArguments from our config.

    Args:
        config: TrainingConfig from our schemas

    Returns:
        TrainingArguments instance
    """
    # Determine strategies based on steps configuration
    eval_strategy = "steps" if config.get("eval_steps", 0) > 0 else "epoch"
    save_strategy = "steps" if config.get("save_steps", 0) > 0 else "epoch"

    return TrainingArguments(
        output_dir=config.get("output_dir", "outputs/training"),
        num_train_epochs=config.get("num_epochs", 8),
        per_device_train_batch_size=config.get("batch_size", 64),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=config.get("learning_rate", 3e-4),
        warmup_steps=config.get("warmup_steps", 1000),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("gradient_clip_norm", 1.0),
        # Mixed precision
        fp16=config.get("mixed_precision") == "fp16",
        bf16=config.get("mixed_precision") == "bf16",
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        # Logging and saving
        logging_steps=config.get("logging_steps", 10),
        save_strategy=save_strategy,
        save_steps=config.get("save_steps") if save_strategy == "steps" else None,
        save_total_limit=config.get("save_total_limit", 3),
        evaluation_strategy=eval_strategy,
        eval_steps=config.get("eval_steps") if eval_strategy == "steps" else None,
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        # Reporting
        report_to=config.get("report_to") if config.get("report_to") is not None else None,
        # Misc
        dataloader_num_workers=0,  # Single worker for stability
        remove_unused_columns=False,  # Keep all columns including move_mask
    )
