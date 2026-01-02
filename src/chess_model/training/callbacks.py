"""Custom training callbacks for ChessGPT."""

from typing import Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class CheckmateMetricsCallback(TrainerCallback):
    """
    Callback to track checkmate-specific metrics during training.

    Logs additional metrics like:
    - Checkmate delivery rate
    - Mate-in-N accuracy
    - False positive rate
    """

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self.checkmate_correct = 0
        self.checkmate_total = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Called when logging metrics.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control
            logs: Dictionary of metrics being logged
        """
        if logs is not None and state.global_step % self.log_every_n_steps == 0:
            # Add checkmate-specific metrics if available
            if self.checkmate_total > 0:
                logs["checkmate_accuracy_running"] = (
                    self.checkmate_correct / self.checkmate_total
                )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Called after evaluation.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control
            metrics: Evaluation metrics
        """
        if metrics is not None:
            # Log checkmate metrics separately
            checkmate_metrics = {
                k: v for k, v in metrics.items() if "checkmate" in k.lower()
            }
            if checkmate_metrics:
                print(f"\n{'='*60}")
                print("Checkmate Metrics:")
                for k, v in checkmate_metrics.items():
                    print(f"  {k}: {v:.4f}")
                print(f"{'='*60}\n")


class ModelSavingCallback(TrainerCallback):
    """
    Enhanced model saving with metadata.

    Saves model with:
    - Training metrics
    - Hyperparameters
    - Timestamp
    - Evaluation results
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called when saving a checkpoint.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control
        """
        import json
        from datetime import datetime
        from pathlib import Path

        # Get checkpoint directory
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        if checkpoint_dir.exists():
            # Save metadata
            metadata = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
                "timestamp": datetime.now().isoformat(),
                "training_args": {
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.num_train_epochs,
                    "batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                },
            }

            # Add latest metrics if available
            if state.log_history:
                latest_metrics = state.log_history[-1]
                metadata["latest_metrics"] = {
                    k: v
                    for k, v in latest_metrics.items()
                    if isinstance(v, (int, float))
                }

            # Save to file
            metadata_file = checkpoint_dir / "training_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved training metadata to {metadata_file}")


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback with custom patience logic.

    Stops training if validation metric doesn't improve for N evaluations.
    """

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
    ):
        """
        Initialize early stopping.

        Args:
            early_stopping_patience: Number of evaluations to wait for improvement
            early_stopping_threshold: Minimum change to count as improvement
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_metric = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Check if we should stop early.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control
            metrics: Evaluation metrics
        """
        if metrics is None:
            return

        # Get the metric we're tracking
        metric_name = args.metric_for_best_model
        if metric_name not in metrics:
            return

        current_metric = metrics[metric_name]

        # Determine if this is an improvement
        if self.best_metric is None:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
        else:
            # Check if improved (consider if higher or lower is better)
            if "loss" in metric_name.lower():
                # Lower is better for loss
                improved = current_metric < (
                    self.best_metric - self.early_stopping_threshold
                )
            else:
                # Higher is better for other metrics
                improved = current_metric > (
                    self.best_metric + self.early_stopping_threshold
                )

            if improved:
                self.best_metric = current_metric
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

        # Check if we should stop
        if self.early_stopping_counter >= self.early_stopping_patience:
            print(
                f"\nEarly stopping triggered! No improvement in {metric_name} "
                f"for {self.early_stopping_patience} evaluations."
            )
            control.should_training_stop = True


class ProgressCallback(TrainerCallback):
    """
    Enhanced progress logging with move predictions.

    Shows:
    - Training progress with time estimates
    - Sample predictions during training
    - Metric trends
    """

    def __init__(self, log_predictions_every_n_steps: int = 500):
        self.log_predictions_every_n_steps = log_predictions_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the end of each training step.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control
        """
        # Log detailed progress periodically
        if state.global_step % self.log_predictions_every_n_steps == 0:
            # Calculate training progress
            if state.max_steps > 0:
                progress = state.global_step / state.max_steps
                print(f"\n{'='*60}")
                print(f"Training Progress: {progress:.1%}")
                print(f"Global Step: {state.global_step}/{state.max_steps}")
                print(f"Epoch: {state.epoch:.2f}")
                if state.log_history:
                    latest = state.log_history[-1]
                    if "loss" in latest:
                        print(f"Latest Loss: {latest['loss']:.4f}")
                    if "learning_rate" in latest:
                        print(f"Learning Rate: {latest['learning_rate']:.2e}")
                print(f"{'='*60}\n")


def get_default_callbacks(config: Optional[Dict] = None):
    """
    Get default set of callbacks for ChessGPT training.

    Args:
        config: Optional configuration dictionary

    Returns:
        List of callback instances
    """
    callbacks = [
        CheckmateMetricsCallback(log_every_n_steps=100),
        ModelSavingCallback(),
        ProgressCallback(log_predictions_every_n_steps=500),
    ]

    # Add early stopping if configured
    if config and config.get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"],
                early_stopping_threshold=config.get("early_stopping_threshold", 0.0),
            )
        )

    return callbacks
