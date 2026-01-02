"""Experiment logging with Weights & Biases."""

from typing import Any, Dict, List, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class ExperimentLogger:
    """Wrapper for Weights & Biases experiment tracking."""

    def __init__(
        self,
        config: Dict[str, Any],
        project: str = "chessgpt-2026",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        resume: Optional[str] = None,
    ):
        """
        Initialize WandB experiment logger.

        Args:
            config: Configuration dictionary to log
            project: WandB project name
            entity: WandB entity (username/team)
            name: Run name
            tags: List of tags for this run
            resume: Resume mode ('allow', 'must', 'never', or None)
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags or [],
            resume=resume,
        )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step number
        """
        wandb.log(metrics, step=step)

    def log_predictions(
        self,
        contexts: List[str],
        predictions: List[str],
        targets: List[str],
        step: int,
        num_samples: int = 10,
    ):
        """
        Log sample predictions as a wandb.Table.

        Args:
            contexts: List of context move sequences
            predictions: List of predicted moves
            targets: List of target moves
            step: Training step number
            num_samples: Number of samples to log
        """
        # Take only first num_samples
        contexts = contexts[:num_samples]
        predictions = predictions[:num_samples]
        targets = targets[:num_samples]

        table = wandb.Table(
            columns=["context", "predicted", "target", "correct"],
            data=[
                [ctx, pred, tgt, "✓" if pred == tgt else "✗"]
                for ctx, pred, tgt in zip(contexts, predictions, targets)
            ],
        )
        wandb.log({"predictions": table}, step=step)

    def log_model(self, model_path: str, name: str = "model", aliases: Optional[List[str]] = None):
        """
        Log model artifact to WandB.

        Args:
            model_path: Path to model directory
            name: Artifact name
            aliases: List of aliases (e.g., ['best', 'latest'])
        """
        artifact = wandb.Artifact(name, type="model")
        artifact.add_dir(model_path)
        wandb.log_artifact(artifact, aliases=aliases or [])

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch model gradients and parameters.

        Args:
            model: PyTorch model to watch
            log: What to log ('gradients', 'parameters', 'all')
            log_freq: Logging frequency
        """
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish the WandB run."""
        wandb.finish()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finish the run."""
        self.finish()
