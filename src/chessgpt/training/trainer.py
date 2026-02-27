"""
Clean PyTorch training loop for the chess transformer.

No HuggingFace Trainer — just explicit forward/backward/step with:
- AdamW optimizer with weight decay
- Cosine annealing learning rate schedule with warmup
- Gradient clipping
- Checkpoint saving per epoch
- Multi-task loss (policy + value + checkmate)
- Mixed precision (autocast) for CUDA
- Gradient accumulation for larger effective batch sizes
- Optional torch.compile on CUDA
- tqdm progress bars for real-time monitoring
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from chessgpt.model.transformer import ChessTransformer
from chessgpt.training.loss import MultiTaskLoss


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_autocast_context(device: torch.device):
    """Return an autocast context manager appropriate for the device.

    CUDA: float16 autocast (works with GradScaler).
    MPS/CPU: no-op (MPS float16 autocast adds overhead that outweighs
    compute savings for models under ~100M params).
    """
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def train(
    model: ChessTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: dict,
    device: torch.device,
    output_dir: str,
    *,
    log_style: str = "tqdm",
) -> dict:
    """
    Train the model and save checkpoints.

    config keys: lr, num_epochs, grad_clip, alpha, beta, warmup_steps,
                 accumulation_steps

    Args:
        log_style: "tqdm" for interactive progress bars (default),
                   "line" for newline-based batch logging (better for tmux/logs).

    Returns: final metrics dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Optional torch.compile — only on CUDA (MPS doesn't support it well yet)
    compiled = False
    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
        compiled = True
        print("torch.compile enabled")

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    accumulation_steps = config.get("accumulation_steps", 1)

    # Cosine schedule with linear warmup
    # Scheduler steps once per *effective* batch (after accumulation)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = config["num_epochs"] * steps_per_epoch
    warmup_steps = config.get("warmup_steps", min(100, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    loss_fn = MultiTaskLoss(
        alpha=config.get("alpha", 0.5),
        beta=config.get("beta", 0.5),
        pad_token_id=0,
    )

    # GradScaler for CUDA float16 (MPS doesn't support it)
    use_scaler = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    autocast_ctx = _get_autocast_context(device)

    use_tqdm = log_style == "tqdm"

    best_val_loss = float("inf")
    no_improve_count = 0
    patience = config.get("patience", 0)  # 0 = disabled
    training_log = []
    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        # --- Train ---
        model.train()
        epoch_losses = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "checkmate_loss": 0.0,
            "total_loss": 0.0,
        }
        num_batches = 0

        optimizer.zero_grad()
        total_batches = len(train_loader)
        # Log every ~10% of batches (at least every 50, at most every 500)
        line_log_interval = max(50, min(500, total_batches // 10))

        if use_tqdm:
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
                leave=True,
            )
            batch_iter = enumerate(pbar)
        else:
            print(f"Epoch {epoch + 1}/{config['num_epochs']} ({total_batches} batches)")
            batch_iter = enumerate(train_loader)

        for batch_idx, batch in batch_iter:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outcome = batch["outcome"].to(device)
            checkmate_available = batch["checkmate_available"].to(device)
            move_mask = batch["move_mask"].to(device)
            checkmate_weight = batch["checkmate_weight"].to(device)

            with autocast_ctx:
                policy_logits, value_logits, checkmate_logits = model(input_ids)

                loss, details = loss_fn(
                    policy_logits,
                    value_logits,
                    checkmate_logits,
                    labels,
                    outcome,
                    checkmate_available,
                    move_mask,
                    checkmate_weight,
                )
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer every accumulation_steps mini-batches (or at end of epoch)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for k, v in details.items():
                epoch_losses[k] += v
            num_batches += 1

            if use_tqdm:
                pbar.set_postfix(
                    loss=f"{details['total_loss']:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
            elif (batch_idx + 1) % line_log_interval == 0 or (batch_idx + 1) == total_batches:
                pct = 100 * (batch_idx + 1) / total_batches
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  batch {batch_idx + 1}/{total_batches} ({pct:.0f}%)"
                    f"  loss={details['total_loss']:.4f}  lr={lr:.2e}"
                )

        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        # --- Validate ---
        val_losses = None
        if val_loader is not None:
            val_losses = evaluate_loss(
                model, val_loader, loss_fn, device, autocast_ctx, use_tqdm=use_tqdm
            )

        # --- Log ---
        entry = {
            "epoch": epoch + 1,
            "train": epoch_losses,
            "val": val_losses,
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_log.append(entry)

        val_str = ""
        if val_losses:
            val_str = f" | val_loss: {val_losses['total_loss']:.4f}"
        print(
            f"Epoch {epoch + 1}/{config['num_epochs']}"
            f" | train_loss: {epoch_losses['total_loss']:.4f}"
            f" (policy: {epoch_losses['policy_loss']:.4f}"
            f", value: {epoch_losses['value_loss']:.4f}"
            f", checkmate: {epoch_losses['checkmate_loss']:.4f})"
            f"{val_str}"
        )

        # --- Checkpoint ---
        val_total = val_losses["total_loss"] if val_losses else epoch_losses["total_loss"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            no_improve_count = 0
            save_checkpoint(model, optimizer, epoch + 1, output_path / "model.pt")
        else:
            no_improve_count += 1
            if patience > 0 and no_improve_count >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    if compiled:
        print("(torch.compile was enabled)")

    # Save training log
    with open(output_path / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    return {"training_log": training_log, "training_time_seconds": elapsed}


def evaluate_loss(
    model: ChessTransformer,
    data_loader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    autocast_ctx=None,
    *,
    use_tqdm: bool = True,
) -> dict[str, float]:
    model.eval()
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "checkmate_loss": 0.0,
        "total_loss": 0.0,
    }
    num_batches = 0

    if autocast_ctx is None:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    with torch.no_grad():
        val_iter = tqdm(data_loader, desc="Validating", leave=False) if use_tqdm else data_loader
        for batch in val_iter:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outcome = batch["outcome"].to(device)
            checkmate_available = batch["checkmate_available"].to(device)
            move_mask = batch["move_mask"].to(device)
            checkmate_weight = batch["checkmate_weight"].to(device)

            with autocast_ctx:
                policy_logits, value_logits, checkmate_logits = model(input_ids)
                _, details = loss_fn(
                    policy_logits,
                    value_logits,
                    checkmate_logits,
                    labels,
                    outcome,
                    checkmate_available,
                    move_mask,
                    checkmate_weight,
                )

            for k, v in details.items():
                totals[k] += v
            num_batches += 1

    for k in totals:
        totals[k] /= max(num_batches, 1)
    return totals


def save_checkpoint(
    model: ChessTransformer,
    optimizer: AdamW,
    epoch: int,
    path: Path,
) -> None:
    # Unwrap compiled model if needed
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": raw_model.config.vocab_size,
                "d_model": raw_model.config.d_model,
                "n_layers": raw_model.config.n_layers,
                "n_heads": raw_model.config.n_heads,
                "max_seq_len": raw_model.config.max_seq_len,
                "dropout": raw_model.config.dropout,
            },
        },
        path,
    )
