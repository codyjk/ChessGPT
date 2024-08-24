import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from chess_model.utils.tokenizer import ChessTokenizer


def train_model(
    model, train_dataloader, val_dataloader, num_epochs, learning_rate, device
):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )
    next_move_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model = None
    patience = 5
    patience_counter = 0

    total_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training Progress")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            next_move_labels = batch["labels"].to(device)

            optimizer.zero_grad()

            next_move_logits = model(input_ids)
            loss = next_move_criterion(next_move_logits, next_move_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"epoch": epoch + 1, "loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                next_move_labels = batch["labels"].to(device)

                next_move_logits = model(input_ids)
                loss = next_move_criterion(next_move_logits, next_move_labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Learning Rate: {current_lr:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    progress_bar.close()
    return best_model


def calculate_random_baseline(dataloader, vocab_size, device):
    total_loss = 0
    next_move_criterion = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Calculating random baseline"):
        batch_size = batch["input_ids"].size(0)

        random_next_move_logits = torch.rand(batch_size, vocab_size).to(device)
        loss = next_move_criterion(random_next_move_logits, batch["labels"].to(device))

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
