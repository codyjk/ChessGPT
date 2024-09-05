import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from chess_model.model.tokenizer import ChessTokenizer


def train_model(
    model, train_dataloader, val_dataloader, num_epochs, learning_rate, device
):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )
    next_move_criterion = nn.CrossEntropyLoss()

    total_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training Progress")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            # input_ids shape: [batch_size, seq_len]
            # Assuming batch_size = 128, seq_len = 50
            input_ids = batch["input_ids"].to(device)

            # next_move_labels shape: [batch_size, seq_len]
            next_move_labels = batch["labels"].to(device)

            # move_mask shape: [batch_size, seq_len]
            move_mask = batch["move_mask"].to(device)

            optimizer.zero_grad()

            # Forward pass next_move_logits shape: [batch_size, seq_len, vocab_size]
            # Assuming batch_size = 128, seq_len = 50, vocab_size = 531
            next_move_logits = model(input_ids)

            loss = calculate_masked_loss(
                next_move_logits, next_move_labels, move_mask, next_move_criterion
            )

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
                move_mask = batch["move_mask"].to(device)
                next_move_logits = model(input_ids)
                loss = calculate_masked_loss(
                    next_move_logits, next_move_labels, move_mask, next_move_criterion
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Learning Rate: {current_lr:.6f}"
        )

    progress_bar.close()
    return model


def calculate_masked_loss(next_move_logits, next_move_labels, move_mask, criterion):
    # Reshape tensors for loss calculation
    # Assuming batch_size = 128, seq_len = 50 for comments
    batch_size, seq_len, vocab_size = next_move_logits.size()

    # Reshape next_move_logits to [batch_size * seq_len, vocab_size]
    # New shape: [6400, 531] (128 * 50 = 6400)
    next_move_logits = next_move_logits.view(-1, vocab_size)

    # Reshape next_move_labels to [batch_size * seq_len]
    # New shape: [6400] (128 * 50 = 6400)
    next_move_labels = next_move_labels.view(-1)

    move_mask = move_mask.view(-1)

    # Calculate loss
    loss = criterion(next_move_logits, next_move_labels)

    # Apply move mask to the loss
    masked_loss = loss * move_mask

    # Average the loss over non-zero elements
    final_loss = masked_loss.sum() / (move_mask.sum() + 1e-8)

    return final_loss
