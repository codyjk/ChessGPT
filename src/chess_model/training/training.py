import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from chess_model.data.progressive_dataset import ProgressiveDataset
from chess_model.utils.tokenizer import ChessTokenizer

# Maybe parameterize this?
DATALOADER_BATCH_SIZE = 256


def fit_tokenizer(csv_file):
    unique_moves = set()
    with open(csv_file, "r") as data:
        for row in data:
            context, _next_move, _is_checkmate, _outcome = row.split(",")
            context = context.strip().split()
            for move in context:
                unique_moves.add(move)

    tokenizer = ChessTokenizer()
    tokenizer.fit(list(unique_moves))
    return tokenizer


def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate, device):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    move_criterion = nn.CrossEntropyLoss()
    checkmate_criterion = nn.BCEWithLogitsLoss()
    outcome_criterion = nn.BCEWithLogitsLoss()

    progressive_dataset = ProgressiveDataset(train_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=DATALOADER_BATCH_SIZE)

    overall_progress_bar = tqdm(total=num_epochs, desc="Overall Progress", position=0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Recreate the train_dataloader each epoch to account for the grown dataset
        train_dataloader = DataLoader(
            progressive_dataset, batch_size=DATALOADER_BATCH_SIZE, shuffle=True
        )

        # Create a progress bar for each epoch
        epoch_progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            position=1,
            leave=False,
        )

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            move_labels = batch["labels"].to(device)
            checkmate_labels = batch["is_checkmate"].to(device)
            outcome_labels = batch["outcome"].to(device)

            optimizer.zero_grad()

            move_logits, checkmate_logits, outcome_logits = model(input_ids)

            move_loss = move_criterion(move_logits, move_labels)
            checkmate_loss = checkmate_criterion(
                checkmate_logits.squeeze(), checkmate_labels
            )
            outcome_loss = outcome_criterion(outcome_logits, outcome_labels)

            loss = move_loss + 0.1 * checkmate_loss + 0.1 * outcome_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update epoch progress bar
            epoch_progress_bar.update(1)
            epoch_progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                move_labels = batch["labels"].to(device)
                checkmate_labels = batch["is_checkmate"].to(device)
                outcome_labels = batch["outcome"].to(device)

                move_logits, checkmate_logits, outcome_logits = model(input_ids)

                move_loss = move_criterion(move_logits, move_labels)
                checkmate_loss = checkmate_criterion(
                    checkmate_logits.squeeze(), checkmate_labels
                )
                outcome_loss = outcome_criterion(outcome_logits, outcome_labels)

                loss = move_loss + 0.1 * checkmate_loss + 0.1 * outcome_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Grow the dataset
        progressive_dataset.grow()

        # Close the epoch progress bar
        epoch_progress_bar.close()

        # Update overall progress bar
        overall_progress_bar.update(1)
        overall_progress_bar.set_postfix(
            {
                "Train Loss": f"{avg_loss:.4f}",
                "Val Loss": f"{avg_val_loss:.4f}",
                "Samples": len(progressive_dataset),
            }
        )

    overall_progress_bar.close()
    return model


def calculate_random_baseline(dataset, vocab_size, device):
    total_loss = 0
    move_criterion = nn.CrossEntropyLoss()
    checkmate_criterion = nn.BCEWithLogitsLoss()
    outcome_criterion = nn.BCEWithLogitsLoss()
    dataloader = DataLoader(dataset, batch_size=DATALOADER_BATCH_SIZE)

    for batch in tqdm(dataloader, desc="Calculating random baseline"):
        batch_size = batch["labels"].size(0)

        random_move_logits = torch.rand(batch_size, vocab_size).to(device)
        random_checkmate_logits = torch.rand(batch_size, 1).to(device)
        random_outcome_logits = torch.rand(batch_size, 3).to(device)

        move_loss = move_criterion(random_move_logits, batch["labels"].to(device))
        checkmate_loss = checkmate_criterion(
            random_checkmate_logits.squeeze(), batch["is_checkmate"].to(device)
        )
        outcome_loss = outcome_criterion(
            random_outcome_logits, batch["outcome"].to(device)
        )

        loss = move_loss + 0.1 * checkmate_loss + 0.1 * outcome_loss
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
