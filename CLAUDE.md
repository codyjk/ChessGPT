# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChessGPT is a custom-trained GPT-2 transformer model that learns to predict and play chess moves. The model has no prior knowledge of chess rules—it learns entirely from sequences of grandmaster-level games in algebraic notation. The architecture is based on GPT-2 and trains on move sequences like `d4 e6 Nf3 g6 Bg5 Bg7 Bxd8 1-0`.

## Development Commands

### Installation

Two installation modes depending on use case:
```bash
# For model training/playing (requires GPU)
poetry install --with model

# For S3 data processing only (e.g., on EC2)
poetry install --without model
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_chess_dataset.py

# Run with verbose output
poetry run pytest -v
```

### Code Quality

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Lint code
poetry run flake8

# Type checking
poetry run mypy src/
```

### Interactive CLI

```bash
# Launch interactive exploration CLI (play against pre-trained models)
poetry run explore
```

### Training Pipeline (Full Workflow)

```bash
# 1. Reduce PGN database to move sequences (segmented by ELO rating)
poetry run reduce-pgn --input-pgn-file data/lichess.pgn --output-dir out

# 2. Prepare training and validation datasets
poetry run prepare-training-data \
  --input-reduced-pgn-file out/master.txt \
  --output-training-data-file out/training-data.csv \
  --output-validation-data-file out/validation-data.csv \
  --max-context-length 10 \
  --validation-split 0.1

# 3. Fit and save tokenizer
poetry run fit-and-save-tokenizer \
  --input-training-data-file out/training-data.csv \
  --output-tokenizer-file out/chess_tokenizer.json

# 4. Train the model
poetry run train-model \
  --input-training-data-file out/training-data.csv \
  --input-validation-data-file out/validation-data.csv \
  --input-tokenizer-file out/chess_tokenizer.json \
  --max-context-length 10 \
  --num-embeddings 256 \
  --num-layers 4 \
  --num-heads 4 \
  --num-epochs 10 \
  --batch-size 128

# 5. Play against the trained model
poetry run play \
  --input-model-file out/chess_transformer_model.pth \
  --input-tokenizer-file out/chess_tokenizer.json \
  --max-context-length 10 \
  --num-embeddings 256 \
  --num-layers 4 \
  --num-heads 4 \
  --color white
```

### S3 Utilities (for AWS data processing)

```bash
poetry run s3-tree
poetry run s3-list-pgns
poetry run s3-extract-games-from-pgns
poetry run s3-extract-checkmates-from-pgns
```

## Architecture

### Core Modules

**`src/chess_model/model/`**
- `transformer.py`: `ChessTransformer` wraps HuggingFace GPT2Model with a linear head for next-move prediction
- `tokenizer.py`: `ChessTokenizer` maps chess moves to token IDs. Uses custom vocabulary fitted from training data (not BPE). Includes `[PAD]` and `[UNK]` special tokens.

**`src/chess_model/data/`**
- `dataset.py`: `ChessDataset` reads CSV files with format `context,is_checkmate,outcome`. Uses memory-mapped file access with cached line offsets for efficient random access without loading entire dataset into memory. Implements outcome-based masking (trains on winning player's moves for decisive games, all moves for draws).

**`src/chess_model/training/`**
- `training.py`: Contains `train_model()` function with AdamW optimizer, ReduceLROnPlateau scheduler, gradient clipping, and masked loss calculation that respects outcome-based move masking.

**`src/pgn_utils/`**
- `pgn_utils.py`: PGN parsing utilities with regex patterns for metadata, moves, and outcomes. Key function: `process_raw_games_from_file()` yields `RawGame` named tuples.
- `count_lines.py`: Fast line counting for progress bars during large file processing.

### Data Flow

1. **PGN → Reduced Moves**: Raw PGN games are parsed and reduced to space-separated move sequences with outcomes (`d4 e6 Nf3 1-0`)
2. **Moves → Training Data**: Move sequences become CSV rows with context windows (previous moves) and labels (next move)
3. **Tokenization**: ChessTokenizer maps each unique move string to an integer ID
4. **Dataset Loading**: ChessDataset provides PyTorch-compatible batches with padding, masking, and outcome labels
5. **Training**: ChessTransformer learns to predict next moves via cross-entropy loss with outcome-based masking
6. **Inference**: Trained model predicts legal moves given game context; CLI handles move validation via python-chess

### Key Design Patterns

- **Memory-efficient data loading**: ChessDataset uses mmap with line offset indexing to handle multi-GB CSV files without loading into RAM
- **Outcome-based training**: Loss function masks out moves from losing players, so model learns winning patterns
- **Left-padding**: Context sequences are left-padded with `[PAD]` tokens to handle variable-length game contexts
- **Hyperparameter matching**: Training and inference scripts must use identical hyperparameters (max_context_length, num_embeddings, num_layers, num_heads)

### Important Constraints

- **Hyperparameter consistency**: The model architecture is frozen at training time. To use a trained model for inference (play/explore), you must pass the exact same hyperparameters used during training.
- **Tokenizer dependency**: The tokenizer vocabulary is fitted from training data. You cannot use a model with a different tokenizer than the one used during training.
- **Max context length**: Truncates games longer than max_context_length by keeping only the most recent moves.

## Common Pitfalls

- **Mismatched hyperparameters**: If playing with a trained model, ensure all architecture params (n_positions, n_embd, n_layer, n_head) match training config
- **Missing tokenizer**: Always generate and save tokenizer before training; load same tokenizer file during inference
- **Large file handling**: Use `shuf -n N` to create smaller subsets for quick testing rather than processing full Lichess databases
- **Device compatibility**: Code automatically detects CUDA, MPS, or CPU. On M1/M2 Macs, PyTorch will use MPS backend.
