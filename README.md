# ChessGPT

A transformer-based chess model trained on grandmaster games. The model learns chess patterns from move sequences without prior knowledge of rules.

## Features

- GPT-2 Medium architecture (303M parameters)
- Trained on 180K+ master-level games (1800+ Elo)
- Outcome-based learning (trains on winning player's moves)
- Two-phase training: general chess → checkmate specialization
- Unified training workflow with automatic model versioning

## Setup

Install dependencies with [Poetry](https://python-poetry.org/):

```bash
# For training and playing (requires GPU)
poetry install --with model

# For data processing only
poetry install --without model
```

## Quick Start

### Play Against Pre-trained Models

```bash
poetry run explore
```

### Train a New Model

```bash
# Unified training command
poetry run train --config phase1_gpt2_medium --name my-model

# View trained models
poetry run models list

# Play against your model
poetry run play --model my-model
```

## Training Pipeline

### Prerequisites

Before training, prepare data and tokenizer once:

```bash
# 1. Extract games from PGN
poetry run reduce-pgn --input-pgn-file data/lichess.pgn --output-dir out

# 2. Convert to training format
poetry run prepare-training-data \
  --input-reduced-pgn-file out/master.txt \
  --output-training-data-file out/training.csv \
  --output-validation-data-file out/validation.csv \
  --max-context-length 100

# 3. Create tokenizer (only needed once)
poetry run fit-and-save-tokenizer \
  --input-training-data-file out/training.csv \
  --output-tokenizer-file out/chess_tokenizer.json
```

### Train Model

Once data is prepared, use the unified training command:

```bash
poetry run train --config phase1_gpt2_medium --name gpt2-baseline
```

This handles model creation, training, checkpointing, and automatic registry. Configuration files in `configs/` specify architecture, hyperparameters, and data paths.

### Two-Phase Training

For checkmate-focused training:

```bash
# Phase 1: General chess (157K examples, 8 epochs)
poetry run train --config phase1_gpt2_medium --name model-phase1

# Phase 2: Checkmate specialization (resume from Phase 1)
poetry run train --config phase2_gpt2_medium --name model-phase2 \
  --resume-from models/model-phase1/checkpoint-best
```

## Model Management

```bash
# List all trained models
poetry run models list

# Show model details
poetry run models show <model-name>

# Compare two models
poetry run models compare model-a model-b

# Tag and organize models
poetry run models tag <model-name> production

# Set recommended model
poetry run models recommend <model-name>
```

## Testing

Run validation tests:

```bash
# Test checkmate ability
poetry run python scripts/test_checkmate_ability.py --model <model-name>

# Test model predictions
poetry run python scripts/test_model_predictions.py --model-dir models/<model-name>/final_model
```

## Configuration

Training is configured via YAML files in `configs/`:

- `configs/model/` - Model architectures (GPT-2 Medium, etc.)
- `configs/training/` - Training hyperparameters
- `configs/data/` - Dataset paths and preprocessing
- `configs/pipeline/` - Complete training pipelines

Override settings via CLI:

```bash
poetry run train --config phase1_gpt2_medium --name my-model --tags experimental
```

## Data Format

Games are represented as space-separated move sequences:

```
d4 e6 Nf3 g6 Bg5 Bg7 Bxd8 1-0
e4 c5 Nf3 e6 d4 cxd4 Nxd4 1/2-1/2
```

The model learns from these sequences using outcome-based masking (only trains on winning player's moves for decisive games).

## Architecture

**GPT-2 Medium**:
- 303M parameters (24 layers, 1024 embedding dim, 16 heads)
- Context length: 100 moves
- Vocabulary: 287 unique chess moves
- Mixed precision (bfloat16) for M1/M2 Macs

## License

MIT
