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

## Cloud Training

Train on cloud GPU providers for 10-50x faster training.

### Quick Start

```bash
# Install deployment tools
poetry install --with model,deploy

# Deploy to Lambda Labs A100
poetry run cloud-train \
  --provider lambda \
  --gpu A100 \
  --config phase1_gpt2_medium \
  --name my-cloud-model

# Monitor progress
poetry run cloud-instances list

# Retrieve trained model (after completion)
# Model will be in models/my-cloud-model/
```

### Supported Providers

- **Lambda Labs** - Full API integration, easiest setup ($0.60-$1.50/hr)
- **SSH** - Any SSH-accessible GPU server (RunPod, Vast.ai, custom servers)

### Cost Estimates

| Provider | GPU | Training Time | Cost |
|----------|-----|---------------|------|
| Lambda A100 | A100 40GB | ~1-2 hours | ~$1-2 |
| Lambda A10 | A10 24GB | ~2-3 hours | ~$1-2 |
| RunPod (spot) | RTX 4090 | ~2-4 hours | ~$1-2 |

*Estimates for Phase 1 training (157K examples)*

### Data Processing

The system handles large Lichess PGN files efficiently:

**Option 1: Use pre-processed data** (recommended)
```bash
# Prepare data locally first
poetry run reduce-pgn --input-pgn-file data/lichess.pgn --output-dir out
poetry run prepare-training-data --input-reduced-pgn-file out/master.txt ...
poetry run fit-and-save-tokenizer --input-training-data-file out/training.csv ...

# Deploy with processed data (transfers small CSV files only)
poetry run cloud-train --provider lambda --gpu A100 --config phase1_gpt2_medium --name my-model
```

**Option 2: Process on cloud** (coming soon)
```bash
# Downloads PGN directly to cloud, processes there
poetry run cloud-train --provider lambda --gpu A100 --config phase1_gpt2_medium --name my-model \
  --prepare-data --pgn-url https://database.lichess.org/standard/lichess_db_standard_rated_2024-06.pgn.zst
```

### Instance Management

```bash
# List active instances
poetry run cloud-instances list

# SSH to instance
poetry run cloud-instances ssh <instance-id>

# Monitor training (future)
poetry run cloud-instances monitor <instance-id>

# Retrieve model (future)
poetry run cloud-instances pull <instance-id>

# Terminate instance (future)
poetry run cloud-instances stop <instance-id>
```

### Setup Requirements

**For Lambda Labs:**
1. Get API key from https://cloud.lambdalabs.com
2. Set environment variable: `export LAMBDA_API_KEY=your_key`
3. Generate SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/chessgpt_deploy`
4. Upload to Lambda: Visit dashboard → SSH Keys → Add `chessgpt_deploy.pub`

**For SSH providers (RunPod, Vast.ai):**
1. Manually provision GPU instance
2. Note SSH details (IP, user, key path)
3. Run `poetry run cloud-train --provider ssh ...`
4. Enter details when prompted

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
