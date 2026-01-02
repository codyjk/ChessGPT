# ChessGPT Examples

This directory contains example scripts demonstrating how to use ChessGPT 2026.

## Training Examples

### `train_with_huggingface.py`

Complete example showing how to train ChessGPT using the HuggingFace Trainer infrastructure.

**Features demonstrated:**
- Loading configuration from YAML files
- Creating model and tokenizer
- Loading datasets with outcome-based masking
- Using ChessTrainer with custom callbacks
- Evaluation and model saving

**Usage:**
```bash
# Make sure data is prepared first
poetry run python examples/train_with_huggingface.py
```

**What it does:**
1. Loads configs from `configs/` directory
2. Initializes GPT-2 baseline model
3. Loads training and validation datasets
4. Trains with:
   - Outcome-based masking (learns from winning player)
   - Chess-specific metrics (move accuracy, legal move rate)
   - Automatic checkpointing and early stopping
   - WandB logging (if configured)
5. Saves final model to `outputs/gpt2_baseline/final_model/`

## Configuration

All examples use the configuration system in `configs/`:
- `configs/model/` - Model architectures (GPT-2, Llama)
- `configs/training/` - Training hyperparameters (Phase 1, Phase 2)
- `configs/data/` - Data paths and preprocessing
- `configs/pipeline/` - Full pipeline configs

## Customization

### Change Model Architecture

Edit the config loading in the example:
```python
# Use Llama instead of GPT-2
model_config = OmegaConf.load("configs/model/llama_1b.yaml")
```

### Adjust Training Parameters

Edit `configs/training/phase1_general.yaml`:
```yaml
num_epochs: 10  # More epochs
batch_size: 128  # Larger batch
learning_rate: 1e-4  # Different LR
```

### Add Custom Callbacks

```python
from chess_model.training import CheckmateMetricsCallback

callbacks = [
    CheckmateMetricsCallback(log_every_n_steps=100),
    # Your custom callback here
]

trainer = ChessTrainer(..., callbacks=callbacks)
```

## Next Steps

After running the training example:
1. Check logs in `outputs/gpt2_baseline/`
2. View metrics in WandB (if configured)
3. Test the model with `poetry run play`
4. Evaluate on test set

See the main README.md for more details.
