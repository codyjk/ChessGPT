#!/bin/bash
# Helper script to play against a trained model

MODEL_NAME="${1:-gpt2-test-5k}"
COLOR="${2:-white}"

# Model paths
MODEL_DIR="models/$MODEL_NAME/final_model"
MODEL_FILE="$MODEL_DIR/model.pth"
TOKENIZER_FILE="$MODEL_DIR/tokenizer.json"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo "Available models:"
    ls -d models/*/final_model 2>/dev/null | sed 's|models/||; s|/final_model||'
    exit 1
fi

# Model hyperparameters (from gpt2-medium config)
MAX_CONTEXT=100
NUM_EMBED=1024
NUM_LAYERS=24
NUM_HEADS=16

echo "Playing against: $MODEL_NAME"
echo "Your color: $COLOR"
echo ""

poetry run play \
    --input-model-file "$MODEL_FILE" \
    --input-tokenizer-file "$TOKENIZER_FILE" \
    --max-context-length $MAX_CONTEXT \
    --num-embeddings $NUM_EMBED \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --color "$COLOR"
