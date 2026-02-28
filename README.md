# ChessGPT

ChessGPT is a custom-built transformer that learns to play chess from move sequences. It has no knowledge of the rules of chess. It learned the game by studying millions of strong games from Lichess (1800+ Elo), and it plays purely from what it learned.

The model has three output heads that work together:

- **Policy**: predicts the next move
- **Value**: predicts the game outcome (white win, draw, black win)
- **Checkmate**: detects when a checkmate move is available

The value and checkmate heads shape the shared transformer backbone so the policy head doesn't just imitate moves. It learns to play toward winning positions and deliver checkmate.

## Setup

```bash
git clone <repo-url>
cd ChessGPT
make
source .venv/bin/activate
```

This installs all dependencies via [uv](https://github.com/astral-sh/uv) and puts the `chessgpt-*` commands on your PATH.

## Playing

There are no pretrained models included in the repo. You need to train one first. The fastest path is to download a month of Lichess data, process it, train a tiny model, and play:

```bash
# Download a month of Lichess games (~10 min)
chessgpt-download --year 2013 --month 1 --output-dir data/

# Process into training data and fit the tokenizer (~5 min)
chessgpt-prepare \
  --input-pgn data/lichess_db_standard_rated_2013-01.pgn \
  --output-csv data/train.csv \
  --fit-tokenizer data/tokenizer.json

# Train a tiny model (~30 sec on GPU, ~5 min on CPU)
chessgpt-train --config configs/tiny.toml --name my_first

# Play!
chessgpt-play --model out/my_first/model.pt
```

This drops you into a chess game in the terminal. You enter moves in algebraic notation (e.g. `e4`, `Nf3`, `Bxe5`). After each model move you'll see diagnostics: value estimates, checkmate probability, and the top predicted moves with their probabilities.

The tiny model plays recognizable chess but starts making mistakes after the opening. Train the small model for a better opponent:

```bash
chessgpt-train --config configs/small.toml --name my_small
chessgpt-play --model out/my_small/model.pt
```

Options for `chessgpt-play`:

- `--color white|black` (default: white)
- `--temperature 0.3` (lower = more deterministic)
- `--top-k 5` (number of candidate moves)
- `--no-save` (skip saving the game as PGN)

## Evaluation

```bash
chessgpt-eval --model out/my_first/model.pt
```

This runs the full evaluation suite: move prediction accuracy, legal move rate by game phase, and mate-in-1 puzzles.

## Model Sizes

| Config | d_model | Layers | Heads | Params | Use case |
|--------|---------|--------|-------|--------|----------|
| tiny   | 128     | 4      | 4     | 2M     | Quick iteration, seconds per run |
| small  | 256     | 8      | 8     | 15M    | Validation, minutes per run |
| medium | 512     | 12     | 8     | 50M    | Real training, hours per run |
| large  | 1024    | 24     | 16    | 300M   | Cloud GPU only |

## Cloud Training

For medium and large models that exceed local GPU capacity. Requires `RUNPOD_API_KEY` or `VASTAI_API_KEY`.

Training runs in a detached tmux session on the remote GPU, so your local machine can disconnect without killing the run.

```bash
# See available GPUs and pricing
chessgpt-cloud list-gpus --provider runpod

# Launch training
chessgpt-cloud train --provider runpod --gpu A100 \
  --config configs/medium.toml --name medium_v1

# Check on it later
chessgpt-cloud status
chessgpt-cloud attach           # live output, Ctrl+B D to detach

# When it's done
chessgpt-cloud download
chessgpt-cloud deprovision
```

## Large-Scale Data Pipeline

For processing large Lichess datasets that exceed local disk, there's an AWS Lambda pipeline. Two Lambda functions (download + prepare) share one Docker image. Infrastructure is managed with Terraform in `infra/`.

```bash
# Process a single month via Lambda
chessgpt-download --year 2017 --month 1 --cloud --bucket <bucket>
chessgpt-prepare --year 2017 --month 1 --cloud --bucket <bucket>

# Merge all months locally
chessgpt-prepare --merge-from-s3 --year 2017 --bucket <bucket> \
  --output-csv data/train_large.csv --fit-tokenizer data/tokenizer_large.json
```

Lambda has a 15-minute timeout and 10 GB `/tmp`. Months up to ~4 GB compressed (2013-2017) work reliably. Larger months (2020+) need a GPU pod via `chessgpt-cloud` instead.

## Architecture

Custom decoder-only transformer with modern internals (Llama/Gemma style):

- **RoPE** for position encoding instead of learned embeddings
- **RMSNorm** before each sublayer (pre-norm)
- **SwiGLU** gated feed-forward network
- **Multi-task loss** across all three heads
- **Outcome-based masking** trains on the winner's moves for decisive games
- **Checkmate weighting** applies 5x loss on checkmate-delivering moves

At inference the model plays on its own. No legality filtering, no search. If it outputs an illegal move, that counts against the legal move rate metric. The model has to learn the rules from data.

## Results

The small model (8.5M params, dropout=0.1, 4.8h on MPS) achieves:

- Legal move rate: 91% (opening 99%, middlegame 88%, endgame 77%)
- Mate-in-1: 42% (5/12 puzzles)
- Move accuracy: 31% top-1, 62% top-5

## Project Structure

```
src/chessgpt/
├── model/          # Transformer (attention, layers, heads, tokenizer)
├── data/           # Dataset, download, data preparation
├── training/       # Training loop, multi-task loss
├── evaluation/     # Metrics, mate-in-1 puzzles
├── inference/      # Move selection
├── cli/            # CLI entry points
├── cloud/          # Cloud GPU training (RunPod, Vast.ai)
├── lambdas/        # AWS Lambda handlers for large-scale data processing
└── pgn/            # PGN parsing
```
