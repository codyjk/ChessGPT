# CLAUDE.md

## Project Overview

ChessGPT v2 is a custom-built decoder-only transformer that learns chess from move sequences. It has three output heads (policy, value, checkmate) to develop checkmate-seeking behavior. The model plays without python-chess at inference — legal move rate is a first-class metric.

## Quick Start

```bash
make                          # Install all dependencies
source .venv/bin/activate     # Put chessgpt-* commands on PATH
make lint                     # Lint
make format                   # Format
make test                     # Test
make check                    # Lint + format check + tests (CI-style)
```

## Training Pipeline

```bash
# 1. Download Lichess PGN
chessgpt-download --year 2013 --month 1 --output-dir data/

# 2. Process PGN → enriched CSV + fit tokenizer
chessgpt-prepare --input-pgn data/lichess_db_standard_rated_2013-01.pgn \
  --output-csv data/train.csv --fit-tokenizer data/tokenizer.json

# 3. Train (always start with tiny config)
chessgpt-train --config configs/tiny.toml --name experiment_v1

# 4. Evaluate
chessgpt-eval --model out/experiment_v1/model.pt

# 5. Play
chessgpt-play --model out/experiment_v1/model.pt
```

## Cloud Training

For medium/large models that exceed local GPU capacity. Requires `RUNPOD_API_KEY` or `VASTAI_API_KEY`.

Training launches in a detached tmux session on the remote GPU so your local machine can disconnect (laptop sleep, network blip) without killing the run.

```bash
# List available GPUs and pricing
chessgpt-cloud list-gpus --provider runpod

# Launch training (returns in ~5 min after provision + upload + install)
chessgpt-cloud train --provider runpod --gpu A100 \
  --config configs/medium.toml --name medium_v1

# Close laptop, go for a walk...

# Check progress
chessgpt-cloud status

# Watch live tqdm output (Ctrl+B, D to detach)
chessgpt-cloud attach

# Download results
chessgpt-cloud download

# Tear down pod (auto-downloads first)
chessgpt-cloud deprovision            # or --no-download to skip

# Evaluate on a cloud GPU (still blocking -- eval is fast)
chessgpt-cloud eval --provider runpod --gpu A100 --name medium_v1
```

One active pod per project. State persists in `.cloud/pod.json`.

## Data Pipeline (AWS Lambda)

For processing large Lichess datasets that exceed local disk. Two Lambda functions (download + prepare) share one Docker image. Infra is in `infra/` (Terraform).

```bash
# Process a single month via Lambda
chessgpt-download --year 2017 --month 1 --cloud --bucket <bucket>
chessgpt-prepare --year 2017 --month 1 --cloud --bucket <bucket>

# After all months complete, merge locally
chessgpt-prepare --merge-from-s3 --year 2017 --bucket <bucket> \
  --output-csv data/train_large.csv --fit-tokenizer data/tokenizer_large.json

# Deploy infra
make deploy          # docker build + push + terraform apply
make logs            # tail both Lambda log groups
```

**Lambda limits**: 15-minute timeout, 10GB `/tmp`. Months up to ~4GB compressed (.zst) work reliably -- this covers 2013-2017. Months from 2020+ (5-20GB compressed) will exceed `/tmp` or timeout. For those, use a GPU pod via `chessgpt-cloud` instead.

## Autonomous Iteration

1. Always start with `configs/tiny.toml` (seconds per run)
2. Train: `uv run chessgpt-train --config configs/tiny.toml --name <name>`
3. Eval: `uv run chessgpt-eval --model out/<name>/model.pt`
4. Compare: read `experiments/log.jsonl`
5. Only scale to `small.toml` after tiny shows improvement

**Primary metric**: `mate_in_1_top1` (higher is better)
**Gate metric**: `legal_move_rate` (must be >80%)

## Architecture

Custom decoder-only transformer with modern internals:
- **attention.py** — Multi-head causal self-attention with RoPE
- **layers.py** — TransformerBlock with RMSNorm + SwiGLU FFN
- **heads.py** — Policy (next move), Value (game outcome), Checkmate (detection)
- **transformer.py** — Full model assembly

Three output heads:
- **Policy**: `[batch, seq, vocab]` — next move prediction at every position
- **Value**: `[batch, 3]` — game outcome (white/draw/black)
- **Checkmate**: `[batch, 1]` — checkmate availability (training signal only)

## Model Configs

| Config | d_model | layers | heads | ~params | Use case |
|--------|---------|--------|-------|---------|----------|
| tiny   | 128     | 4      | 4     | 2M      | Rapid iteration (seconds) |
| small  | 256     | 8      | 8     | 15M     | Validation (minutes) |
| medium | 512     | 12     | 8     | 50M     | Real training (hours) |
| large  | 1024    | 24     | 16    | 300M    | Cloud GPU only |

## Key Design Decisions

- **No python-chess at inference** — model plays on its own
- **Outcome-based masking** — train on winner's moves only for decisive games
- **Checkmate weighting** — 5x loss weight for checkmate-delivering moves
- **RoPE** — no learned position embeddings, generalizes to longer sequences
- **Pre-norm** — RMSNorm before each sublayer (more stable than GPT-2's post-norm)
- **mmap dataset** — memory-efficient random access for large CSV files
- **num_workers=0** — macOS MPS multiprocessing is buggy

## Project Structure

```
src/chessgpt/
├── model/          # Transformer architecture (attention, layers, heads, tokenizer)
├── data/           # Dataset, download, data preparation
├── training/       # Training loop + multi-task loss
├── evaluation/     # Metrics + mate-in-1 puzzles
├── inference/      # Pure AI move selection
├── cli/            # CLI commands (download, prepare, train, eval, play, cloud)
├── cloud/          # Cloud GPU training (provider ABC, SSH, runner, state, pricing)
│   └── providers/  # Concrete backends (runpod, vastai)
├── lambdas/        # AWS Lambda handlers for data pipeline
└── pgn/            # PGN parsing utilities
```
