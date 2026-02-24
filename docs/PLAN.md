# ChessGPT v2 Project Plan

## Problem Statement

v1 wrapped HuggingFace's `GPT2Model` as a black box (37 lines). It learned "what moves do winning players make?" but treated every move equally -- the checkmate-delivering move got the same gradient signal as move 1 (`d4`). The model had zero checkmate-seeking behavior.

## Core Design Decisions

- **Pure AI inference**: No `python-chess` at runtime, ever. Legal move rate is a first-class metric.
- **Checkmate head is training-only**: Improves backbone representations but doesn't trigger runtime search. Diagnostics only during play.
- **Game size**: Most checkmates happen in middlegame/endgame (moves 25-60+), context lengths set accordingly (60-200 depending on tier).
- **Pre-commit hooks**: ruff lint + format + pytest on every commit.
- **Learning-focused structure**: attention.py, layers.py, heads.py, transformer.py -- each concept standalone.

---

## Phase 0: Clean Slate

Branch off from the last trusted commit (`5580d99`, Nov 2024) and build forward. The original core -- tokenizer, dataset, PGN parsing -- is solid and battle-tested. Keep that foundation and rebuild everything else.

**What we keep from v1**:
- PGN parsing with regex patterns
- `ChessTokenizer` class
- mmap-based CSV loading (adapted for new format)

**What we delete**: Everything from the 5 AI-generated commits that added ~10K lines of mostly dead/broken code (Llama/LoRA wrappers, cloud deployment stubs, 30+ unused YAML configs, duplicate test scripts).

---

## Phase 1: Tooling Modernization

| Change | From | To | Why |
|--------|------|----|-----|
| Package manager | Poetry | uv | 10-100x faster, simpler |
| Linting/formatting | black + isort + flake8 | ruff | One tool, milliseconds |
| Config | Hydra/YAML/Pydantic | TOML | Simple, no magic |
| Experiment tracking | wandb | JSONL | Local-first, no dependencies |

**Dependency cleanup**: Remove `transformers`, `hydra-core`, `omegaconf`, `pydantic`, `wandb`, `peft`, `accelerate`. Keep `torch`, `python-chess`, `tqdm`, `numpy`. Add `zstandard`.

**Project structure**:
```
src/chessgpt/
  model/          # Transformer architecture (attention, layers, heads, tokenizer)
  data/           # Dataset, download, data preparation
  training/       # Training loop + multi-task loss
  evaluation/     # Metrics + mate-in-1 puzzles
  inference/      # Pure AI move selection
  cli/            # CLI commands (download, prepare, train, eval, play)
  pgn/            # PGN parsing utilities
configs/          # tiny.toml, small.toml, medium.toml, large.toml
tests/            # 5 test files covering all modules
experiments/      # Structured experiment logs (JSONL)
```

---

## Phase 2: The Custom Chess Transformer

### Architecture: Modern Decoder-Only Transformer

Same family as GPT-2 but with 2024/2025 improvements (Llama, Gemma, Qwen style):

**`attention.py` -- Multi-head causal self-attention with RoPE**

RoPE (Rotary Position Embeddings) replaces GPT-2's learned position embeddings. Instead of a lookup table, it rotates Q/K vectors by position-proportional angles. Two advantages for chess:
1. **Relative position**: attention depends on distance (`i - j`), not absolute position
2. **Length generalization**: no hard limit on sequence length

**`layers.py` -- TransformerBlock with RMSNorm + SwiGLU FFN**

Each block: attention sub-layer + FFN sub-layer, with:
- **RMSNorm** instead of LayerNorm (simpler, ~15% faster)
- **SwiGLU** instead of GELU (gated activation, consistently better since Llama 1)
- **Pre-norm** instead of post-norm (more stable training)

```python
x = x + self.attention(self.attn_norm(x))   # pre-norm + residual
x = x + self.ffn(self.ffn_norm(x))          # pre-norm + residual
```

**`heads.py` -- Three chess-specific output heads**

| Head | Task | Shape | Loss |
|------|------|-------|------|
| **Policy** | Next move prediction | `[batch, seq, vocab]` | Cross-entropy (weighted at checkmate positions) |
| **Value** | Game outcome (W/D/B) | `[batch, 3]` | Cross-entropy |
| **Checkmate** | Checkmate available? | `[batch, 1]` | BCE with logits (training signal only) |

### Why This Fixes Checkmate

- **Value head**: teaches the model that positions near checkmate are extreme (near-certain win) -- develops an internal sense of "keep pushing"
- **Checkmate head**: directly teaches "a checkmate move exists here" -- model learns mating patterns
- **Policy head with checkmate weighting**: 5x loss on the final mating move

### Model Sizes

| Config | d_model | layers | heads | ~params | Use case |
|--------|---------|--------|-------|---------|----------|
| tiny   | 128     | 4      | 4     | 2M      | Rapid iteration (seconds) |
| small  | 256     | 8      | 8     | 15M     | Validation (minutes) |
| medium | 512     | 12     | 8     | 50M     | Real training (hours) |
| large  | 1024    | 24     | 16    | 300M    | Cloud GPU only |

---

## Phase 3: Training Strategy

### Multi-Task Loss

```
total_loss = policy_loss + alpha * value_loss + beta * checkmate_loss
```

Starting weights: alpha=0.5, beta=0.5.

**Policy loss**: cross-entropy at every position, with outcome-based masking (train on winner's moves for decisive games) and checkmate move weighting (5x for the mating move).

**Value loss**: cross-entropy at the last position against game outcome [white=0, draw=1, black=2].

**Checkmate loss**: BCE with logits at the last position, binary target.

### Multi-Move Checkmate Learning

The value head creates a gradient the policy head can follow:
- 3 moves before mate: "white is ~90% winning"
- 2 moves before mate: "white is ~97% winning"
- 1 move before mate: "white is ~99.5% winning"

### Training Loop

Clean PyTorch loop (no HuggingFace Trainer): AdamW, cosine LR schedule with warmup, gradient clipping, per-epoch checkpointing. Speed optimizations: tqdm progress bars, mixed precision autocast, gradient accumulation, optional torch.compile on CUDA.

---

## Phase 4: Data Pipeline

### Download
Lichess monthly databases (`.pgn.zst`). Stream, decompress with `zstandard`, write PGN.

### Processing: PGN to Enriched CSV
1. Parse and filter (1800+ Elo, valid results)
2. Annotate checkmate positions with `python-chess` (data prep only)
3. Write CSV: `moves,outcome,checkmate_move_idx,ply_count`

### Data Scaling

| Stage | Games | Use case |
|-------|-------|----------|
| Dev | 10K-20K | Tiny model iteration |
| Validation | 50K-100K | Small model |
| Full | 270K+ | Medium/large training |

---

## Phase 5: Inference -- Pure AI

```python
move history -> tokenize -> model forward -> sample from logits -> done
```

No legal move filtering, no board tracking, no retry loop. If the model outputs an illegal move, that's a failure we measure and improve on.

Value and checkmate heads at inference are diagnostics only -- displayed during play but don't influence move selection.

---

## Phase 6: Evaluation & Scaling

### Evaluation Suite

1. **Move prediction accuracy**: top-1 and top-5 on held-out games
2. **Legal move rate**: by game phase (opening, middlegame, endgame)
3. **Mate-in-1 benchmark**: curated puzzles, primary metric for checkmate problem

### Gate Metrics

| Metric | Gate | Target |
|--------|------|--------|
| `mate_in_1_top1` | Primary | Higher is better |
| `legal_move_rate` | Must pass | >80% |

### Scaling Decision Tree

```
Tiny -> eval -> mate>0% & legal>50%?
  |yes
Small -> eval -> metrics improved?
  |yes
Medium -> eval -> legal>95% & mate>50%?
  |yes
Large (cloud GPU)
```

Each step is a gate. Don't scale without evidence that things are working.

---

## Training Results Log

### Tiny (2M params)
- **tiny_v3** (10 epochs, 270K games, 38 min on MPS):
  - Legal: **82%** (opening 96%, mid 71%, endgame 0%)
  - Mate-in-1: **33%** (4/12 puzzles)
  - Move accuracy: 22% top-1, 46% top-5
  - Saturated around epoch 5-7
  - **Decision**: passed gates -> scale to small

### Small (8.5M params)
- **small_v1** (no dropout, lr=1e-4): overfitted at epoch 5
- **small_v2** (dropout=0.1, lr=3e-4, 15 epochs, 4.8h):
  - Legal: **91%** (opening 99%, mid 88%, endgame 77%)
  - Mate-in-1: **42%** (5/12 puzzles)
  - Move accuracy: 31% top-1, 62% top-5
  - **Decision**: improvements transfer -> scale to medium

### Key Training Insights
- Always use dropout (0.1 minimum) -- prevents severe overfitting
- lr=3e-4 works better than 1e-4 for these model sizes
- Expanded puzzle set from 2 to 12 effective tests
