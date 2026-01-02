# Data Strategy for ChessGPT 2026

## Available Data

**Lichess Database:**
- File: `lichess_db_standard_rated_2016-04.pgn.zst`
- Size: 1.0GB compressed
- Games: ~6 million rated games
- Date: April 2016

## Training Strategy

### Phase 1: Small-Scale Validation (Initial Training)

**Data Subset:**
- **100,000 games maximum**
- **5-6 moves per game maximum**
- Focus on opening phase only
- ELO: 1800+ (master level and above)

**Rationale:**
- Quick iteration and testing
- Validate training pipeline end-to-end
- Fast convergence for initial model
- Low computational cost
- Easier to debug issues

**Processing Steps:**
1. Decompress `.zst` file
2. Parse PGN with python-chess
3. Filter:
   - Games with ELO ≥ 1800
   - Games with ≥ 5 moves
   - Truncate to first 5-6 moves
4. Generate 100K examples
5. Split: 90K train / 10K validation

### Phase 2: Medium-Scale Training (After Validation)

**Data Subset:**
- 500,000 games
- Up to 20 moves per game
- ELO: 1800+
- Include checkmate games separately

### Phase 3: Full-Scale Training (Production)

**Data Subset:**
- All 6M games
- Full game length (up to 100 moves)
- All ELO ranges (segmented)
- Full checkmate dataset from puzzles

## Quick Test Dataset

For rapid testing (Phase 1.4 validation):
- **10,000 games**
- **5 moves per game**
- **~30 minutes training time**
- Validate infrastructure works

## File Organization

```
data/
├── lichess_db_standard_rated_2016-04.pgn.zst  # Raw data (6M games)
├── processed/
│   ├── quick_test/
│   │   ├── training.csv         (9K games, 5 moves)
│   │   ├── validation.csv       (1K games, 5 moves)
│   │   └── tokenizer.json
│   ├── small_scale/
│   │   ├── training.csv         (90K games, 5-6 moves)
│   │   ├── validation.csv       (10K games, 5-6 moves)
│   │   └── tokenizer.json
│   └── full/
│       ├── general_training.csv (5.4M games, full length)
│       ├── checkmate_training.csv (500K checkmate positions)
│       ├── validation.csv       (600K games)
│       └── tokenizer.json
```

## Expected Model Behavior (Small-Scale)

With 100K games of 5-6 moves each:
- **Opening knowledge**: Strong (e4, d4, Nf3, etc.)
- **Middlegame**: Weak (not enough data)
- **Endgame**: Nonexistent (no data)
- **Checkmate**: Poor (insufficient examples)
- **Legal move rate**: >90% for openings
- **Top-1 accuracy**: 40-60% on opening moves

This is expected and acceptable for validation purposes.
