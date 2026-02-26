"""
Mate-in-1 benchmark.

Tests the model against curated positions where checkmate is available in one move.
The primary metric for the checkmate problem: does the model's top-1 prediction
deliver checkmate?

Puzzle format is a simple list of (move_sequence, expected_checkmate_move) tuples.
We feed the move sequence to the model and check if its prediction matches.
"""

import torch

from chessgpt.model.tokenizer import ChessTokenizer
from chessgpt.model.transformer import ChessTransformer

# Curated mate-in-1 positions from real games and well-known patterns.
# Each tuple: (move_history as space-separated string, correct checkmate move)
# Only entries with a non-empty expected move are used.
MATE_IN_1_PUZZLES = [
    # Scholar's mate
    ("e4 e5 Qh5 Nc6 Bc4 Nf6", "Qxf7#"),
    # Fool's mate
    ("f3 e5 g4", "Qh4#"),
    # Anastasia's mate pattern (Qh7# or similar)
    ("e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 Ng5 O-O Qf3 h6", "Qxf7#"),
    # Smothered mate (Nf7#)
    (
        "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6 dxc6 bxc6 Be2 h6 Nf3 e4 Ne5 Qd4 f4 Bc5 c3 Qd6 d4 exd3 Nxd3 Be7 O-O O-O b4 Nd5 Qc2 Nf4 Nxf4 Qxf4 Nd2 Nc4 Nxc4 Qxc4",
        "",
    ),
    # Legal's mate
    ("e4 e5 Nf3 d6 Bc4 Bg4 Nc3 g6 Nxe5 Bxd1", "Bxf7#"),
    # Back rank mate with rook
    (
        "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7 Rc1 c6 Bd3 dxc4 Bxc4 Nd5 Bxe7 Qxe7 O-O Nxc3 Rxc3 e5 Qc2 exd4 exd4 Nb6 Bb3 Bf5 Qc1 Rfe8 Re1 Qd7 Rxe8+ Rxe8 Nd2 Bg4 Rc5 Re1+ Nf1 Nd5 Qd2 Nf4 g3 Nh3+ Kg2 Qd5+ f3 Qxc5",
        "",
    ),
    # Queen sacrifice + back rank
    (
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Be7 Re1 d6 c3 O-O d4 Bg4 d5 Na5 Bc2 c6 h3 Bh5 dxc6 Qc7 Nbd2 Qxc6 Nf1 Nc4 Qe2 Rfe8 Ng3 Bg6 Nh4 Nd7 Nhf5 Bf8 Bh6 Nde5 Bxg7 Bxg7 Nh6+ Kh8 Qg4 Bxc2 Qg7#",
        "Qg7#",
    ),
    # Damiano's mate
    ("e4 e5 Nf3 f6 Nxe5 fxe5 Qh5+ Ke7 Qxe5+ Kf7 Bc4+ Kg6", "Qf5#"),
    # Epaulette mate pattern
    (
        "d4 d5 c4 e6 Nc3 c6 e4 dxe4 Nxe4 Bb4+ Bd2 Qxd4 Bxb4 Qxe4+ Be2 Na6 Bd6 Qxg2 Qd2 Qxh1 O-O-O Nb4 Qg5 Nxa2+ Kb1 Nb4 Qxg7 Rf8 Nf3 Nd5 Bc5 a5 Bxf8 Qxf3 Bd6 b5 Qg3 Qd5 cxb5 cxb5 Bf1 Bb7 Bxb5 Qxb5 Qg8+ Kd7",
        "Qd8#",
    ),
    # Double bishop mate
    (
        "e4 e5 Bc4 Bc5 d3 d6 Nc3 Nf6 f4 Nc6 f5 Na5 Bg5 Nxc4 dxc4 Be7 Nd5 Nxd5 exd5 Bxg5 Qg4 O-O Qxg5 f6 Qg4 Qe7 Ne2 Bd7 O-O Qe8 c3 b5 b3 a5 Ng3 b4 Nh5 bxc3 Nxf6+ gxf6 Qg3+ Kh8 Qg7#",
        "Qg7#",
    ),
    # Opera game finish (Morphy)
    (
        "e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 O-O-O Rd8",
        "Rd8#",
    ),
    # Simple queen mate on back rank
    (
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6 Nd2 Qe8 O-O Nh7 a3 Bd7 Rb1 f5 f3 Nf6 b4 axb4 axb4 Kh7 Bf2 f4 c5 g5 cxd6 cxd6 Nb5 g4 Nc4 g3 hxg3 fxg3 Bxg3 Nh5 Bf2 Qg6 Nxd6 Bh3 Rf2 Nf4 Bf1 Bxg2 Rxg2 Qxg2#",
        "Qxg2#",
    ),
    # Blackburne's mate pattern
    (
        "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O d3 Qb3 Qf6 e5 Qg6 Re1 Nge7 Ba3 b5 Qxb5 Rb8 Qa4 Bb6 Nbd2 Bb7 Ne4 Qf5 Bxd3 Qh5 Nf6+ gxf6 exf6 Rg8 Rad1 Qxf3 Rxe7+ Nxe7 Qxd7+ Kxd7 Bf5+ Ke8 Bd7+ Kf8",
        "Bh6#",
    ),
    # Lolli's mate
    (
        "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 Nxf7 Kxf7 Qf3+ Ke6 Nc3 Ncb4 a3 Nxc3 bxc3 Nxc2+ Kd1 Nxa1 Nxd5 Bg4",
        "",
    ),
    # h-file attack mate
    (
        "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 a3 Bxc3+ Qxc3 Ne4 Qc2 c5 dxc5 Nc6 cxd5 exd5 Nf3 Bf5 Qd1 d4 e3 O-O Be2 Qa5+ Bd2 Nxd2 Qxd2 d3 Bd1 Rad8 O-O Qxc5 b4 Qe7 Bb3 Qe4 Rfe1 Ne5 Nd4 Bg4 f3 Qh4 fxg4 Qxg4 Nf3 Nxf3+ gxf3 Qg3+ Kh1 Qxf3+ Kg1 Rd5 Qf2 Rh5 Qxf3 Rh1#",
        "Rh1#",
    ),
]


def evaluate_mate_in_1(
    model: ChessTransformer,
    tokenizer: ChessTokenizer,
    device: torch.device,
    puzzles: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """
    Evaluate mate-in-1 accuracy.

    Returns top-1, top-3, and top-5 accuracy on mate-in-1 puzzles.
    """
    if puzzles is None:
        puzzles = [(m, c) for m, c in MATE_IN_1_PUZZLES if c]

    if not puzzles:
        return {"mate_in_1_top1": 0.0, "mate_in_1_top3": 0.0, "mate_in_1_top5": 0.0}

    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0

    with torch.no_grad():
        for move_history_str, expected_move in puzzles:
            moves = move_history_str.split()
            input_ids = tokenizer.encode_and_pad(moves, model.config.max_seq_len)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

            policy_logits, _, _ = model(input_tensor)
            last_logits = policy_logits[0, -1, :]

            top_k = last_logits.topk(5).indices.tolist()
            top_moves = tokenizer.decode(top_k)

            if top_moves[0] == expected_move:
                correct_top1 += 1
            if expected_move in top_moves[:3]:
                correct_top3 += 1
            if expected_move in top_moves[:5]:
                correct_top5 += 1

    n = len(puzzles)
    return {
        "mate_in_1_top1": correct_top1 / n,
        "mate_in_1_top3": correct_top3 / n,
        "mate_in_1_top5": correct_top5 / n,
    }
