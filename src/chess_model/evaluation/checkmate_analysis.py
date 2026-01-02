"""
Checkmate analysis utilities for ChessGPT.

This module provides functions to:
1. Detect mate-in-N positions (N = 1, 2, 3)
2. Find the best mating move in a position
3. Evaluate model performance on mate puzzles
"""
from typing import Optional, List, Tuple
import chess
import chess.engine


def is_checkmate_position(board: chess.Board) -> bool:
    """
    Check if the current position is checkmate.

    Args:
        board: Chess board

    Returns:
        True if position is checkmate, False otherwise
    """
    return board.is_checkmate()


def find_mate_in_n(
    board: chess.Board,
    max_depth: int = 3,
    time_limit: float = 1.0
) -> Tuple[Optional[int], Optional[chess.Move]]:
    """
    Detect if position has mate-in-N and find the mating move.

    Uses minimax search with alpha-beta pruning to detect forced checkmate.

    Args:
        board: Chess board to analyze
        max_depth: Maximum depth to search (1, 2, or 3)
        time_limit: Time limit for search in seconds

    Returns:
        Tuple of (mate_in_n, best_move) where:
        - mate_in_n: Number of moves to mate (None if no mate found)
        - best_move: Best mating move (None if no mate found)
    """
    # Try each depth from 1 to max_depth
    for depth in range(1, max_depth + 1):
        mate_move = _search_mate_at_depth(board, depth)
        if mate_move is not None:
            return depth, mate_move

    return None, None


def _search_mate_at_depth(board: chess.Board, depth: int) -> Optional[chess.Move]:
    """
    Search for mate at a specific depth using minimax.

    Args:
        board: Chess board
        depth: Exact depth to search

    Returns:
        Mating move if found, None otherwise
    """
    if depth <= 0:
        return None

    for move in board.legal_moves:
        board.push(move)

        # If this move leads to immediate checkmate, we found mate-in-1
        if depth == 1 and board.is_checkmate():
            board.pop()
            return move

        # For deeper mates, check if opponent has no good response
        if depth > 1:
            is_forced_mate = True
            for opponent_move in board.legal_moves:
                board.push(opponent_move)

                # Recursively check if we can force mate after opponent's response
                mate_move = _search_mate_at_depth(board, depth - 2)

                board.pop()

                # If opponent has any move that doesn't lead to mate, not forced
                if mate_move is None:
                    is_forced_mate = False
                    break

            board.pop()

            if is_forced_mate and depth > 1:
                # Check if opponent has any legal moves
                board.push(move)
                has_moves = len(list(board.legal_moves)) > 0
                board.pop()

                if has_moves:  # Only return if not immediate checkmate
                    return move
        else:
            board.pop()

    return None


def find_mate_in_one(board: chess.Board) -> Optional[chess.Move]:
    """
    Find mate-in-one move if it exists.

    Args:
        board: Chess board

    Returns:
        Mating move or None
    """
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()

        if is_mate:
            return move

    return None


def find_mate_in_two(board: chess.Board) -> Optional[chess.Move]:
    """
    Find mate-in-two move if it exists.

    Uses simple minimax: for each candidate move, check if all opponent
    responses can be met with checkmate.

    Args:
        board: Chess board

    Returns:
        First move of mate-in-two sequence or None
    """
    for move in board.legal_moves:
        board.push(move)

        # Check if opponent has any move that avoids mate-in-1
        all_responses_lose = True
        for opponent_move in board.legal_moves:
            board.push(opponent_move)

            # Check if we have mate-in-1 after opponent's move
            mate_move = find_mate_in_one(board)

            board.pop()

            if mate_move is None:
                all_responses_lose = False
                break

        board.pop()

        if all_responses_lose and len(list(board.legal_moves)) > 0:
            return move

    return None


def analyze_position(board: chess.Board) -> dict:
    """
    Analyze a position for checkmate patterns.

    Args:
        board: Chess board

    Returns:
        Dictionary with analysis results:
        {
            'is_checkmate': bool,
            'mate_in_1': Optional[chess.Move],
            'mate_in_2': Optional[chess.Move],
            'mate_in_n': Optional[int],
            'best_mating_move': Optional[chess.Move],
        }
    """
    result = {
        'is_checkmate': board.is_checkmate(),
        'mate_in_1': None,
        'mate_in_2': None,
        'mate_in_n': None,
        'best_mating_move': None,
    }

    if result['is_checkmate']:
        return result

    # Check mate-in-1
    mate_in_1 = find_mate_in_one(board)
    if mate_in_1:
        result['mate_in_1'] = mate_in_1
        result['mate_in_n'] = 1
        result['best_mating_move'] = mate_in_1
        return result

    # Check mate-in-2
    mate_in_2 = find_mate_in_two(board)
    if mate_in_2:
        result['mate_in_2'] = mate_in_2
        result['mate_in_n'] = 2
        result['best_mating_move'] = mate_in_2
        return result

    # For deeper mates, could extend but expensive
    # mate_in_n, best_move = find_mate_in_n(board, max_depth=3)
    # if mate_in_n:
    #     result['mate_in_n'] = mate_in_n
    #     result['best_mating_move'] = best_move

    return result


def evaluate_model_on_puzzles(
    model,
    tokenizer,
    puzzles: List[Tuple[str, str]],
    top_k: int = 5
) -> dict:
    """
    Evaluate model's checkmate-finding ability on puzzle positions.

    Args:
        model: Trained ChessGPT model
        tokenizer: Chess tokenizer
        puzzles: List of (fen, solution_move) tuples
        top_k: Consider top-k model predictions

    Returns:
        Dictionary with evaluation metrics:
        {
            'total_puzzles': int,
            'mate_in_1_correct': int,
            'mate_in_1_accuracy': float,
            'mate_in_2_correct': int,
            'mate_in_2_accuracy': float,
            'top_1_accuracy': float,
            'top_k_accuracy': float,
        }
    """
    results = {
        'total_puzzles': len(puzzles),
        'mate_in_1_correct': 0,
        'mate_in_1_total': 0,
        'mate_in_2_correct': 0,
        'mate_in_2_total': 0,
        'top_1_correct': 0,
        'top_k_correct': 0,
    }

    for fen, solution_move in puzzles:
        board = chess.Board(fen)
        analysis = analyze_position(board)

        # Count puzzles by type
        if analysis['mate_in_1']:
            results['mate_in_1_total'] += 1
        elif analysis['mate_in_2']:
            results['mate_in_2_total'] += 1

        # Get model predictions (this would need to be implemented)
        # For now, placeholder
        # model_predictions = model.predict(board, top_k=top_k)

        # Check if solution is in top-k predictions
        # if solution_move in model_predictions[:1]:
        #     results['top_1_correct'] += 1
        # if solution_move in model_predictions[:top_k]:
        #     results['top_k_correct'] += 1

    # Calculate accuracies
    if results['mate_in_1_total'] > 0:
        results['mate_in_1_accuracy'] = (
            results['mate_in_1_correct'] / results['mate_in_1_total']
        )

    if results['mate_in_2_total'] > 0:
        results['mate_in_2_accuracy'] = (
            results['mate_in_2_correct'] / results['mate_in_2_total']
        )

    if results['total_puzzles'] > 0:
        results['top_1_accuracy'] = (
            results['top_1_correct'] / results['total_puzzles']
        )
        results['top_k_accuracy'] = (
            results['top_k_correct'] / results['total_puzzles']
        )

    return results


def extract_checkmate_positions_from_games(
    game_file: str,
    output_file: str,
    max_games: Optional[int] = None
) -> None:
    """
    Extract positions before checkmate from game file.

    Args:
        game_file: Input file with games (one per line)
        output_file: Output file for FEN positions
        max_games: Maximum games to process (None for all)
    """
    import chess.pgn

    checkmate_positions = []
    games_processed = 0

    with open(game_file, 'r') as f:
        for line in f:
            if max_games and games_processed >= max_games:
                break

            # Parse game
            moves = line.strip().split()
            if not moves:
                continue

            # Remove outcome
            if moves[-1] in ['1-0', '0-1', '1/2-1/2']:
                moves = moves[:-1]

            # Play through game to find checkmate positions
            board = chess.Board()
            for i, move_san in enumerate(moves):
                try:
                    move = board.parse_san(move_san)
                    board.push(move)

                    # Check if this led to checkmate
                    if board.is_checkmate():
                        # Save position before checkmate
                        board.pop()
                        checkmate_positions.append({
                            'fen': board.fen(),
                            'solution': move_san,
                            'moves_played': i,
                        })
                        break
                except Exception:
                    # Invalid move, skip game
                    break

            games_processed += 1

    # Save to file
    import json
    with open(output_file, 'w') as f:
        json.dump(checkmate_positions, f, indent=2)

    print(f"Extracted {len(checkmate_positions)} checkmate positions from {games_processed} games")
    print(f"Saved to: {output_file}")
