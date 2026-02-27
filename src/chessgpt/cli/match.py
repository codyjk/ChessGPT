"""CLI: Run automated model-vs-engine matches."""

import argparse
from pathlib import Path

from chessgpt.cli.eval import load_model
from chessgpt.evaluation.engine_match import run_match
from chessgpt.training.trainer import get_device


def main():
    parser = argparse.ArgumentParser(description="Match ChessGPT against a UCI engine")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--engine", type=str, required=True, help="Path to UCI engine binary")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--engine-time", type=float, default=0.5, help="Seconds per engine move")
    parser.add_argument("--engine-elo", type=int, default=None, help="Set UCI_Elo if supported")
    parser.add_argument("--temperature", type=float, default=0.3, help="Model sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Model top-k sampling")
    parser.add_argument(
        "--retry-illegal",
        action="store_true",
        help="On illegal move, try next-best move instead of forfeiting",
    )
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model(args.model, device)
    model_dir = Path(args.model).parent
    model_name = f"ChessGPT ({model_dir.name})"
    print(f"Loaded model: d_model={model.config.d_model}, {model.config.n_layers} layers")
    print(f"Engine: {args.engine}")
    print(f"Games: {args.games}, engine time: {args.engine_time}s")
    if args.engine_elo:
        print(f"Engine ELO limit: {args.engine_elo}")
    print()

    summary = run_match(
        model=model,
        tokenizer=tokenizer,
        engine_path=args.engine,
        num_games=args.games,
        engine_time=args.engine_time,
        engine_elo=args.engine_elo,
        temperature=args.temperature,
        top_k=args.top_k,
        retry_illegal=args.retry_illegal,
        output_dir=model_dir,
        model_name=model_name,
    )

    print(f"\n{summary}")
    print(f"\nPGNs saved to: {model_dir / 'matches'}")
