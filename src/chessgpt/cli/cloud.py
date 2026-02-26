"""CLI: Run training and evaluation on cloud GPUs.

Wraps the existing chessgpt-train and chessgpt-eval commands, running them
on a remote GPU instance via SSH. The training code stays untouched — this
module only handles provisioning, file sync, and remote execution.

Usage:
    chessgpt-cloud train --provider runpod --gpu A100 --config configs/large.toml --name large_v1
    chessgpt-cloud eval  --provider runpod --gpu A100 --name large_v1
    chessgpt-cloud list-gpus --provider runpod [--gpu A100]
"""

from __future__ import annotations

import argparse
import sys

from chessgpt.cloud.pricing import estimate_cost, format_cost_estimate
from chessgpt.cloud.providers import get_provider, list_providers
from chessgpt.cloud.runner import _detect_config_size, run_cloud_eval, run_cloud_train


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across train/eval subcommands."""
    available = ", ".join(list_providers())
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help=f"Cloud provider ({available})",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        required=True,
        help="GPU type (e.g. A100, RTX_4090, H100)",
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--disk-gb", type=int, default=50, help="Disk space in GB (default: 50)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Local base output directory (default: out)",
    )


def _cmd_train(args: argparse.Namespace) -> None:
    """Handle the 'train' subcommand."""
    provider = get_provider(args.provider)

    # Show cost estimate if available
    config_size = _detect_config_size(args.config)
    est = estimate_cost(args.gpu, config_size)
    if est:
        print(format_cost_estimate(est))
        print()

    run_cloud_train(
        config_path=args.config,
        experiment_name=args.name,
        provider=provider,
        gpu_type=args.gpu,
        gpu_count=args.gpu_count,
        disk_gb=args.disk_gb,
        output_dir=args.output_dir,
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    """Handle the 'eval' subcommand."""
    provider = get_provider(args.provider)

    run_cloud_eval(
        experiment_name=args.name,
        provider=provider,
        gpu_type=args.gpu,
        gpu_count=args.gpu_count,
        disk_gb=args.disk_gb,
        output_dir=args.output_dir,
    )


def _cmd_list_gpus(args: argparse.Namespace) -> None:
    """Handle the 'list-gpus' subcommand."""
    provider = get_provider(args.provider)
    gpu_filter = getattr(args, "gpu", None)
    offers = provider.list_gpu_offers(gpu_type=gpu_filter)

    if not offers:
        print(f"No GPU offers found on {provider.name}")
        if gpu_filter:
            print(f"  (filtered by: {gpu_filter})")
        return

    print(f"\n{'GPU Type':<35} {'VRAM':>6} {'$/hr':>8} {'Available':>10}")
    print("-" * 65)
    for offer in offers:
        vram = f"{offer.vram_gb}GB" if offer.vram_gb else "?"
        avail = "yes" if offer.available else "no"
        print(f"{offer.gpu_type:<35} {vram:>6} {offer.cost_per_hour:>7.2f} {avail:>10}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser. Extracted for testability."""
    parser = argparse.ArgumentParser(
        prog="chessgpt-cloud",
        description="Run ChessGPT training/evaluation on cloud GPUs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train a model on a cloud GPU")
    _add_common_args(train_parser)
    train_parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    train_parser.add_argument("--name", type=str, required=True, help="Experiment name")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model on a cloud GPU")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--name", type=str, required=True, help="Experiment name to evaluate")

    # list-gpus
    list_parser = subparsers.add_parser("list-gpus", help="List available GPUs from a provider")
    available = ", ".join(list_providers())
    list_parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help=f"Cloud provider ({available})",
    )
    list_parser.add_argument("--gpu", type=str, default=None, help="Filter by GPU type")

    return parser


def _load_dotenv() -> None:
    """Load .env file from the project root if it exists."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def main() -> None:
    _load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            _cmd_train(args)
        elif args.command == "eval":
            _cmd_eval(args)
        elif args.command == "list-gpus":
            _cmd_list_gpus(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
