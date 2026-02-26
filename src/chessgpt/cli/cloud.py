"""CLI: Run training and evaluation on cloud GPUs.

Wraps the existing chessgpt-train and chessgpt-eval commands, running them
on a remote GPU instance via SSH. Training launches in a detached tmux session
so the local machine can disconnect without killing the run.

Usage:
    chessgpt-cloud train --provider runpod --gpu A100 --config configs/large.toml --name large_v1
    chessgpt-cloud status
    chessgpt-cloud attach
    chessgpt-cloud download
    chessgpt-cloud deprovision [--no-download]
    chessgpt-cloud eval  --provider runpod --gpu A100 --name large_v1
    chessgpt-cloud list-gpus --provider runpod [--gpu A100]
"""

from __future__ import annotations

import argparse
import sys

from chessgpt.cloud.pricing import estimate_cost, format_cost_estimate
from chessgpt.cloud.providers import get_provider, list_providers
from chessgpt.cloud.runner import (
    _detect_config_size,
    cloud_attach,
    cloud_deprovision,
    cloud_download,
    cloud_status,
    run_cloud_eval,
    run_cloud_train,
)


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


def _cmd_status(args: argparse.Namespace) -> None:
    """Handle the 'status' subcommand."""
    cloud_status()


def _cmd_attach(args: argparse.Namespace) -> None:
    """Handle the 'attach' subcommand."""
    cloud_attach()


def _cmd_download(args: argparse.Namespace) -> None:
    """Handle the 'download' subcommand."""
    cloud_download(output_dir=args.output_dir)


def _cmd_deprovision(args: argparse.Namespace) -> None:
    """Handle the 'deprovision' subcommand."""
    cloud_deprovision(download=not args.no_download, output_dir=args.output_dir)


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

    # status
    subparsers.add_parser("status", help="Check training progress on the active pod")

    # attach
    subparsers.add_parser("attach", help="Attach to live training output (tmux)")

    # download
    download_parser = subparsers.add_parser("download", help="Download results from the active pod")
    download_parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Local base output directory (default: out)",
    )

    # deprovision
    deprovision_parser = subparsers.add_parser("deprovision", help="Tear down the active pod")
    deprovision_parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help="Skip downloading results before terminating",
    )
    deprovision_parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Local base output directory (default: out)",
    )

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
        elif args.command == "status":
            _cmd_status(args)
        elif args.command == "attach":
            _cmd_attach(args)
        elif args.command == "download":
            _cmd_download(args)
        elif args.command == "deprovision":
            _cmd_deprovision(args)
        elif args.command == "list-gpus":
            _cmd_list_gpus(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
