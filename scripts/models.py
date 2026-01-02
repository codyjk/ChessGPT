"""
Model registry CLI for ChessGPT.

Manage trained models with versioning, metadata, and easy discovery.

Commands:
- list: List all registered models
- show: Show details for a specific model
- compare: Compare two models
- tag: Add tags to a model
- recommend: Set a model as recommended
- delete: Remove a model from registry
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional


REGISTRY_PATH = Path("models/registry.json")


def load_registry() -> dict:
    """Load model registry."""
    if not REGISTRY_PATH.exists():
        return {}

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(registry: dict):
    """Save model registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def list_models(args):
    """List all registered models."""
    registry = load_registry()

    if not registry:
        print("No models registered yet.")
        print("\nRegister a model by training with:")
        print("  poetry run train --config <config> --name <model-name>")
        return

    print("=" * 80)
    print("Registered Models")
    print("=" * 80)

    # Sort by creation date (newest first)
    models = sorted(
        registry.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True
    )

    for name, info in models:
        recommended = " [RECOMMENDED]" if info.get("recommended") else ""
        print(f"\n{name}{recommended}")
        print(f"  Architecture: {info.get('architecture', 'unknown')}")
        print(f"  Created: {info.get('created_at', 'unknown')}")

        if "training_time" in info:
            print(f"  Training time: {info['training_time']}")

        if "metrics" in info:
            metrics = info["metrics"]
            if "eval_loss" in metrics:
                print(f"  Eval loss: {metrics['eval_loss']:.4f}")
            if "move_accuracy" in metrics:
                print(f"  Move accuracy: {metrics['move_accuracy']:.2%}")

        if "tags" in info and info["tags"]:
            print(f"  Tags: {', '.join(info['tags'])}")

    print("\n" + "=" * 80)
    print(f"Total: {len(registry)} models")


def show_model(args):
    """Show details for a specific model."""
    registry = load_registry()

    if args.name not in registry:
        print(f"Model not found: {args.name}")
        print(f"\nAvailable models:")
        for name in registry.keys():
            print(f"  - {name}")
        sys.exit(1)

    info = registry[args.name]

    print("=" * 80)
    print(f"Model: {args.name}")
    print("=" * 80)

    print(f"\nGeneral:")
    print(f"  Architecture: {info.get('architecture', 'unknown')}")
    print(f"  Path: {info.get('path', 'unknown')}")
    print(f"  Created: {info.get('created_at', 'unknown')}")
    print(f"  Recommended: {info.get('recommended', False)}")

    if "training_time" in info:
        print(f"  Training time: {info['training_time']}")

    if "tags" in info and info["tags"]:
        print(f"  Tags: {', '.join(info['tags'])}")

    if "hyperparams" in info:
        print(f"\nHyperparameters:")
        for key, value in info["hyperparams"].items():
            if value is not None:
                print(f"  {key}: {value}")

    if "metrics" in info:
        print(f"\nMetrics:")
        for key, value in info["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print(f"\nUse with:")
    print(f"  poetry run play --model {args.name}")
    print(f"  poetry run explore --model {args.name}")


def compare_models(args):
    """Compare two models."""
    registry = load_registry()

    if args.model1 not in registry:
        print(f"Model not found: {args.model1}")
        sys.exit(1)

    if args.model2 not in registry:
        print(f"Model not found: {args.model2}")
        sys.exit(1)

    info1 = registry[args.model1]
    info2 = registry[args.model2]

    print("=" * 80)
    print(f"Comparing: {args.model1} vs {args.model2}")
    print("=" * 80)

    # Compare architectures
    print(f"\nArchitecture:")
    print(f"  {args.model1}: {info1.get('architecture', 'unknown')}")
    print(f"  {args.model2}: {info2.get('architecture', 'unknown')}")

    # Compare hyperparameters
    if "hyperparams" in info1 and "hyperparams" in info2:
        print(f"\nHyperparameters:")
        all_params = set(info1["hyperparams"].keys()) | set(info2["hyperparams"].keys())
        for param in sorted(all_params):
            val1 = info1["hyperparams"].get(param, "N/A")
            val2 = info2["hyperparams"].get(param, "N/A")
            diff_marker = " ✗" if val1 != val2 else ""
            print(f"  {param}:")
            print(f"    {args.model1}: {val1}")
            print(f"    {args.model2}: {val2}{diff_marker}")

    # Compare metrics
    if "metrics" in info1 and "metrics" in info2:
        print(f"\nMetrics:")
        all_metrics = set(info1["metrics"].keys()) | set(info2["metrics"].keys())
        for metric in sorted(all_metrics):
            val1 = info1["metrics"].get(metric)
            val2 = info2["metrics"].get(metric)

            if val1 is not None and val2 is not None:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    diff_pct = (diff / abs(val1)) * 100 if val1 != 0 else 0
                    better = "↑" if diff > 0 else "↓" if diff < 0 else "="

                    print(f"  {metric}:")
                    print(f"    {args.model1}: {val1:.4f}")
                    print(f"    {args.model2}: {val2:.4f} ({better} {diff:+.4f}, {diff_pct:+.1f}%)")
                else:
                    print(f"  {metric}:")
                    print(f"    {args.model1}: {val1}")
                    print(f"    {args.model2}: {val2}")

    # Compare training time
    if "training_time" in info1 and "training_time" in info2:
        print(f"\nTraining Time:")
        print(f"  {args.model1}: {info1['training_time']}")
        print(f"  {args.model2}: {info2['training_time']}")


def tag_model(args):
    """Add tag to a model."""
    registry = load_registry()

    if args.name not in registry:
        print(f"Model not found: {args.name}")
        sys.exit(1)

    if "tags" not in registry[args.name]:
        registry[args.name]["tags"] = []

    if args.tag not in registry[args.name]["tags"]:
        registry[args.name]["tags"].append(args.tag)
        save_registry(registry)
        print(f"✓ Added tag '{args.tag}' to {args.name}")
    else:
        print(f"Tag '{args.tag}' already exists on {args.name}")


def recommend_model(args):
    """Set a model as recommended."""
    registry = load_registry()

    if args.name not in registry:
        print(f"Model not found: {args.name}")
        sys.exit(1)

    # Clear recommended flag from all models
    for model_info in registry.values():
        model_info["recommended"] = False

    # Set recommended flag on target model
    registry[args.name]["recommended"] = True

    save_registry(registry)
    print(f"✓ Set {args.name} as recommended model")


def delete_model(args):
    """Remove a model from registry."""
    registry = load_registry()

    if args.name not in registry:
        print(f"Model not found: {args.name}")
        sys.exit(1)

    if not args.force:
        response = input(f"Delete {args.name} from registry? (y/N): ")
        if response.lower() != "y":
            print("Cancelled")
            return

    del registry[args.name]
    save_registry(registry)
    print(f"✓ Removed {args.name} from registry")
    print("\nNote: Model files still exist on disk. To delete them:")
    print(f"  rm -rf models/{args.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Model registry CLI for ChessGPT"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List all models")
    list_parser.set_defaults(func=list_models)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show model details")
    show_parser.add_argument("name", help="Model name")
    show_parser.set_defaults(func=show_model)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model1", help="First model")
    compare_parser.add_argument("model2", help="Second model")
    compare_parser.set_defaults(func=compare_models)

    # Tag command
    tag_parser = subparsers.add_parser("tag", help="Add tag to model")
    tag_parser.add_argument("name", help="Model name")
    tag_parser.add_argument("tag", help="Tag to add")
    tag_parser.set_defaults(func=tag_model)

    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Set recommended model")
    recommend_parser.add_argument("name", help="Model name")
    recommend_parser.set_defaults(func=recommend_model)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete model from registry")
    delete_parser.add_argument("name", help="Model name")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=delete_model)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
