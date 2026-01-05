"""Instance management CLI."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from .remote import (
    ssh_interactive,
    get_tmux_output,
    check_training_status,
    load_deployments
)
from .sync import retrieve_trained_model
from .providers import get_provider, Instance, SSHConfig


def load_instance_from_deployment(instance_id: str) -> Instance:
    """Load instance from deployment history."""
    deployments = load_deployments()

    # Find deployment by instance_id
    deployment = next((d for d in deployments if d.get("instance_id") == instance_id), None)

    if not deployment:
        raise ValueError(f"No deployment found for instance: {instance_id}")

    # Reconstruct instance (this is a simplified version - in production would need full state)
    # For now, just use SSH provider with saved state
    print(f"Instance ID: {deployment['instance_id']}")
    print(f"Model: {deployment['model_name']}")
    print(f"Started: {deployment['started_at']}")

    # This is a placeholder - real implementation would load full instance state
    raise NotImplementedError("Instance state loading not yet fully implemented")


def cmd_list(args):
    """List active deployments."""
    deployments = load_deployments()

    if not deployments:
        print("No deployments found")
        return

    print("\n Active Deployments:")
    print("-" * 80)
    print(f"{'Instance ID':<20} {'Model':<25} {'Started':<20} {'Status':<10}")
    print("-" * 80)

    for deployment in deployments:
        instance_id = deployment.get("instance_id", "N/A")
        model_name = deployment.get("model_name", "N/A")
        started_at = deployment.get("started_at", "N/A")
        status = deployment.get("status", "unknown")

        # Format timestamp
        if started_at != "N/A":
            try:
                dt = datetime.fromisoformat(started_at)
                started_at = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass

        print(f"{instance_id:<20} {model_name:<25} {started_at:<20} {status:<10}")

    print()


def cmd_ssh(args):
    """Open SSH session to instance."""
    print(f"Opening SSH session to {args.instance_id}...")
    print("\n⚠️  Instance state management not yet implemented")
    print("For now, use: ssh -i <key> <user>@<host>")
    print("\nTo attach to training session:")
    print(f"  tmux attach -t chessgpt-<model-name>")


def cmd_monitor(args):
    """Monitor training progress."""
    print(f"Monitoring instance {args.instance_id}...")
    print("\n⚠️  Instance state management not yet implemented")
    print("For now, SSH to instance and run:")
    print(f"  tmux attach -t chessgpt-<model-name>")


def cmd_pull(args):
    """Download trained model from instance."""
    print(f"Retrieving model from instance {args.instance_id}...")
    print("\n⚠️  Instance state management not yet implemented")
    print("For now, use rsync manually:")
    print(f"  rsync -avz <user>@<host>:~/ChessGPT/models/<model-name>/ ./models/<model-name>/")


def cmd_stop(args):
    """Stop/terminate instance."""
    instance_id = args.instance_id

    print(f"Stopping instance {instance_id}...")

    if not args.force:
        response = input(f"\n⚠️  This will terminate instance {instance_id}. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    print("\n⚠️  Instance state management not yet implemented")
    print("For now, terminate instance manually via provider dashboard")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage cloud GPU instances",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    parser_list = subparsers.add_parser("list", help="List active deployments")
    parser_list.set_defaults(func=cmd_list)

    # SSH command
    parser_ssh = subparsers.add_parser("ssh", help="Open SSH session")
    parser_ssh.add_argument("instance_id", help="Instance ID")
    parser_ssh.set_defaults(func=cmd_ssh)

    # Monitor command
    parser_monitor = subparsers.add_parser("monitor", help="Monitor training progress")
    parser_monitor.add_argument("instance_id", help="Instance ID")
    parser_monitor.set_defaults(func=cmd_monitor)

    # Pull command
    parser_pull = subparsers.add_parser("pull", help="Download trained model")
    parser_pull.add_argument("instance_id", help="Instance ID")
    parser_pull.set_defaults(func=cmd_pull)

    # Stop command
    parser_stop = subparsers.add_parser("stop", help="Terminate instance")
    parser_stop.add_argument("instance_id", help="Instance ID")
    parser_stop.add_argument("--force", action="store_true", help="Skip confirmation")
    parser_stop.set_defaults(func=cmd_stop)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
