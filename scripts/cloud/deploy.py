"""Main cloud deployment CLI."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .providers import get_provider, Instance
from .remote import (
    setup_remote_environment,
    clone_or_rsync_project,
    install_dependencies,
    start_training_in_tmux,
    attach_to_tmux,
    check_training_status,
    load_deployments
)
from .sync import sync_training_data, retrieve_trained_model


def deploy_full_workflow(
    provider: str,
    gpu_type: str,
    config_name: str,
    model_name: str,
    prepare_data: bool = False,
    pgn_url: Optional[str] = None,
    auto_shutdown: bool = False
) -> Instance:
    """
    End-to-end deployment workflow.

    Steps:
    1. Provision GPU instance
    2. Setup environment (Python, Poetry, dependencies)
    3. Sync code and data
    4. Start training in tmux
    5. (Optional) Wait and retrieve model
    """
    print("=" * 60)
    print("ChessGPT Cloud Deployment")
    print("=" * 60)

    # Step 1: Provision
    print("\n1. Provisioning GPU instance...")
    print(f"   Provider: {provider}")
    print(f"   GPU: {gpu_type}")

    provider_obj = get_provider(provider)
    instance = provider_obj.provision(gpu_type)

    print(f"   ✓ Instance: {instance.id}")
    print(f"   ✓ IP: {instance.ip}")
    print(f"   ✓ Cost: ${instance.price_per_hour:.2f}/hr")

    try:
        # Step 2: Setup environment
        print("\n2. Setting up environment...")
        setup_remote_environment(instance)
        print("   ✓ System packages installed")
        print("   ✓ Poetry installed")
        print("   ✓ GPU verified")

        # Step 3: Transfer code
        print("\n3. Syncing project files...")
        clone_or_rsync_project(instance, Path.cwd())

        # Step 4: Install dependencies
        print("\n4. Installing dependencies...")
        install_dependencies(instance)

        # Step 5: Sync data (or prepare on cloud)
        if prepare_data:
            print("\n5. Preparing data on cloud...")
            if not pgn_url:
                raise ValueError("--pgn-url required when using --prepare-data")
            # TODO: Implement cloud data preparation in future phase
            raise NotImplementedError("Cloud data preparation not yet implemented. Use pre-processed data for now.")
        else:
            print("\n5. Syncing training data...")
            sync_training_data(instance, config_name)

        # Step 6: Start training
        print("\n6. Starting training...")
        print(f"   Config: {config_name}")
        print(f"   Model name: {model_name}")

        session_name = start_training_in_tmux(instance, config_name, model_name)

        print(f"\n{'=' * 60}")
        print("✓ Training started successfully!")
        print(f"{'=' * 60}")
        print(f"\nInstance: {instance.id}")
        print(f"IP: {instance.ip}")
        print(f"Tmux session: {session_name}")
        print(f"\nTo monitor training:")
        print(f"  poetry run cloud-instances monitor {instance.id}")
        print(f"\nTo attach to tmux:")
        print(f"  poetry run cloud-instances ssh {instance.id}")
        print(f"  tmux attach -t {session_name}")
        print(f"\nTo retrieve trained model:")
        print(f"  poetry run cloud-instances pull {instance.id}")
        print(f"\nTo terminate instance:")
        print(f"  poetry run cloud-instances stop {instance.id}")

        if auto_shutdown:
            print("\nAuto-shutdown enabled - will wait for training completion")
            print("Press Ctrl+C to cancel and leave instance running")
            # TODO: Implement wait and auto-shutdown in future phase

        return instance

    except Exception as e:
        print(f"\n❌ Error during deployment: {e}")
        print("\nCleaning up...")
        try:
            provider_obj.terminate(instance)
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed: {cleanup_error}")
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy ChessGPT training to cloud GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy to Lambda Labs A100
  poetry run cloud-train --provider lambda --gpu A100 --config phase1_gpt2_medium --name my-model

  # Deploy to custom SSH server
  poetry run cloud-train --provider ssh --gpu "RTX 4090" --config micro_test_gpt2_medium --name test

  # With data preparation (future)
  poetry run cloud-train --provider lambda --gpu A100 --config phase1_gpt2_medium --name my-model \\
      --prepare-data --pgn-url https://database.lichess.org/standard/lichess_db_standard_rated_2024-06.pgn.zst
        """
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=["lambda", "ssh"],
        help="Cloud provider (lambda=Lambda Labs, ssh=custom SSH server)"
    )

    parser.add_argument(
        "--gpu",
        required=True,
        help="GPU type (e.g., A100, RTX 4090)"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Training config name (from configs/pipeline/)"
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Model name for output directory"
    )

    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare data on cloud (download PGN, process)"
    )

    parser.add_argument(
        "--pgn-url",
        help="PGN download URL (required with --prepare-data)"
    )

    parser.add_argument(
        "--auto-shutdown",
        action="store_true",
        help="Automatically terminate instance after training completes"
    )

    args = parser.parse_args()

    try:
        deploy_full_workflow(
            provider=args.provider,
            gpu_type=args.gpu,
            config_name=args.config,
            model_name=args.name,
            prepare_data=args.prepare_data,
            pgn_url=args.pgn_url,
            auto_shutdown=args.auto_shutdown
        )
    except KeyboardInterrupt:
        print("\n\nDeployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
