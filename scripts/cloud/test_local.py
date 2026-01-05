"""Local testing utilities for cloud deployment."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .providers import CloudProvider, Instance, SSHConfig


class LocalTestProvider(CloudProvider):
    """
    Mock provider for local testing.
    Uses localhost SSH to simulate cloud deployment.
    """

    def __init__(self):
        self.instances = {}

    def provision(self, gpu_type: str, region: Optional[str] = None) -> Instance:
        """
        Create mock instance pointing to localhost.
        """
        print("\n=== Local Test Provider ===")
        print("Creating mock instance (localhost)\n")

        # Check if SSH to localhost is possible
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "echo test"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("❌ Cannot SSH to localhost")
            print("\nTo enable localhost SSH testing:")
            print("  1. Enable SSH: System Settings → General → Sharing → Remote Login")
            print("  2. Or set up passwordless SSH:")
            print("     ssh-keygen -t rsa -f ~/.ssh/id_rsa (if you don't have a key)")
            print("     cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys")
            print("     chmod 600 ~/.ssh/authorized_keys")
            sys.exit(1)

        print("✓ Localhost SSH working")

        import getpass
        import socket

        ssh_config = SSHConfig(
            host="localhost",
            user=getpass.getuser(),
            port=22,
            key_file=None  # Use default SSH config
        )

        instance_id = f"local-test-{gpu_type.lower().replace(' ', '-')}"

        instance = Instance(
            id=instance_id,
            provider="local-test",
            gpu_type=gpu_type,
            ip="127.0.0.1",
            ssh_config=ssh_config,
            price_per_hour=0.0,  # Free!
            status="running"
        )

        self.instances[instance_id] = instance

        print(f"\n✓ Mock instance created: {instance_id}")
        print(f"  SSH: {ssh_config.user}@localhost")
        print(f"  Cost: $0.00/hr (local testing)")
        print()

        return instance

    def get_ssh_config(self, instance: Instance) -> SSHConfig:
        """Get SSH config."""
        return instance.ssh_config

    def terminate(self, instance: Instance) -> None:
        """Mock termination."""
        print(f"\n✓ Mock instance {instance.id} 'terminated' (no actual resources)")
        if instance.id in self.instances:
            del self.instances[instance.id]

    def get_pricing(self, gpu_type: str) -> float:
        """Free for local testing."""
        return 0.0


def test_ssh_connection():
    """Test basic SSH connectivity to localhost."""
    print("Testing SSH connection to localhost...")

    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "echo 'SSH working'"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ SSH to localhost working")
        return True
    else:
        print("❌ SSH to localhost failed")
        print("\nSetup instructions:")
        print("  1. Enable Remote Login in System Settings")
        print("  2. Or set up SSH keys:")
        print("     ssh-keygen -t rsa -f ~/.ssh/id_rsa")
        print("     cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys")
        return False


def test_rsync():
    """Test rsync functionality."""
    import tempfile
    import shutil

    print("\nTesting rsync...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")

        # Test rsync to localhost
        dest = Path(tmpdir) / "dest"
        dest.mkdir()

        result = subprocess.run(
            ["rsync", "-avz", str(test_file), f"localhost:{dest}/"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and (dest / "test.txt").exists():
            print("✓ rsync working")
            return True
        else:
            print("❌ rsync failed")
            return False


def test_remote_command_execution():
    """Test executing commands via SSH."""
    print("\nTesting remote command execution...")

    commands = [
        ("echo test", "Basic echo"),
        ("python3 --version", "Python version"),
        ("which poetry || echo 'Poetry not found'", "Poetry check"),
    ]

    for cmd, desc in commands:
        result = subprocess.run(
            ["ssh", "localhost", cmd],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"✓ {desc}: {output}")
        else:
            print(f"⚠️  {desc}: Failed")


def test_deployment_dry_run():
    """
    Dry run test - shows what would happen without executing.
    """
    print("\n" + "="*60)
    print("Deployment Dry Run Test")
    print("="*60)

    print("\nThis would:")
    print("  1. ✓ Provision instance (localhost)")
    print("  2. ✓ SSH to localhost")
    print("  3. ✓ Rsync project files")
    print("  4. ✓ Run: poetry install --with model")
    print("  5. ✓ Rsync training data")
    print("  6. ✓ Start training in tmux")
    print("\nAll commands would execute on localhost for testing.")


def main():
    """Run local tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test cloud deployment locally")
    parser.add_argument(
        "--test",
        choices=["ssh", "rsync", "commands", "all", "dry-run"],
        default="all",
        help="Which test to run"
    )

    args = parser.parse_args()

    print("="*60)
    print("Cloud Deployment Local Testing")
    print("="*60)

    if args.test in ["all", "dry-run"]:
        test_deployment_dry_run()

    if args.test == "dry-run":
        print("\nRun 'poetry run cloud-test' for actual localhost tests")
        return

    if args.test in ["ssh", "all"]:
        if not test_ssh_connection():
            sys.exit(1)

    if args.test in ["rsync", "all"]:
        if not test_rsync():
            sys.exit(1)

    if args.test in ["commands", "all"]:
        test_remote_command_execution()

    if args.test == "all":
        print("\n" + "="*60)
        print("✓ All local tests passed!")
        print("="*60)
        print("\nYou can now test full deployment with:")
        print("  poetry run cloud-train --provider local-test --gpu test --config micro_test_gpt2_medium --name test-local")


if __name__ == "__main__":
    main()
