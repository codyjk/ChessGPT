"""Remote execution utilities for cloud instances."""

import subprocess
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import json

from .providers import Instance, SSHConfig


def build_ssh_command(ssh_config: SSHConfig, command: Optional[str] = None) -> list[str]:
    """Build SSH command with proper options."""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_config.port),
    ]

    if ssh_config.key_file:
        ssh_cmd.extend(["-i", ssh_config.key_file])

    ssh_cmd.append(f"{ssh_config.user}@{ssh_config.host}")

    if command:
        ssh_cmd.append(command)

    return ssh_cmd


def ssh_exec(
    instance: Instance,
    command: str,
    capture_output: bool = True,
    check: bool = True
) -> subprocess.CompletedProcess:
    """Execute command on remote instance via SSH."""
    ssh_cmd = build_ssh_command(instance.ssh_config, command)

    result = subprocess.run(
        ssh_cmd,
        capture_output=capture_output,
        text=True,
        check=check
    )

    return result


def ssh_interactive(instance: Instance) -> None:
    """Open interactive SSH session."""
    ssh_cmd = build_ssh_command(instance.ssh_config)
    subprocess.run(ssh_cmd)


def verify_gpu_available(instance: Instance) -> bool:
    """Verify GPU is accessible on instance."""
    try:
        result = ssh_exec(instance, "nvidia-smi", capture_output=True, check=False)
        return result.returncode == 0
    except Exception:
        return False


def setup_remote_environment(instance: Instance) -> None:
    """Setup Python environment on remote instance."""
    print("Installing system dependencies...")

    # Update and install basics
    ssh_exec(instance, """
        sudo apt-get update &&
        sudo apt-get install -y python3-pip git wget unzip zstd
    """)

    # Install Poetry
    print("Installing Poetry...")
    ssh_exec(instance, """
        curl -sSL https://install.python-poetry.org | python3 - &&
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    """)

    # Verify GPU
    print("Verifying GPU...")
    if not verify_gpu_available(instance):
        raise RuntimeError("GPU not available on instance")

    result = ssh_exec(instance, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"✓ GPU detected: {result.stdout.strip()}")


def clone_or_rsync_project(instance: Instance, local_path: Path) -> None:
    """Transfer project to remote instance."""
    print("Transferring project files...")

    # Create remote directory
    ssh_exec(instance, "mkdir -p ~/ChessGPT")

    # Use rsync for efficient transfer
    from .sync import rsync_files
    rsync_files(instance, [
        "src/",
        "scripts/",
        "configs/",
        "pyproject.toml",
        "poetry.lock",
        "CLAUDE.md"
    ], remote_path="~/ChessGPT")

    print("✓ Project files synced")


def install_dependencies(instance: Instance) -> None:
    """Install Python dependencies via Poetry."""
    print("Installing Python dependencies (this may take 5-10 minutes)...")

    ssh_exec(instance, """
        cd ~/ChessGPT &&
        export PATH="$HOME/.local/bin:$PATH" &&
        poetry install --with model
    """)

    print("✓ Dependencies installed")


def start_training_in_tmux(
    instance: Instance,
    config_name: str,
    model_name: str,
    output_dir: str = "models"
) -> str:
    """Start training in detached tmux session."""
    session_name = f"chessgpt-{model_name}"

    # Build training command (identical to local)
    train_cmd = f"""
        cd ~/ChessGPT &&
        export PATH="$HOME/.local/bin:$PATH" &&
        export WANDB_MODE=offline &&
        poetry run train \\
            --config {config_name} \\
            --name {model_name} \\
            --output-dir {output_dir}
    """

    # Kill existing session if it exists
    ssh_exec(instance, f"tmux kill-session -t {session_name} 2>/dev/null || true", check=False)

    # Start new tmux session
    ssh_exec(instance, f"tmux new-session -d -s {session_name} '{train_cmd}'")

    # Save deployment state
    save_deployment({
        "instance_id": instance.id,
        "model_name": model_name,
        "config_name": config_name,
        "tmux_session": session_name,
        "started_at": datetime.now().isoformat(),
        "status": "training"
    })

    return session_name


def attach_to_tmux(instance: Instance, session_name: str) -> None:
    """Attach to tmux session interactively."""
    ssh_cmd = build_ssh_command(instance.ssh_config, f"tmux attach -t {session_name}")
    subprocess.run(ssh_cmd)


def get_tmux_output(instance: Instance, session_name: str, lines: int = 50) -> str:
    """Get recent output from tmux session."""
    result = ssh_exec(instance, f"tmux capture-pane -t {session_name} -p | tail -{lines}")
    return result.stdout


def check_training_status(instance: Instance, session_name: str) -> dict:
    """Check if training is still running."""
    # Check if tmux session exists
    result = ssh_exec(instance, f"tmux has-session -t {session_name} 2>/dev/null", check=False)
    session_exists = result.returncode == 0

    if not session_exists:
        return {"running": False, "status": "completed or crashed"}

    # Get recent output
    output = get_tmux_output(instance, session_name, lines=100)

    # Parse for completion indicators
    completed = any(phrase in output for phrase in [
        "Training complete",
        "✓ Training complete",
        "Model saved to"
    ])

    # Parse for errors
    errored = any(phrase in output for phrase in [
        "Error:",
        "Traceback",
        "CUDA out of memory",
        "RuntimeError"
    ])

    return {
        "running": not (completed or errored),
        "completed": completed,
        "errored": errored,
        "last_output": output[-500:] if output else ""  # Last 500 chars
    }


def save_deployment(deployment_data: dict) -> None:
    """Save deployment state to local .cloudgpt directory."""
    deployments_file = Path(".cloudgpt/deployments.json")
    deployments_file.parent.mkdir(exist_ok=True)

    # Load existing deployments
    if deployments_file.exists():
        with open(deployments_file) as f:
            deployments = json.load(f)
    else:
        deployments = []

    # Add new deployment
    deployments.append(deployment_data)

    # Save
    with open(deployments_file, "w") as f:
        json.dump(deployments, f, indent=2)


def load_deployments() -> list[dict]:
    """Load deployment history."""
    deployments_file = Path(".cloudgpt/deployments.json")
    if not deployments_file.exists():
        return []

    with open(deployments_file) as f:
        return json.load(f)
