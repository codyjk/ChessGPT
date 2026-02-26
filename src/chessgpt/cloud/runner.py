"""Orchestrator for cloud training runs.

Manages the full lifecycle: provision → upload → install → train → download →
deprovision. The training code itself is untouched — we just run the existing
chessgpt-train and chessgpt-eval CLI commands on the remote machine over SSH.
"""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

from chessgpt.cloud import ssh
from chessgpt.cloud.pricing import format_cost_summary
from chessgpt.cloud.provider import CloudProvider, InstanceSpec

if TYPE_CHECKING:
    import paramiko

# Remote working directory on the GPU instance
REMOTE_PROJECT_DIR = "/root/chessgpt"

# Files and directories to upload. These are relative to the local project root.
UPLOAD_PATHS = [
    "src",
    "configs",
    "pyproject.toml",
]

# Data files uploaded separately (CSVs only, not raw PGNs)
DATA_GLOBS = ["data/*.csv", "data/*.json"]


def _detect_config_size(config_path: str) -> str:
    """Extract config size name from the file path (e.g. 'configs/tiny.toml' → 'tiny')."""
    return Path(config_path).stem


def _merge_jsonl(remote_lines: list[str], local_path: Path) -> None:
    """Merge remote log.jsonl entries into the local file, deduplicating by timestamp.

    Each JSONL line is a dict with a 'timestamp' field. We use this as a unique
    key to avoid duplicate entries when the same run is logged both locally and
    remotely.
    """
    existing_timestamps: set[str] = set()
    if local_path.exists():
        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = entry.get("timestamp", "")
                    if ts:
                        existing_timestamps.add(ts)
                except json.JSONDecodeError:
                    continue

    local_path.parent.mkdir(parents=True, exist_ok=True)
    new_entries = []
    for line in remote_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp", "")
            if ts and ts not in existing_timestamps:
                new_entries.append(line)
                existing_timestamps.add(ts)
        except json.JSONDecodeError:
            continue

    if new_entries:
        with open(local_path, "a") as f:
            for entry_line in new_entries:
                f.write(entry_line + "\n")
        print(f"  Merged {len(new_entries)} new entries into {local_path}")


def _resolve_data_files(project_root: Path) -> list[Path]:
    """Resolve data glob patterns to concrete file paths."""
    files: list[Path] = []
    for pattern in DATA_GLOBS:
        files.extend(project_root.glob(pattern))
    return files


def run_cloud_train(
    config_path: str,
    experiment_name: str,
    provider: CloudProvider,
    gpu_type: str,
    *,
    gpu_count: int = 1,
    disk_gb: int = 50,
    output_dir: str = "out",
    project_root: Path | None = None,
) -> Path:
    """Run a full training job on a cloud GPU.

    Orchestration sequence:
    1. Provision GPU instance
    2. Upload project source, configs, and data
    3. Install dependencies (uv)
    4. Run chessgpt-train
    5. Download trained model + logs
    6. Merge experiment logs
    7. Deprovision instance (always, via finally)

    On KeyboardInterrupt, attempts to download the current best checkpoint
    before deprovisioning — the trainer saves model.pt at each best-validation
    epoch, so we never lose more than one epoch of work.

    Args:
        config_path: Path to the TOML config file.
        experiment_name: Name for this experiment (used for output directory).
        provider: CloudProvider instance to use.
        gpu_type: GPU type to request (e.g. "A100", "RTX_4090").
        gpu_count: Number of GPUs to request.
        disk_gb: Disk space in GB.
        output_dir: Local base output directory.
        project_root: Local project root (auto-detected if None).

    Returns:
        Path to the local output directory containing the downloaded model.
    """
    if project_root is None:
        project_root = Path.cwd()

    spec = InstanceSpec(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        disk_gb=disk_gb,
    )

    config_size = _detect_config_size(config_path)
    local_output = Path(output_dir) / experiment_name
    instance_info = None
    client = None
    start_time = time.time()

    print(f"Cloud training: {experiment_name} ({config_size}) on {provider.name}/{gpu_type}")

    try:
        # Step 1: Provision
        print(f"\n[1/7] Provisioning {gpu_type} on {provider.name}...")
        instance_info = provider.provision(spec)
        print(
            f"  Instance {instance_info.instance_id} ready "
            f"({instance_info.host}:{instance_info.port})"
        )
        print(f"  Cost: ${instance_info.cost_per_hour:.2f}/hr")

        # Step 2: Connect
        print("\n[2/7] Connecting via SSH...")
        client = ssh.connect(
            host=instance_info.host,
            port=instance_info.port,
            username=instance_info.username,
            ssh_key_path=instance_info.ssh_key_path,
        )
        print("  Connected.")

        # Step 3: Upload
        print("\n[3/7] Uploading project files...")
        file_count = _upload_project(client, project_root)
        print(f"  Uploaded {file_count} files.")

        # Step 4: Install dependencies
        print("\n[4/7] Installing dependencies...")
        ssh.run_command(
            client,
            f"cd {REMOTE_PROJECT_DIR} && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "export PATH=$HOME/.local/bin:$PATH && "
            "uv sync --all-extras",
        )

        # Step 5: Train
        print(f"\n[5/7] Training {experiment_name}...")
        remote_output = f"{REMOTE_PROJECT_DIR}/out"
        ssh.run_command(
            client,
            f"cd {REMOTE_PROJECT_DIR} && "
            f"export PATH=$HOME/.local/bin:$PATH && "
            f"uv run chessgpt-train --config {shlex.quote(config_path)} "
            f"--name {shlex.quote(experiment_name)} "
            f"--output-dir {shlex.quote(remote_output)}",
        )

        # Step 6: Download results
        print(f"\n[6/7] Downloading results to {local_output}...")
        _download_results(client, experiment_name, local_output, project_root)

        # Step 7: Cost summary
        elapsed_hours = (time.time() - start_time) / 3600
        print("\n[7/7] Done!")
        print(format_cost_summary(gpu_type, elapsed_hours, instance_info.cost_per_hour))

    except KeyboardInterrupt:
        print("\n\nInterrupted! Attempting to save current checkpoint...")
        if client is not None and instance_info is not None:
            try:
                _download_results(client, experiment_name, local_output, project_root)
                print("  Checkpoint saved successfully.")
            except Exception as download_exc:
                print(f"  Could not download checkpoint: {download_exc}")
        raise

    finally:
        if client is not None:
            client.close()
        if instance_info is not None:
            print(f"\nDeprovisioning {instance_info.instance_id}...")
            provider.deprovision(instance_info.instance_id)
            print("  Instance terminated.")

    return local_output


def run_cloud_eval(
    experiment_name: str,
    provider: CloudProvider,
    gpu_type: str,
    *,
    gpu_count: int = 1,
    disk_gb: int = 50,
    output_dir: str = "out",
    project_root: Path | None = None,
) -> None:
    """Run evaluation on a cloud GPU for an already-trained model.

    Similar lifecycle to run_cloud_train, but uploads the trained model
    and runs chessgpt-eval instead.

    Args:
        experiment_name: Name of the experiment to evaluate.
        provider: CloudProvider instance to use.
        gpu_type: GPU type to request.
        gpu_count: Number of GPUs.
        disk_gb: Disk space in GB.
        output_dir: Local base output directory containing the model.
        project_root: Local project root (auto-detected if None).
    """
    if project_root is None:
        project_root = Path.cwd()

    spec = InstanceSpec(gpu_type=gpu_type, gpu_count=gpu_count, disk_gb=disk_gb)
    local_model_dir = Path(output_dir) / experiment_name
    instance_info = None
    client = None

    if not (local_model_dir / "model.pt").exists():
        raise FileNotFoundError(f"No model found at {local_model_dir / 'model.pt'}")

    print(f"Cloud eval: {experiment_name} on {provider.name}/{gpu_type}")

    try:
        # Provision + connect
        print(f"\n[1/5] Provisioning {gpu_type} on {provider.name}...")
        instance_info = provider.provision(spec)
        print(f"  Instance {instance_info.instance_id} ready")

        print("\n[2/5] Connecting via SSH...")
        client = ssh.connect(
            host=instance_info.host,
            port=instance_info.port,
            username=instance_info.username,
            ssh_key_path=instance_info.ssh_key_path,
        )

        # Upload project + model
        print("\n[3/5] Uploading project + model...")
        _upload_project(client, project_root)
        remote_model_dir = f"{REMOTE_PROJECT_DIR}/out/{shlex.quote(experiment_name)}"
        ssh.run_command(client, f"mkdir -p {remote_model_dir}", stream=False)
        ssh.upload_directory(client, local_model_dir, remote_model_dir)

        # Install + eval
        print("\n[4/5] Installing dependencies and running eval...")
        ssh.run_command(
            client,
            f"cd {REMOTE_PROJECT_DIR} && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "export PATH=$HOME/.local/bin:$PATH && "
            "uv sync --all-extras",
        )
        ssh.run_command(
            client,
            f"cd {REMOTE_PROJECT_DIR} && "
            f"export PATH=$HOME/.local/bin:$PATH && "
            f"uv run chessgpt-eval --model {remote_model_dir}/model.pt "
            f"--log-file {REMOTE_PROJECT_DIR}/experiments/log.jsonl",
        )

        # Download updated logs
        print("\n[5/5] Downloading logs...")
        _download_results(client, experiment_name, local_model_dir, project_root)
        print("Done!")

    finally:
        if client is not None:
            client.close()
        if instance_info is not None:
            print(f"\nDeprovisioning {instance_info.instance_id}...")
            provider.deprovision(instance_info.instance_id)


def _upload_project(
    client: paramiko.SSHClient,
    project_root: Path,
) -> int:
    """Upload source code, configs, and data to the remote instance.

    Args:
        client: Connected SSH/SFTP client.
        project_root: Local project root directory.

    Returns:
        Number of files uploaded.
    """
    total = 0

    # Upload main project directories and files
    for rel_path in UPLOAD_PATHS:
        local = project_root / rel_path
        remote = f"{REMOTE_PROJECT_DIR}/{rel_path}"
        if local.is_dir():
            total += ssh.upload_directory(client, local, remote)
        elif local.is_file():
            ssh.upload_file(client, local, remote)
            total += 1

    # Upload data files (CSVs, tokenizer JSON — not PGNs)
    data_files = _resolve_data_files(project_root)
    for data_file in data_files:
        rel = data_file.relative_to(project_root)
        remote_path = f"{REMOTE_PROJECT_DIR}/{rel.as_posix()}"
        ssh.upload_file(client, data_file, remote_path)
        total += 1

    return total


def _download_results(
    client: paramiko.SSHClient,
    experiment_name: str,
    local_output: Path,
    project_root: Path,
) -> None:
    """Download model output and merge experiment logs."""
    remote_output = f"{REMOTE_PROJECT_DIR}/out/{experiment_name}"

    # Download the full output directory
    local_output.mkdir(parents=True, exist_ok=True)
    count = ssh.download_directory(client, remote_output, local_output)
    print(f"  Downloaded {count} files to {local_output}")

    # Merge experiment logs
    remote_log = f"{REMOTE_PROJECT_DIR}/experiments/log.jsonl"
    try:
        _, stdout, _ = ssh.run_command(client, f"cat {remote_log}", stream=False)
        lines = stdout.strip().split("\n")
        local_log = project_root / "experiments" / "log.jsonl"
        _merge_jsonl(lines, local_log)
    except RuntimeError:
        # No remote log yet (eval might not have run)
        pass
