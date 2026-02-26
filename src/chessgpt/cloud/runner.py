"""Orchestrator for cloud training runs.

Manages the full lifecycle: provision → upload → install → launch in tmux →
disconnect. Separate commands (status, attach, download, deprovision) reconnect
to the running pod via saved state in .cloud/pod.json.

The training code itself is untouched -- we just run the existing
chessgpt-train and chessgpt-eval CLI commands on the remote machine over SSH.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from chessgpt.cloud import ssh
from chessgpt.cloud.pricing import format_cost_summary
from chessgpt.cloud.provider import CloudProvider, InstanceSpec
from chessgpt.cloud.state import PodState, clear_state, load_state, save_state

if TYPE_CHECKING:
    import paramiko

# Remote working directory on the GPU instance
REMOTE_PROJECT_DIR = "/root/chessgpt"

# Remote log and sentinel files
REMOTE_LOG_FILE = f"{REMOTE_PROJECT_DIR}/train.log"
REMOTE_EXIT_CODE_FILE = f"{REMOTE_PROJECT_DIR}/.train_exit_code"

# tmux session name
TMUX_SESSION = "chessgpt-train"

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


def _check_training_done(
    client: paramiko.SSHClient,
) -> tuple[bool, int | None]:
    """Check if training has finished by looking for the exit code sentinel.

    Returns:
        (is_done, exit_code) -- exit_code is None if still running.
    """
    try:
        _, stdout, _ = ssh.run_command(client, f"cat {REMOTE_EXIT_CODE_FILE}", stream=False)
        code = int(stdout.strip())
        return True, code
    except (RuntimeError, ValueError):
        return False, None


def run_cloud_train(
    config_path: str,
    experiment_name: str,
    provider: CloudProvider,
    gpu_type: str,
    *,
    gpu_count: int = 1,
    disk_gb: int = 50,
    project_root: Path | None = None,
) -> None:
    """Provision a cloud GPU, upload code, and launch training in a detached tmux session.

    Unlike the old blocking approach, this returns as soon as training is running.
    Use cloud_status/cloud_attach/cloud_download/cloud_deprovision to manage.

    On setup failure (provision, connect, upload, install), the pod IS deprovisioned
    so no orphaned pods are left. On success, the pod stays running.

    Args:
        config_path: Path to the TOML config file.
        experiment_name: Name for this experiment (used for output directory).
        provider: CloudProvider instance to use.
        gpu_type: GPU type to request (e.g. "A100", "RTX_4090").
        gpu_count: Number of GPUs to request.
        disk_gb: Disk space in GB.
        project_root: Local project root (auto-detected if None).
    """
    if project_root is None:
        project_root = Path.cwd()

    # Block if a pod is already active
    existing = load_state(project_root)
    if existing is not None:
        raise RuntimeError(
            f"A pod is already active: {existing.instance_id} "
            f"(experiment: {existing.experiment_name}). "
            f"Use 'chessgpt-cloud deprovision' first."
        )

    spec = InstanceSpec(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        disk_gb=disk_gb,
    )

    config_size = _detect_config_size(config_path)
    instance_info = None
    client = None

    print(f"Cloud training: {experiment_name} ({config_size}) on {provider.name}/{gpu_type}")

    try:
        # Step 1: Provision
        print(f"\n[1/5] Provisioning {gpu_type} on {provider.name}...")
        instance_info = provider.provision(spec)
        print(
            f"  Instance {instance_info.instance_id} ready "
            f"({instance_info.host}:{instance_info.port})"
        )
        print(f"  Cost: ${instance_info.cost_per_hour:.2f}/hr")

        # Step 2: Connect
        print("\n[2/5] Connecting via SSH...")
        client = ssh.connect(
            host=instance_info.host,
            port=instance_info.port,
            username=instance_info.username,
            ssh_key_path=instance_info.ssh_key_path,
        )
        print("  Connected.")

        # Step 3: Upload
        print("\n[3/5] Uploading project files...")
        file_count = _upload_project(client, project_root)
        print(f"  Uploaded {file_count} files.")

        # Step 4: Install dependencies + tmux
        print("\n[4/5] Installing dependencies...")
        ssh.run_command(
            client,
            f"cd {REMOTE_PROJECT_DIR} && pip install -e . 2>&1",
        )
        ssh.run_command(
            client,
            "which tmux || (apt-get update -qq && apt-get install -y -qq tmux)",
            stream=False,
        )

        # Step 5: Launch training in detached tmux
        print(f"\n[5/5] Launching training in tmux session '{TMUX_SESSION}'...")
        remote_output = f"{REMOTE_PROJECT_DIR}/out"
        train_cmd = (
            f"cd {REMOTE_PROJECT_DIR} && "
            f"chessgpt-train --config {shlex.quote(config_path)} "
            f"--name {shlex.quote(experiment_name)} "
            f"--output-dir {shlex.quote(remote_output)} "
            f"2>&1 | tee {REMOTE_LOG_FILE}; "
            f"echo ${{PIPESTATUS[0]}} > {REMOTE_EXIT_CODE_FILE}"
        )
        tmux_cmd = (
            f"tmux new-session -d -s {TMUX_SESSION} "
            f"{shlex.quote(f'bash -c {shlex.quote(train_cmd)}')}"
        )
        ssh.run_command(client, tmux_cmd, stream=False)

        # Save state so other commands can reconnect
        pod_state = PodState.from_instance_info(
            instance_info,
            experiment_name=experiment_name,
            config_path=config_path,
            provider_name=provider.name,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        state_path = save_state(pod_state, project_root)
        print(f"  Pod state saved to {state_path}")
        print(f"\n  Training is running on {instance_info.instance_id}.")
        print("  Use 'chessgpt-cloud status' to check progress.")
        print("  Use 'chessgpt-cloud attach' to watch live output.")
        print("  Use 'chessgpt-cloud download' to fetch results.")
        print("  Use 'chessgpt-cloud deprovision' to tear down the pod.")

    except BaseException:
        # Setup failed -- deprovision to avoid orphaned pods
        if instance_info is not None:
            print(f"\nSetup failed. Deprovisioning {instance_info.instance_id}...")
            provider.deprovision(instance_info.instance_id)
            print("  Instance terminated.")
        raise

    finally:
        if client is not None:
            client.close()


def cloud_status(
    *,
    project_root: Path | None = None,
) -> None:
    """Check training progress on the active pod.

    SSHes in, checks the sentinel file, and tails the log.
    """
    if project_root is None:
        project_root = Path.cwd()

    pod = load_state(project_root)
    if pod is None:
        print("No active pod. Use 'chessgpt-cloud train' to start one.")
        return

    print(f"Pod: {pod.instance_id} ({pod.gpu_type})")
    print(f"Experiment: {pod.experiment_name}")
    print(f"Started: {pod.started_at}")
    print(f"Cost: ${pod.cost_per_hour:.2f}/hr")

    client = None
    try:
        client = ssh.connect(
            host=pod.host,
            port=pod.port,
            username=pod.username,
            ssh_key_path=pod.ssh_key_path,
        )

        is_done, exit_code = _check_training_done(client)

        if is_done:
            if exit_code == 0:
                print("\nTraining COMPLETED successfully.")
            else:
                print(f"\nTraining FAILED (exit code {exit_code}).")
            print("Use 'chessgpt-cloud download' to fetch results.")
        else:
            print("\nTraining is still RUNNING.")

        # Tail the log
        print("\n--- Last 15 lines of training log ---")
        try:
            _, stdout, _ = ssh.run_command(client, f"tail -n 15 {pod.log_file}", stream=False)
            print(stdout.rstrip())
        except RuntimeError:
            print("  (no log output yet)")
        print("---")

        # Elapsed time
        start = datetime.fromisoformat(pod.started_at)
        elapsed_hours = (datetime.now(timezone.utc) - start).total_seconds() / 3600
        estimated_cost = elapsed_hours * pod.cost_per_hour
        print(f"\nElapsed: {elapsed_hours:.1f}h, estimated cost: ${estimated_cost:.2f}")

    finally:
        if client is not None:
            client.close()


def cloud_attach(
    *,
    project_root: Path | None = None,
) -> None:
    """Attach to the tmux training session for live output.

    Uses native ssh (not paramiko) because paramiko can't do interactive tmux.
    This replaces the current process via os.execvp.
    """
    if project_root is None:
        project_root = Path.cwd()

    pod = load_state(project_root)
    if pod is None:
        print("No active pod. Use 'chessgpt-cloud train' to start one.")
        sys.exit(1)

    ssh_key_path = os.path.expanduser(pod.ssh_key_path)
    ssh_args = [
        "ssh",
        "-t",
        "-i",
        ssh_key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-p",
        str(pod.port),
        f"{pod.username}@{pod.host}",
        f"tmux attach-session -t {pod.tmux_session}",
    ]

    print(f"Attaching to tmux session '{pod.tmux_session}' on {pod.instance_id}...")
    print("  (Ctrl+B, D to detach)\n")

    # Replace this process with ssh
    os.execvp("ssh", ssh_args)


def cloud_download(
    *,
    output_dir: str = "out",
    project_root: Path | None = None,
) -> Path:
    """Download results from the active pod.

    Returns:
        Path to the local output directory.
    """
    if project_root is None:
        project_root = Path.cwd()

    pod = load_state(project_root)
    if pod is None:
        raise RuntimeError("No active pod. Use 'chessgpt-cloud train' to start one.")

    local_output = Path(output_dir) / pod.experiment_name

    client = None
    try:
        print(f"Connecting to {pod.instance_id}...")
        client = ssh.connect(
            host=pod.host,
            port=pod.port,
            username=pod.username,
            ssh_key_path=pod.ssh_key_path,
        )

        print(f"Downloading results to {local_output}...")
        _download_results(client, pod.experiment_name, local_output, project_root)
        print("Download complete.")

    finally:
        if client is not None:
            client.close()

    return local_output


def cloud_deprovision(
    *,
    download: bool = True,
    output_dir: str = "out",
    project_root: Path | None = None,
    provider: CloudProvider | None = None,
) -> None:
    """Tear down the active pod, optionally downloading results first.

    Args:
        download: If True (default), download results before terminating.
        output_dir: Local base output directory.
        project_root: Local project root (auto-detected if None).
        provider: CloudProvider instance. If None, auto-resolved from pod state.
    """
    if project_root is None:
        project_root = Path.cwd()

    pod = load_state(project_root)
    if pod is None:
        print("No active pod to deprovision.")
        return

    if provider is None:
        from chessgpt.cloud.providers import get_provider

        provider = get_provider(pod.provider_name)

    # Download results if requested
    if download:
        client = None
        try:
            print("Downloading results before deprovisioning...")
            client = ssh.connect(
                host=pod.host,
                port=pod.port,
                username=pod.username,
                ssh_key_path=pod.ssh_key_path,
            )
            local_output = Path(output_dir) / pod.experiment_name
            _download_results(client, pod.experiment_name, local_output, project_root)
        except Exception as exc:
            print(f"  Warning: download failed ({exc}), proceeding with deprovision.")
        finally:
            if client is not None:
                client.close()

    # Terminate the pod
    print(f"Deprovisioning {pod.instance_id}...")
    provider.deprovision(pod.instance_id)
    print("  Instance terminated.")

    # Clear saved state
    clear_state(project_root)
    print("  Pod state cleared.")

    # Cost summary
    start = datetime.fromisoformat(pod.started_at)
    elapsed_hours = (datetime.now(timezone.utc) - start).total_seconds() / 3600
    print(format_cost_summary(pod.gpu_type, elapsed_hours, pod.cost_per_hour))


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
    and runs chessgpt-eval instead. Still blocking -- eval is fast.

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
            f"cd {REMOTE_PROJECT_DIR} && pip install -e . 2>&1",
        )
        ssh.run_command(
            client,
            f"cd {REMOTE_PROJECT_DIR} && "
            f"chessgpt-eval --model {remote_model_dir}/model.pt "
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

    # Upload data files (CSVs, tokenizer JSON -- not PGNs)
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
