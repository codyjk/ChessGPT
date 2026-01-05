"""Data transfer utilities for cloud deployment."""

import subprocess
from pathlib import Path
from typing import Union, List

from .providers import Instance, SSHConfig


def build_rsync_command(
    ssh_config: SSHConfig,
    local_paths: List[Union[str, Path]],
    remote_path: str = "~/ChessGPT",
    exclude: List[str] = None
) -> List[str]:
    """Build rsync command for file transfer."""
    if exclude is None:
        exclude = [
            ".git",
            ".venv",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
            "*.log",
            "wandb",
            "models/*",  # Don't sync existing models
            "trained_models/*",
            ".cloudgpt",
        ]

    # Build SSH command for rsync
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    if ssh_config.key_file:
        ssh_cmd += f" -i {ssh_config.key_file}"
    ssh_cmd += f" -p {ssh_config.port}"

    rsync_cmd = [
        "rsync",
        "-avz",  # archive, verbose, compress
        "--progress",
        f"--rsh={ssh_cmd}",
    ]

    # Add exclusions
    for pattern in exclude:
        rsync_cmd.extend(["--exclude", pattern])

    # Add source paths
    for path in local_paths:
        rsync_cmd.append(str(path))

    # Add destination
    rsync_cmd.append(f"{ssh_config.user}@{ssh_config.host}:{remote_path}/")

    return rsync_cmd


def rsync_files(
    instance: Instance,
    local_paths: List[Union[str, Path]],
    remote_path: str = "~/ChessGPT",
    exclude: List[str] = None
) -> None:
    """Transfer files to remote instance using rsync."""
    rsync_cmd = build_rsync_command(instance.ssh_config, local_paths, remote_path, exclude)

    try:
        subprocess.run(rsync_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"rsync failed: {e}")


def rsync_from_remote(
    instance: Instance,
    remote_path: str,
    local_path: Union[str, Path],
    exclude: List[str] = None
) -> None:
    """Transfer files from remote instance to local machine."""
    if exclude is None:
        exclude = [".git", "__pycache__", "*.pyc"]

    ssh_config = instance.ssh_config

    # Build SSH command for rsync
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    if ssh_config.key_file:
        ssh_cmd += f" -i {ssh_config.key_file}"
    ssh_cmd += f" -p {ssh_config.port}"

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        f"--rsh={ssh_cmd}",
    ]

    # Add exclusions
    for pattern in exclude:
        rsync_cmd.extend(["--exclude", pattern])

    # Source and destination
    rsync_cmd.append(f"{ssh_config.user}@{ssh_config.host}:{remote_path}")
    rsync_cmd.append(str(local_path))

    try:
        subprocess.run(rsync_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"rsync from remote failed: {e}")


def sync_training_data(
    instance: Instance,
    config_name: str,
    local_project_root: Path = None
) -> None:
    """
    Sync training data based on config.
    Smart logic: small files via rsync, large files suggest S3.
    """
    if local_project_root is None:
        local_project_root = Path.cwd()

    # Load config to find data paths
    from omegaconf import OmegaConf
    import yaml

    pipeline_config_path = local_project_root / "configs" / "pipeline" / f"{config_name}.yaml"
    if not pipeline_config_path.exists():
        raise FileNotFoundError(f"Config not found: {pipeline_config_path}")

    with open(pipeline_config_path) as f:
        pipeline_config = yaml.safe_load(f)

    # Load data config
    data_config_name = pipeline_config.get("data")
    data_config_path = local_project_root / "configs" / "data" / f"{data_config_name}.yaml"

    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_config_path}")

    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    # Get data file paths
    tokenizer_path = local_project_root / data_config.get("tokenizer_path", "out/chess_tokenizer.json")
    training_data_path = local_project_root / data_config.get("training_data_path")
    validation_data_path = local_project_root / data_config.get("validation_data_path")

    # Check if files exist
    data_files = [tokenizer_path, training_data_path, validation_data_path]
    missing_files = [f for f in data_files if not f.exists()]

    if missing_files:
        print(f"\n⚠️  Missing data files:")
        for f in missing_files:
            print(f"   - {f}")
        raise FileNotFoundError("Please prepare training data first or use --prepare-data flag")

    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in data_files) / (1024 ** 2)
    print(f"\nData size: {total_size_mb:.1f}MB")

    if total_size_mb < 100:
        print("Using rsync for data transfer...")
        rsync_files(instance, [str(f) for f in data_files], remote_path="~/ChessGPT")
        print("✓ Data files synced")
    else:
        print("\n⚠️  Large dataset detected (>100MB)")
        print("Consider using S3 for faster transfer or --prepare-data to process on cloud")
        print("Proceeding with rsync (may be slow)...")
        rsync_files(instance, [str(f) for f in data_files], remote_path="~/ChessGPT")
        print("✓ Data files synced")


def retrieve_trained_model(
    instance: Instance,
    model_name: str,
    local_models_dir: Path = None
) -> None:
    """Download trained model from remote instance."""
    if local_models_dir is None:
        local_models_dir = Path("models")

    local_models_dir.mkdir(exist_ok=True)

    remote_model_path = f"~/ChessGPT/models/{model_name}"
    local_model_path = local_models_dir / model_name

    print(f"Downloading model: {model_name}")
    print(f"From: {remote_model_path}")
    print(f"To: {local_model_path}")

    rsync_from_remote(instance, remote_model_path, local_model_path)

    print(f"✓ Model downloaded to {local_model_path}")
