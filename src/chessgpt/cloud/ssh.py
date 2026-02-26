"""SSH/SFTP wrapper for remote command execution and file transfer.

All cloud providers ultimately expose an SSH-able machine. This module provides
the shared transport layer that the runner uses regardless of which provider
provisioned the instance. Uses paramiko rather than rsync because SFTP is
guaranteed available on every provider (rsync is not).
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path, PurePosixPath

import paramiko

# Patterns to skip when uploading the project tree.
# These are large or ephemeral and never needed on the remote side.
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".ruff_cache",
    "out",
    "*.pgn",
    "*.pgn.zst",
    ".venv",
    "node_modules",
)


def connect(
    host: str,
    port: int,
    username: str,
    ssh_key_path: str,
    *,
    retries: int = 12,
    initial_delay: float = 5.0,
    max_delay: float = 30.0,
) -> paramiko.SSHClient:
    """Connect to a remote host via SSH with exponential backoff.

    There's always a gap between "provider says ready" and "sshd is actually
    accepting connections". We retry with backoff to bridge that window.

    Args:
        host: Remote hostname or IP.
        port: SSH port (may be non-standard for providers like RunPod).
        username: SSH username (usually 'root').
        ssh_key_path: Path to the private key file.
        retries: Maximum number of connection attempts.
        initial_delay: Seconds to wait after the first failure.
        max_delay: Cap on the backoff delay.

    Returns:
        Connected paramiko.SSHClient ready for commands/SFTP.

    Raises:
        ConnectionError: If all retries are exhausted.
    """
    client = paramiko.SSHClient()
    # AutoAddPolicy is appropriate for ephemeral cloud instances where host keys
    # change on every provision. Not suitable for persistent/known hosts.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key_path = os.path.expanduser(ssh_key_path)
    delay = initial_delay

    for attempt in range(1, retries + 1):
        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                key_filename=key_path,
                timeout=10,
                allow_agent=False,
                look_for_keys=False,
            )
            return client
        except (
            paramiko.SSHException,
            OSError,
        ) as exc:
            if attempt == retries:
                raise ConnectionError(
                    f"Failed to connect to {host}:{port} after {retries} attempts"
                ) from exc
            print(f"  SSH attempt {attempt}/{retries} failed ({exc}), retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)

    # Unreachable, but satisfies the type checker
    raise ConnectionError(f"Failed to connect to {host}:{port}")


def _should_exclude(path: Path, exclude_patterns: tuple[str, ...]) -> bool:
    """Check whether a path matches any exclusion pattern."""
    name = path.name
    for pattern in exclude_patterns:
        if pattern.startswith("*"):
            # Suffix match (e.g. "*.pgn")
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True
    return False


def upload_directory(
    client: paramiko.SSHClient,
    local_dir: Path,
    remote_dir: str,
    *,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> int:
    """Upload a local directory tree to the remote host via SFTP.

    Creates the remote directory structure as needed. Skips files matching
    the exclude patterns (e.g. .git, __pycache__, large PGN files).

    Args:
        client: Connected SSH client.
        local_dir: Local directory to upload.
        remote_dir: Absolute path on the remote host.
        exclude: Tuple of patterns to skip.

    Returns:
        Number of files uploaded.
    """
    sftp = client.open_sftp()
    file_count = 0

    try:
        _sftp_mkdir_p(sftp, remote_dir)

        for local_path in sorted(local_dir.rglob("*")):
            # Check exclusions against every component of the relative path
            rel = local_path.relative_to(local_dir)
            if any(_should_exclude(part, exclude) for part in [Path(p) for p in rel.parts]):
                continue

            remote_path = str(PurePosixPath(remote_dir) / rel.as_posix())

            if local_path.is_dir():
                _sftp_mkdir_p(sftp, remote_path)
            elif local_path.is_file():
                sftp.put(str(local_path), remote_path)
                file_count += 1
    finally:
        sftp.close()

    return file_count


def download_directory(
    client: paramiko.SSHClient,
    remote_dir: str,
    local_dir: Path,
) -> int:
    """Download a remote directory tree to the local filesystem via SFTP.

    Args:
        client: Connected SSH client.
        remote_dir: Absolute path on the remote host.
        local_dir: Local directory to download into.

    Returns:
        Number of files downloaded.
    """
    sftp = client.open_sftp()
    file_count = 0

    try:
        file_count = _sftp_download_recursive(sftp, remote_dir, local_dir)
    finally:
        sftp.close()

    return file_count


def _sftp_download_recursive(
    sftp: paramiko.SFTPClient,
    remote_dir: str,
    local_dir: Path,
) -> int:
    """Recursively download all files from a remote directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for entry in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path = local_dir / entry.filename

        if stat.S_ISDIR(entry.st_mode or 0):
            count += _sftp_download_recursive(sftp, remote_path, local_path)
        else:
            sftp.get(remote_path, str(local_path))
            count += 1

    return count


def run_command(
    client: paramiko.SSHClient,
    command: str,
    *,
    stream: bool = True,
) -> tuple[int, str, str]:
    """Execute a command on the remote host.

    When stream=True (the default), stdout and stderr are printed in real-time
    as they arrive — critical for seeing tqdm progress bars and training output.
    The full output is also captured and returned.

    Args:
        client: Connected SSH client.
        command: Shell command to run.
        stream: If True, print output lines in real-time.

    Returns:
        Tuple of (exit_code, stdout_text, stderr_text).

    Raises:
        RuntimeError: If the command exits with a non-zero status.
    """
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport is not active")

    channel = transport.open_session()
    channel.exec_command(command)

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    # Read output as it arrives, interleaving stdout and stderr
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
        if channel.recv_ready():
            data = channel.recv(4096).decode("utf-8", errors="replace")
            stdout_chunks.append(data)
            if stream:
                print(data, end="", flush=True)

        if channel.recv_stderr_ready():
            data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
            stderr_chunks.append(data)
            if stream:
                print(data, end="", flush=True)

        # Avoid busy-spinning when no data is available yet
        if not channel.recv_ready() and not channel.recv_stderr_ready():
            time.sleep(0.1)

    # Drain any remaining data after exit
    while channel.recv_ready():
        data = channel.recv(4096).decode("utf-8", errors="replace")
        stdout_chunks.append(data)
        if stream:
            print(data, end="", flush=True)

    while channel.recv_stderr_ready():
        data = channel.recv_stderr(4096).decode("utf-8", errors="replace")
        stderr_chunks.append(data)
        if stream:
            print(data, end="", flush=True)

    exit_code = channel.recv_exit_status()
    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    if exit_code != 0:
        raise RuntimeError(
            f"Command failed (exit {exit_code}): {command}\nstderr: {stderr_text[-500:]}"
        )

    return exit_code, stdout_text, stderr_text


def upload_file(
    client: paramiko.SSHClient,
    local_path: Path,
    remote_path: str,
) -> None:
    """Upload a single file to the remote host, creating parent directories as needed.

    Args:
        client: Connected SSH client.
        local_path: Local file to upload.
        remote_path: Absolute destination path on the remote host.
    """
    sftp = client.open_sftp()
    try:
        parent = str(PurePosixPath(remote_path).parent)
        _sftp_mkdir_p(sftp, parent)
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def _sftp_mkdir_p(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    """Recursively create remote directories (like mkdir -p)."""
    path = PurePosixPath(remote_path)
    # Build each ancestor from root to leaf, e.g. /root, /root/chessgpt, ...
    dirs_to_create = []
    for parent in reversed(path.parents):
        if str(parent) != "/":
            dirs_to_create.append(str(parent))
    dirs_to_create.append(str(path))

    for d in dirs_to_create:
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)
