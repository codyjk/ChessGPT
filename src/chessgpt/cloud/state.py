"""Pod state persistence for reconnecting to running cloud instances.

Saves connection info to .cloud/pod.json so separate CLI invocations
(status, attach, download, deprovision) can find the active pod without
keeping an SSH connection open the entire time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from chessgpt.cloud.provider import InstanceInfo

STATE_DIR = ".cloud"
STATE_FILE = "pod.json"


@dataclass
class PodState:
    """Everything needed to reconnect to a running training pod."""

    instance_id: str
    host: str
    port: int
    username: str
    ssh_key_path: str
    gpu_type: str
    cost_per_hour: float
    experiment_name: str
    config_path: str
    provider_name: str
    started_at: str  # ISO 8601
    tmux_session: str = "chessgpt-train"
    log_file: str = "/root/chessgpt/train.log"
    exit_code_file: str = "/root/chessgpt/.train_exit_code"

    @classmethod
    def from_instance_info(
        cls,
        info: InstanceInfo,
        *,
        experiment_name: str,
        config_path: str,
        provider_name: str,
        started_at: str,
    ) -> PodState:
        """Create a PodState from an InstanceInfo plus training metadata."""
        return cls(
            instance_id=info.instance_id,
            host=info.host,
            port=info.port,
            username=info.username,
            ssh_key_path=info.ssh_key_path,
            gpu_type=info.gpu_type,
            cost_per_hour=info.cost_per_hour,
            experiment_name=experiment_name,
            config_path=config_path,
            provider_name=provider_name,
            started_at=started_at,
        )

    def to_instance_info(self) -> InstanceInfo:
        """Convert back to an InstanceInfo for provider API calls."""
        return InstanceInfo(
            instance_id=self.instance_id,
            host=self.host,
            port=self.port,
            username=self.username,
            ssh_key_path=self.ssh_key_path,
            gpu_type=self.gpu_type,
            cost_per_hour=self.cost_per_hour,
        )


def save_state(pod: PodState, project_root: Path) -> Path:
    """Write pod state to .cloud/pod.json. Returns the path written."""
    state_dir = project_root / STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / STATE_FILE
    state_path.write_text(json.dumps(asdict(pod), indent=2) + "\n")
    return state_path


def load_state(project_root: Path) -> PodState | None:
    """Load pod state from .cloud/pod.json, or None if no active pod."""
    state_path = project_root / STATE_DIR / STATE_FILE
    if not state_path.exists():
        return None
    data = json.loads(state_path.read_text())
    return PodState(**data)


def clear_state(project_root: Path) -> None:
    """Remove .cloud/pod.json. Safe to call if file doesn't exist."""
    state_path = project_root / STATE_DIR / STATE_FILE
    state_path.unlink(missing_ok=True)
