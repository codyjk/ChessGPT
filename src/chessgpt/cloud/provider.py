"""ABC and dataclasses for cloud GPU providers.

Separating the interface from concrete implementations allows adding new
providers (Lambda, CoreWeave, etc.) without changing the runner or SSH layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class InstanceSpec:
    """What we want from a cloud GPU instance."""

    gpu_type: str  # e.g. "A100", "RTX_4090"
    gpu_count: int = 1
    disk_gb: int = 50
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


@dataclass
class InstanceInfo:
    """What the provider gives us back after provisioning."""

    instance_id: str
    host: str
    port: int
    username: str
    ssh_key_path: str
    gpu_type: str
    cost_per_hour: float


@dataclass
class ProviderStatus:
    """Current state of a provisioned instance."""

    instance_id: str
    state: str  # "provisioning", "running", "stopped", "terminated"
    uptime_seconds: float = 0.0
    estimated_cost: float = 0.0


@dataclass
class GpuOffer:
    """A GPU offer from a provider."""

    gpu_type: str
    gpu_count: int
    cost_per_hour: float
    vram_gb: int = 0
    available: bool = True
    extra: dict = field(default_factory=dict)


class CloudProvider(ABC):
    """Abstract base class for cloud GPU providers.

    Each provider implements four methods: list_gpu_offers, provision,
    status, and deprovision. Everything else (file sync, remote execution)
    goes through the shared SSH layer.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'runpod', 'vastai')."""

    @abstractmethod
    def list_gpu_offers(self, gpu_type: str | None = None) -> list[GpuOffer]:
        """List available GPU offers, optionally filtered by type."""

    @abstractmethod
    def provision(self, spec: InstanceSpec) -> InstanceInfo:
        """Provision an instance. Blocks until SSH is reachable."""

    @abstractmethod
    def status(self, instance_id: str) -> ProviderStatus:
        """Get the current status of an instance."""

    @abstractmethod
    def deprovision(self, instance_id: str) -> None:
        """Tear down an instance. Idempotent (safe in finally blocks)."""
