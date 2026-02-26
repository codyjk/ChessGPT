"""RunPod cloud provider implementation.

Uses the runpod Python SDK to manage GPU pod instances. RunPod exposes SSH
on non-standard ports via TCP proxy (e.g. {pod_id}-ssh.proxy.runpod.net:22226).
Requires the RUNPOD_API_KEY environment variable.
"""

from __future__ import annotations

import os
import time

from chessgpt.cloud.provider import (
    CloudProvider,
    GpuOffer,
    InstanceInfo,
    InstanceSpec,
    ProviderStatus,
)

# Map friendly GPU names to RunPod's internal GPU identifiers
_GPU_TYPE_MAP: dict[str, str] = {
    "A100": "NVIDIA A100 80GB PCIe",
    "A100_80GB": "NVIDIA A100 80GB PCIe",
    "A100_40GB": "NVIDIA A100-SXM4-40GB",
    "H100": "NVIDIA H100 80GB HBM3",
    "H100_SXM": "NVIDIA H100 80GB HBM3",
    "RTX_4090": "NVIDIA GeForce RTX 4090",
    "RTX_3090": "NVIDIA GeForce RTX 3090",
    "A6000": "NVIDIA RTX A6000",
    "A40": "NVIDIA A40",
    "L4": "NVIDIA L4",
    "L40": "NVIDIA L40",
    "L40S": "NVIDIA L40S",
}


def _get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise OSError(
            "RUNPOD_API_KEY environment variable is not set. "
            "Get your API key at https://www.runpod.io/console/user/settings"
        )
    return key


class RunPodProvider(CloudProvider):
    """RunPod cloud GPU provider.

    Provisions on-demand GPU pods via the RunPod API. Each pod is an ephemeral
    Docker container with SSH access on a non-standard port.
    """

    @property
    def name(self) -> str:
        return "runpod"

    def list_gpu_offers(self, gpu_type: str | None = None) -> list[GpuOffer]:
        """List available RunPod GPU types and their current pricing."""
        import runpod

        runpod.api_key = _get_api_key()
        gpus = runpod.get_gpus()

        offers = []
        for gpu in gpus:
            gpu_id = gpu.get("id", "")
            display_name = gpu.get("displayName", gpu_id)

            # Filter by type if requested
            if gpu_type:
                runpod_id = _GPU_TYPE_MAP.get(gpu_type, gpu_type)
                if gpu_id != runpod_id and display_name != runpod_id:
                    continue

            offers.append(
                GpuOffer(
                    gpu_type=display_name,
                    gpu_count=1,
                    cost_per_hour=gpu.get("communityPrice", 0.0),
                    vram_gb=gpu.get("memoryInGb", 0),
                    available=True,
                    extra={"runpod_id": gpu_id},
                )
            )

        return sorted(offers, key=lambda o: o.cost_per_hour)

    def provision(self, spec: InstanceSpec) -> InstanceInfo:
        """Provision a RunPod GPU pod and wait until SSH is reachable.

        Creates an on-demand pod with the requested GPU type and Docker image.
        Polls the RunPod API until the pod reaches RUNNING status, then extracts
        the SSH connection details from the pod's port mappings.
        """
        import runpod

        runpod.api_key = _get_api_key()
        runpod_gpu_id = _GPU_TYPE_MAP.get(spec.gpu_type, spec.gpu_type)

        pod = runpod.create_pod(
            name=f"chessgpt-{int(time.time())}",
            image_name=spec.image,
            gpu_type_id=runpod_gpu_id,
            gpu_count=spec.gpu_count,
            volume_in_gb=0,
            container_disk_in_gb=spec.disk_gb,
            ports="22/tcp",
            docker_args="",
        )

        pod_id = pod["id"]
        print(f"  Pod {pod_id} created, waiting for RUNNING state...")

        # Poll until running (timeout after 10 minutes)
        for _ in range(120):
            status = runpod.get_pod(pod_id)
            desired_status = status.get("desiredStatus", "")
            runtime = status.get("runtime")

            if desired_status == "RUNNING" and runtime:
                ports = runtime.get("ports", [])
                ssh_port_info = next(
                    (p for p in ports if p.get("privatePort") == 22),
                    None,
                )
                if ssh_port_info:
                    host = ssh_port_info.get("ip", f"{pod_id}-ssh.proxy.runpod.net")
                    port = ssh_port_info.get("publicPort", 22)

                    # RunPod uses the user's SSH key from their account settings
                    ssh_key_path = os.environ.get(
                        "RUNPOD_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519")
                    )

                    cost = status.get("costPerHr", 0.0)

                    return InstanceInfo(
                        instance_id=pod_id,
                        host=host,
                        port=port,
                        username="root",
                        ssh_key_path=ssh_key_path,
                        gpu_type=spec.gpu_type,
                        cost_per_hour=cost,
                    )

            time.sleep(5)

        # Timed out — clean up and raise
        self.deprovision(pod_id)
        raise TimeoutError(f"Pod {pod_id} did not reach RUNNING state within 10 minutes")

    def status(self, instance_id: str) -> ProviderStatus:
        """Get the current status of a RunPod pod."""
        import runpod

        runpod.api_key = _get_api_key()
        pod = runpod.get_pod(instance_id)

        desired = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime", {}) or {}
        uptime = runtime.get("uptimeInSeconds", 0)
        cost_hr = pod.get("costPerHr", 0.0)

        state_map = {
            "RUNNING": "running",
            "EXITED": "stopped",
            "TERMINATED": "terminated",
            "CREATED": "provisioning",
        }

        return ProviderStatus(
            instance_id=instance_id,
            state=state_map.get(desired, "unknown"),
            uptime_seconds=uptime,
            estimated_cost=(uptime / 3600) * cost_hr,
        )

    def deprovision(self, instance_id: str) -> None:
        """Terminate a RunPod pod. Idempotent — safe in finally blocks."""
        import runpod

        runpod.api_key = _get_api_key()
        try:
            runpod.terminate_pod(instance_id)
        except Exception:
            # Idempotent: if the pod is already gone, that's fine
            pass


def create_provider() -> RunPodProvider:
    """Factory function used by the provider registry."""
    return RunPodProvider()
