"""Vast.ai cloud provider implementation.

Vast.ai is a GPU marketplace — instances come from individual hosts rather than
a single datacenter. We search for offers matching our requirements, pick the
cheapest reliable one, and create an instance. Uses the vastai CLI tool via
subprocess (it's a standalone binary, not a Python import).

Requires the VASTAI_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from chessgpt.cloud.provider import (
    CloudProvider,
    GpuOffer,
    InstanceInfo,
    InstanceSpec,
    ProviderStatus,
)

# Minimum reliability score (0-1) for Vast.ai hosts. Below this threshold,
# machines are too flaky for multi-hour training runs.
MIN_RELIABILITY = 0.95

# Map friendly names to Vast.ai GPU model strings
_GPU_TYPE_MAP: dict[str, str] = {
    "A100": "A100",
    "A100_80GB": "A100_80GB",
    "A100_40GB": "A100",
    "H100": "H100",
    "RTX_4090": "RTX 4090",
    "RTX_3090": "RTX 3090",
    "A6000": "RTX A6000",
    "L40S": "L40S",
}


def _get_api_key() -> str:
    key = os.environ.get("VASTAI_API_KEY", "")
    if not key:
        raise OSError(
            "VASTAI_API_KEY environment variable is not set. "
            "Get your API key at https://cloud.vast.ai/cli/"
        )
    return key


def _run_vastai(*args: str) -> str:
    """Run a vastai CLI command and return stdout.

    Raises subprocess.CalledProcessError on non-zero exit.
    """
    env = {**os.environ, "VAST_API_KEY": _get_api_key()}
    result = subprocess.run(
        ["vastai", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return result.stdout


class VastAiProvider(CloudProvider):
    """Vast.ai GPU marketplace provider.

    Searches the marketplace for available GPU offers, filters by reliability,
    and provisions the cheapest matching instance.
    """

    @property
    def name(self) -> str:
        return "vastai"

    def list_gpu_offers(self, gpu_type: str | None = None) -> list[GpuOffer]:
        """Search Vast.ai marketplace for GPU offers."""
        query = "reliability>0.95 num_gpus=1 rentable=true"
        if gpu_type:
            vast_gpu = _GPU_TYPE_MAP.get(gpu_type, gpu_type)
            query += f" gpu_name={vast_gpu}"

        raw = _run_vastai("search", "offers", query, "--raw")
        try:
            offers_data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        offers = []
        for item in offers_data:
            offers.append(
                GpuOffer(
                    gpu_type=item.get("gpu_name", "unknown"),
                    gpu_count=item.get("num_gpus", 1),
                    cost_per_hour=item.get("dph_total", 0.0),
                    vram_gb=int(item.get("gpu_ram", 0) / 1024) if item.get("gpu_ram") else 0,
                    available=True,
                    extra={
                        "offer_id": item.get("id"),
                        "reliability": item.get("reliability", 0.0),
                        "dlperf": item.get("dlperf", 0.0),
                    },
                )
            )

        return sorted(offers, key=lambda o: o.cost_per_hour)

    def provision(self, spec: InstanceSpec) -> InstanceInfo:
        """Find the cheapest reliable offer and create an instance.

        Searches the Vast.ai marketplace, picks the best offer, creates an
        instance, and waits for SSH to become reachable.
        """
        vast_gpu = _GPU_TYPE_MAP.get(spec.gpu_type, spec.gpu_type)
        query = (
            f"reliability>{MIN_RELIABILITY} "
            f"num_gpus>={spec.gpu_count} "
            f"gpu_name={vast_gpu} "
            f"disk_space>={spec.disk_gb} "
            "rentable=true"
        )

        raw = _run_vastai("search", "offers", query, "--raw")
        offers = json.loads(raw)

        if not offers:
            raise RuntimeError(f"No Vast.ai offers found for {spec.gpu_type} (query: {query})")

        # Pick cheapest
        best = min(offers, key=lambda o: o.get("dph_total", float("inf")))
        offer_id = best["id"]
        cost = best.get("dph_total", 0.0)

        print(f"  Selected offer {offer_id} at ${cost:.2f}/hr")

        # Create instance
        create_output = _run_vastai(
            "create",
            "instance",
            str(offer_id),
            "--image",
            spec.image,
            "--disk",
            str(spec.disk_gb),
            "--ssh",
            "--raw",
        )
        create_data = json.loads(create_output)
        instance_id = str(create_data.get("new_contract", create_data.get("id", "")))

        if not instance_id:
            raise RuntimeError(f"Failed to create Vast.ai instance: {create_output}")

        print(f"  Instance {instance_id} created, waiting for SSH...")

        # Poll until running with SSH info (timeout 10 minutes)
        ssh_key_path = os.environ.get("VASTAI_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

        for _ in range(120):
            try:
                show_raw = _run_vastai("show", "instance", instance_id, "--raw")
                instance_data = json.loads(show_raw)

                status = instance_data.get("actual_status", "")
                ssh_host = instance_data.get("ssh_host", "")
                ssh_port = instance_data.get("ssh_port", 0)

                if status == "running" and ssh_host and ssh_port:
                    return InstanceInfo(
                        instance_id=instance_id,
                        host=ssh_host,
                        port=int(ssh_port),
                        username="root",
                        ssh_key_path=ssh_key_path,
                        gpu_type=spec.gpu_type,
                        cost_per_hour=cost,
                    )
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass

            time.sleep(5)

        self.deprovision(instance_id)
        raise TimeoutError(f"Instance {instance_id} did not become ready within 10 minutes")

    def status(self, instance_id: str) -> ProviderStatus:
        """Get current status of a Vast.ai instance."""
        raw = _run_vastai("show", "instance", instance_id, "--raw")
        data = json.loads(raw)

        status = data.get("actual_status", "unknown")
        state_map = {
            "running": "running",
            "loading": "provisioning",
            "exited": "stopped",
        }

        start_date = data.get("start_date", 0)
        uptime = time.time() - start_date if start_date else 0
        cost_hr = data.get("dph_total", 0.0)

        return ProviderStatus(
            instance_id=instance_id,
            state=state_map.get(status, "unknown"),
            uptime_seconds=max(uptime, 0),
            estimated_cost=(uptime / 3600) * cost_hr if uptime > 0 else 0,
        )

    def deprovision(self, instance_id: str) -> None:
        """Destroy a Vast.ai instance. Idempotent."""
        try:
            _run_vastai("destroy", "instance", instance_id)
        except Exception:
            # Idempotent: if the instance is already gone, that's fine
            pass


def create_provider() -> VastAiProvider:
    """Factory function used by the provider registry."""
    return VastAiProvider()
