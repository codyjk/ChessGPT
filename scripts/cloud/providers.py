"""Cloud GPU provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import json


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    user: str
    port: int = 22
    key_file: Optional[str] = None


@dataclass
class Instance:
    """Cloud GPU instance."""
    id: str
    provider: str
    gpu_type: str
    ip: str
    ssh_config: SSHConfig
    price_per_hour: float
    region: Optional[str] = None
    status: str = "running"


class CloudProvider(ABC):
    """Base class for cloud GPU providers."""

    @abstractmethod
    def provision(self, gpu_type: str, region: Optional[str] = None) -> Instance:
        """Provision a new GPU instance."""
        pass

    @abstractmethod
    def get_ssh_config(self, instance: Instance) -> SSHConfig:
        """Get SSH connection details."""
        pass

    @abstractmethod
    def terminate(self, instance: Instance) -> None:
        """Terminate instance."""
        pass

    @abstractmethod
    def get_pricing(self, gpu_type: str) -> float:
        """Get hourly rate for GPU type."""
        pass


class SSHProvider(CloudProvider):
    """Generic SSH-based provider for manually provisioned instances."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("configs/cloud/custom.yaml")
        self.instances: dict[str, Instance] = {}

    def provision(self, gpu_type: str, region: Optional[str] = None) -> Instance:
        """
        Interactive provisioning - prompts user for SSH details.
        Assumes instance is already running and accessible.
        """
        print("\n=== SSH Provider: Manual Instance Setup ===")
        print("Please provide details for your GPU instance:\n")

        host = input("Instance IP or hostname: ").strip()
        user = input(f"SSH user [{os.getenv('USER', 'ubuntu')}]: ").strip() or os.getenv('USER', 'ubuntu')
        port = input("SSH port [22]: ").strip() or "22"
        key_file = input(f"SSH key path [~/.ssh/id_rsa]: ").strip() or "~/.ssh/id_rsa"

        # Expand tilde in key_file path
        key_file = os.path.expanduser(key_file)

        # Verify key file exists
        if not os.path.exists(key_file):
            raise FileNotFoundError(f"SSH key not found: {key_file}")

        hourly_rate_input = input("Hourly rate (for cost tracking) [$0.00]: ").strip() or "0.00"
        hourly_rate = float(hourly_rate_input)

        # Generate instance ID
        instance_id = f"ssh-{host.replace('.', '-')}"

        ssh_config = SSHConfig(
            host=host,
            user=user,
            port=int(port),
            key_file=key_file
        )

        instance = Instance(
            id=instance_id,
            provider="ssh",
            gpu_type=gpu_type,
            ip=host,
            ssh_config=ssh_config,
            price_per_hour=hourly_rate,
            region=region
        )

        self.instances[instance_id] = instance
        return instance

    def get_ssh_config(self, instance: Instance) -> SSHConfig:
        """Get SSH configuration."""
        return instance.ssh_config

    def terminate(self, instance: Instance) -> None:
        """SSH provider doesn't terminate - user manages manually."""
        print(f"\n⚠️  SSH Provider: Instance {instance.id} not terminated")
        print("Please terminate the instance manually if needed.")
        if instance.id in self.instances:
            del self.instances[instance.id]

    def get_pricing(self, gpu_type: str) -> float:
        """Return user-specified pricing."""
        return 0.0  # User specifies during provision


class LambdaProvider(CloudProvider):
    """Lambda Labs cloud provider."""

    API_BASE = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        import requests
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Lambda API key required. Set LAMBDA_API_KEY environment variable or pass api_key parameter."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        # GPU pricing (as of 2024)
        self.pricing = {
            "A100": 1.10,
            "A100-80GB": 1.50,
            "A10": 0.60,
            "RTX6000": 0.50,
        }

    def provision(self, gpu_type: str, region: Optional[str] = None) -> Instance:
        """Provision Lambda Labs instance via API."""
        import requests

        # Map our GPU names to Lambda instance types
        instance_type_map = {
            "A100": "gpu_1x_a100",
            "A100-80GB": "gpu_1x_a100_sxm4",
            "A10": "gpu_1x_a10",
            "RTX6000": "gpu_1x_rtx6000",
        }

        instance_type = instance_type_map.get(gpu_type)
        if not instance_type:
            raise ValueError(f"Unsupported GPU type: {gpu_type}. Choose from: {list(self.pricing.keys())}")

        # Check available regions
        regions_response = self.session.get(f"{self.API_BASE}/instance-types")
        regions_response.raise_for_status()
        available_regions = regions_response.json().get("data", {}).get(instance_type, {}).get("regions_with_capacity_available", [])

        if not available_regions:
            raise RuntimeError(f"No capacity available for {gpu_type}")

        target_region = region if region in available_regions else available_regions[0]

        # Launch instance
        launch_data = {
            "region_name": target_region,
            "instance_type_name": instance_type,
            "ssh_key_names": ["chessgpt-deploy"],  # Assumes key uploaded to Lambda
        }

        print(f"Launching {gpu_type} in {target_region}...")
        launch_response = self.session.post(f"{self.API_BASE}/instance-operations/launch", json=launch_data)
        launch_response.raise_for_status()

        instance_data = launch_response.json()["data"]["instance_ids"][0]
        instance_id = instance_data

        # Wait for instance to be running
        print("Waiting for instance to be ready...")
        import time
        for _ in range(60):  # 5 minute timeout
            details_response = self.session.get(f"{self.API_BASE}/instances/{instance_id}")
            details_response.raise_for_status()
            instance_info = details_response.json()["data"]

            if instance_info["status"] == "active":
                ssh_config = SSHConfig(
                    host=instance_info["ip"],
                    user="ubuntu",
                    port=22,
                    key_file=os.path.expanduser("~/.ssh/chessgpt_deploy")
                )

                instance = Instance(
                    id=instance_id,
                    provider="lambda",
                    gpu_type=gpu_type,
                    ip=instance_info["ip"],
                    ssh_config=ssh_config,
                    price_per_hour=self.pricing[gpu_type],
                    region=target_region,
                    status="running"
                )

                print(f"✓ Instance ready: {instance_id} ({instance.ip})")
                return instance

            time.sleep(5)

        raise RuntimeError("Instance failed to start within timeout")

    def get_ssh_config(self, instance: Instance) -> SSHConfig:
        """Get SSH configuration."""
        return instance.ssh_config

    def terminate(self, instance: Instance) -> None:
        """Terminate Lambda Labs instance."""
        print(f"Terminating instance {instance.id}...")
        response = self.session.post(f"{self.API_BASE}/instance-operations/terminate", json={"instance_ids": [instance.id]})
        response.raise_for_status()
        print(f"✓ Instance {instance.id} terminated")

    def get_pricing(self, gpu_type: str) -> float:
        """Get hourly rate."""
        return self.pricing.get(gpu_type, 0.0)


def get_provider(provider_name: str, **kwargs) -> CloudProvider:
    """Factory function to get provider instance."""
    # Import here to avoid circular dependency
    if provider_name.lower() == "local-test":
        from .test_local import LocalTestProvider
        return LocalTestProvider(**kwargs)

    providers = {
        "lambda": LambdaProvider,
        "ssh": SSHProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}. Choose from: {list(providers.keys()) + ['local-test']}")

    return provider_class(**kwargs)
