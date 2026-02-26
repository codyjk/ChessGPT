"""Cloud provider registry.

Lazy-imports providers to avoid requiring all provider SDKs at import time.
A user who only has the runpod SDK installed won't hit ImportErrors from vastai.
"""

from __future__ import annotations

from chessgpt.cloud.provider import CloudProvider

_PROVIDERS: dict[str, str] = {
    "runpod": "chessgpt.cloud.providers.runpod",
    "vastai": "chessgpt.cloud.providers.vastai",
}


def get_provider(name: str) -> CloudProvider:
    """Look up a cloud provider by name and return an instance.

    Args:
        name: Provider name (e.g. "runpod", "vastai").

    Returns:
        An initialized CloudProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
        ImportError: If the provider's SDK is not installed.
    """
    if name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    import importlib

    module = importlib.import_module(_PROVIDERS[name])
    # Each provider module exports a create_provider() factory function
    return module.create_provider()


def list_providers() -> list[str]:
    """Return the names of all registered providers."""
    return sorted(_PROVIDERS.keys())
