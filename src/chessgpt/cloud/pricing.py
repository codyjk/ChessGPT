"""Cost estimation and display for cloud GPU training runs.

Reference prices are approximate and may drift from actual provider pricing.
They exist to give ballpark estimates before provisioning — the real cost
comes from the provider's API after the run completes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reference GPU prices ($/hr). These are rough medians across providers
# as of early 2026. Actual prices vary by provider, availability, and region.
REFERENCE_PRICES: dict[str, float] = {
    "RTX_3090": 0.25,
    "RTX_4090": 0.45,
    "A100_40GB": 1.20,
    "A100_80GB": 1.60,
    "A100": 1.60,
    "H100": 2.50,
    "H100_SXM": 3.20,
    "L40S": 0.80,
    "A6000": 0.50,
}

# Rough training time estimates (hours) per config size on a single GPU.
# Based on 270K games. Actual time depends on batch size and GPU speed.
TRAINING_TIME_ESTIMATES: dict[str, dict[str, float]] = {
    "RTX_4090": {"tiny": 0.05, "small": 0.4, "medium": 2.0, "large": 12.0},
    "A100": {"tiny": 0.03, "small": 0.25, "medium": 1.2, "large": 7.0},
    "H100": {"tiny": 0.02, "small": 0.15, "medium": 0.8, "large": 4.5},
}


@dataclass
class CostEstimate:
    """Estimated cost for a training run."""

    gpu_type: str
    config_size: str
    estimated_hours: float
    cost_per_hour: float

    @property
    def estimated_cost(self) -> float:
        return self.estimated_hours * self.cost_per_hour


def estimate_cost(gpu_type: str, config_size: str) -> CostEstimate | None:
    """Estimate the cost of a training run.

    Args:
        gpu_type: GPU type (e.g. "A100", "RTX_4090").
        config_size: Config name (e.g. "tiny", "small", "medium", "large").

    Returns:
        CostEstimate if we have reference data for the GPU, None otherwise.
    """
    price = REFERENCE_PRICES.get(gpu_type)
    times = TRAINING_TIME_ESTIMATES.get(gpu_type)

    if price is None:
        return None

    hours = times.get(config_size, 0.0) if times else 0.0
    return CostEstimate(
        gpu_type=gpu_type,
        config_size=config_size,
        estimated_hours=hours,
        cost_per_hour=price,
    )


def format_cost_summary(
    gpu_type: str,
    actual_hours: float,
    cost_per_hour: float,
) -> str:
    """Format a human-readable cost summary for display after a run.

    Args:
        gpu_type: GPU type used.
        actual_hours: Wall-clock hours the instance ran.
        cost_per_hour: Actual cost/hr from the provider.

    Returns:
        Multi-line string suitable for printing.
    """
    actual_cost = actual_hours * cost_per_hour
    lines = [
        "--- Cost Summary ---",
        f"  GPU:           {gpu_type}",
        f"  Duration:      {actual_hours:.2f} hours ({actual_hours * 60:.0f} min)",
        f"  Rate:          ${cost_per_hour:.2f}/hr",
        f"  Total cost:    ${actual_cost:.2f}",
    ]
    return "\n".join(lines)


def format_cost_estimate(estimate: CostEstimate) -> str:
    """Format a pre-run cost estimate for display.

    Args:
        estimate: CostEstimate from estimate_cost().

    Returns:
        Multi-line string suitable for printing.
    """
    lines = [
        "--- Cost Estimate ---",
        f"  GPU:           {estimate.gpu_type}",
        f"  Config:        {estimate.config_size}",
        f"  Est. duration: {estimate.estimated_hours:.1f} hours",
        f"  Rate:          ${estimate.cost_per_hour:.2f}/hr",
        f"  Est. cost:     ${estimate.estimated_cost:.2f}",
    ]
    return "\n".join(lines)
