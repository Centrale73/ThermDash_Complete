"""Environmental impact adapters using EcoLogits (cloud APIs) and CodeCarbon (local compute).

Design rules:
- Cloud inference estimates come from EcoLogits (preserving version, assumptions, and warnings).
- Local compute measurements come from CodeCarbon (CPU/GPU/RAM; excludes disk/network/cooling).
- Unsupported models/providers return status='not_estimated' (never silently coerced to 0.0).
- Environmental estimates are strictly separated from financial cost calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import time
from typing import Any


@dataclass(frozen=True)
class CloudEnvironmentalEstimate:
    provider: str
    model: str
    status: str  # "estimated" or "not_estimated"
    energy_kwh: float | None = None
    gwp_kgco2eq: float | None = None
    library: str = "ecologits"
    library_version: str = "not_installed"
    warnings: list[str] = field(default_factory=list)
    assumptions: str = (
        "EcoLogits models cloud inference using H100/16-bit/vLLM/fixed-batch assumptions "
        "and excludes non-text modalities, TPUs, networking, and storage."
    )


@dataclass(frozen=True)
class LocalEnergyMeasurement:
    status: str  # "measured" or "not_measured"
    energy_kwh: float | None = None
    emissions_kgco2eq: float | None = None
    duration_s: float = 0.0
    library: str = "codecarbon"
    library_version: str = "not_installed"
    boundary_note: str = (
        "CodeCarbon tracks local CPU, GPU, and RAM power (RAPL/NVML or load fallback); "
        "excludes disk I/O, networking, displays, and external cooling."
    )
    error: str | None = None


def estimate_cloud_impact(
    provider: str,
    model_name: str,
    output_tokens: int,
    request_latency_s: float = 1.0,
) -> CloudEnvironmentalEstimate:
    """Estimate cloud LLM environmental impact via EcoLogits if installed and supported."""
    try:
        version = importlib.metadata.version("ecologits")
        from ecologits.tracers.utils import llm_impacts  # type: ignore
    except Exception:
        return CloudEnvironmentalEstimate(
            provider=provider,
            model=model_name,
            status="not_estimated",
            warnings=["ecologits package is not installed; install with `pip install ecologits`."],
        )

    provider_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "mistral": "mistralai",
        "google": "google_genai",
    }
    eco_provider = provider_map.get(provider.lower())
    if not eco_provider:
        return CloudEnvironmentalEstimate(
            provider=provider,
            model=model_name,
            status="not_estimated",
            library_version=version,
            warnings=[f"Provider '{provider}' is not mapped in EcoLogits."],
        )

    try:
        impacts = llm_impacts(
            provider=eco_provider,
            model_name=model_name,
            output_token_count=max(1, int(output_tokens)),
            request_latency=max(0.01, float(request_latency_s)),
        )
        if impacts is None:
            return CloudEnvironmentalEstimate(
                provider=provider,
                model=model_name,
                status="not_estimated",
                library_version=version,
                warnings=[f"Model '{model_name}' is not supported by EcoLogits {version}."],
            )

        energy_val = getattr(impacts.energy, "value", None)
        gwp_val = getattr(impacts.gwp, "value", None)
        if isinstance(energy_val, object) and hasattr(energy_val, "min") and hasattr(energy_val, "max"):
            energy_kwh = float((energy_val.min + energy_val.max) / 2.0)
        else:
            energy_kwh = float(energy_val) if energy_val is not None else None

        if isinstance(gwp_val, object) and hasattr(gwp_val, "min") and hasattr(gwp_val, "max"):
            gwp_kg = float((gwp_val.min + gwp_val.max) / 2.0)
        else:
            gwp_kg = float(gwp_val) if gwp_val is not None else None

        warns = [str(w) for w in (getattr(impacts, "warnings", None) or [])]
        return CloudEnvironmentalEstimate(
            provider=eco_provider,
            model=model_name,
            status="estimated" if energy_kwh is not None else "not_estimated",
            energy_kwh=energy_kwh,
            gwp_kgco2eq=gwp_kg,
            library_version=version,
            warnings=warns,
        )
    except Exception as exc:
        return CloudEnvironmentalEstimate(
            provider=eco_provider,
            model=model_name,
            status="not_estimated",
            library_version=version,
            warnings=[f"EcoLogits estimation unavailable: {exc}"],
        )


class LocalCodeCarbonTracker:
    """Context manager to measure local compute energy/emissions with CodeCarbon."""

    def __init__(self, project_name: str = "thermdash_local_eval"):
        self.project_name = project_name
        self._tracker: Any = None
        self._start: float = 0.0
        self.result: LocalEnergyMeasurement | None = None

    def __enter__(self) -> "LocalCodeCarbonTracker":
        self._start = time.perf_counter()
        try:
            from codecarbon import EmissionsTracker  # type: ignore

            self._tracker = EmissionsTracker(
                project_name=self.project_name,
                save_to_file=False,
                log_level="error",
            )
            self._tracker.start()
        except Exception as exc:
            self._tracker = None
            self.result = LocalEnergyMeasurement(
                status="not_measured",
                error=f"CodeCarbon unavailable: {exc}",
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.perf_counter() - self._start
        if self._tracker is None:
            if self.result is None:
                self.result = LocalEnergyMeasurement(status="not_measured", duration_s=duration)
            return
        try:
            version = importlib.metadata.version("codecarbon")
            emissions = self._tracker.stop()
            energy = getattr(self._tracker, "_total_energy", None)
            energy_kwh = float(energy.kWh) if energy is not None and hasattr(energy, "kWh") else None
            self.result = LocalEnergyMeasurement(
                status="measured",
                energy_kwh=energy_kwh,
                emissions_kgco2eq=float(emissions) if emissions is not None else None,
                duration_s=duration,
                library_version=version,
            )
        except Exception as exc:
            self.result = LocalEnergyMeasurement(
                status="not_measured",
                duration_s=duration,
                error=f"CodeCarbon stop failed: {exc}",
            )
