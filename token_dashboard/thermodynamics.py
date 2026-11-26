"""Thermodynamic efficiency computation layer for LLM inference.

Preserves core E_in, E_out, W_useful, and eta formulas while adding:
- Reasoning / test-time-scaling token & multiplier support (Joule 2026)
- Confidence intervals (IQR low/high bounds)
- Rework loop penalty and latency tracking
All energy units in Joules.
"""

from __future__ import annotations

from dataclasses import dataclass


class MissingEnergyConstantsError(ValueError):
    """Raised when no energy constants exist for a given model or alias."""


@dataclass(frozen=True)
class EnergyConstants:
    """Per-model-GPU energy constants for LLM inference."""

    model: str
    gpu: str
    e_prefill: float  # Joules per input token
    e_decode: float  # Joules per output token
    kappa: float = 1.0  # Model scaling factor
    source: str = ""  # Citation for measurement / interpolation
    measured: bool = False  # False if estimated/interpolated
    is_reasoning: bool = False  # True for test-time-scaling models
    reasoning_multiplier: float = 1.0  # Applied when reasoning tokens are not explicitly provided
    iqr_low: float = 0.52  # Lower bound multiplier (Joule 2026 IQR 0.16/0.31)
    iqr_high: float = 1.94  # Upper bound multiplier (Joule 2026 IQR 0.60/0.31)
    quantization: str = "fp16"  # Assumed precision (fp16, fp8, int8, int4)


@dataclass(frozen=True)
class EfficiencyReport:
    """Thermodynamic efficiency diagnostics for an LLM response."""

    e_in: float  # Joules — median total energy consumed
    e_in_low: float  # Joules — lower IQR bound
    e_in_high: float  # Joules — upper IQR bound
    e_out: float  # Joules — decode-phase energy (kappa * n_out * e_decode)
    w_useful: float  # Joules — useful work (Q * e_out / rework_factor)
    q: float | None  # Quality gate score [0, 1], or None if unassessed
    quality_adjusted: bool  # True only if q was provided from real scoring
    eta: float  # Efficiency (w_useful / e_in), clamped to [0, 1]
    waste_joules: float  # Energy waste (e_in - w_useful)
    waste_share: float  # Waste share (1 - eta)
    rework_turns: int  # Number of retry/correction turns (1 = no rework)
    latency_ms: float | None  # Observed response latency in ms
    constants: EnergyConstants


def energy_input(
    n_in: int,
    n_out: int,
    c: EnergyConstants,
    n_reasoning: int = 0,
) -> float:
    """Calculate total energy input E_in in Joules.

    If explicit reasoning tokens (n_reasoning) are provided:
        E_in = kappa * (n_in * e_prefill + (n_out + n_reasoning) * e_decode)
    Otherwise, if c.is_reasoning is True:
        E_in = kappa * (n_in * e_prefill + n_out * e_decode * c.reasoning_multiplier)
    """
    if n_in < 0 or n_out < 0 or n_reasoning < 0:
        raise ValueError("Token counts must be non-negative")

    if n_reasoning > 0:
        decode_energy = (n_out + n_reasoning) * c.e_decode
    elif c.is_reasoning:
        decode_energy = n_out * c.e_decode * c.reasoning_multiplier
    else:
        decode_energy = n_out * c.e_decode

    return float(c.kappa * (n_in * c.e_prefill + decode_energy))


def useful_work(
    n_out: int,
    c: EnergyConstants,
    q: float,
    rework_turns: int = 1,
) -> float:
    """Calculate useful work W_useful in Joules, discounted by rework loops."""
    if n_out < 0:
        raise ValueError("Token counts must be non-negative")
    turns = max(1, int(rework_turns))
    q_clamped = max(0.0, min(1.0, float(q)))
    e_out = float(c.kappa * n_out * c.e_decode)
    return float((q_clamped * e_out) / turns)


def efficiency(
    n_in: int,
    n_out: int,
    c: EnergyConstants,
    q: float | None = None,
    n_reasoning: int = 0,
    rework_turns: int = 1,
    latency_ms: float | None = None,
) -> EfficiencyReport:
    """Compute thermodynamic efficiency, confidence bounds, and waste diagnostics."""
    if n_in < 0 or n_out < 0 or n_reasoning < 0:
        raise ValueError("Token counts must be non-negative")

    turns = max(1, int(rework_turns))
    quality_adjusted = q is not None
    q_eff = 1.0 if q is None else max(0.0, min(1.0, float(q)))

    e_in = energy_input(n_in, n_out, c, n_reasoning=n_reasoning)
    e_in_low = float(e_in * c.iqr_low)
    e_in_high = float(e_in * c.iqr_high)
    e_out = float(c.kappa * n_out * c.e_decode)
    w_useful = float((q_eff * e_out) / turns)

    if e_in <= 0.0:
        eta = 0.0
        waste_joules = 0.0
        waste_share = 0.0
    else:
        eta = max(0.0, min(1.0, w_useful / e_in))
        waste_joules = max(0.0, e_in - w_useful)
        waste_share = max(0.0, min(1.0, 1.0 - eta))

    return EfficiencyReport(
        e_in=e_in,
        e_in_low=e_in_low,
        e_in_high=e_in_high,
        e_out=e_out,
        w_useful=w_useful,
        q=q,
        quality_adjusted=quality_adjusted,
        eta=eta,
        waste_joules=waste_joules,
        waste_share=waste_share,
        rework_turns=turns,
        latency_ms=latency_ms,
        constants=c,
    )
