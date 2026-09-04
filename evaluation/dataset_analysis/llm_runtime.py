"""Small shared helpers for model-backed dataset-analysis commands."""

from __future__ import annotations

import math
from typing import Any, Iterable

from dsh_plugin.agent_eval.credentials import load_openai_key_from_configured_env


def configured_openai_key() -> bool:
    """Load the operator-selected OpenAI key without persisting or logging it."""

    return load_openai_key_from_configured_env()


def response_usage(response: Any) -> dict[str, int]:
    """Return the common token counters exposed by an OpenAI response."""

    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(
            getattr(input_details, "cached_tokens", 0) or 0
        ),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(
            getattr(output_details, "reasoning_tokens", 0) or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    """Compute a linearly interpolated percentile, returning zero if empty."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (
        position - lower
    )
