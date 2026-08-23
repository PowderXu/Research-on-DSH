"""Compatibility boundary for the pinned PyPI SkillOpt command modules."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from types import ModuleType
from typing import Any


SKILLOPT_VERSION = "0.2.0"
_CLI_ENTRY_POINTS = {
    "train": "skillopt-train",
    "eval_only": "skillopt-eval",
}


def load_skillopt_cli(command: str) -> ModuleType:
    """Load a pinned SkillOpt CLI module through its installed entry point."""

    entry_name = _CLI_ENTRY_POINTS.get(command)
    if entry_name is None:
        raise ValueError(f"unsupported SkillOpt command: {command}")

    try:
        installed = distribution("skillopt")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"SkillOpt {SKILLOPT_VERSION} is required; install the optimizer extra"
        ) from exc
    if installed.version != SKILLOPT_VERSION:
        raise RuntimeError(
            f"SkillOpt {SKILLOPT_VERSION} is required, found {installed.version}"
        )

    matches = [entry for entry in installed.entry_points if entry.name == entry_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"SkillOpt {SKILLOPT_VERSION} must expose one {entry_name!r} entry point"
        )
    module_name, separator, function_name = matches[0].value.partition(":")
    if not separator or function_name != "main":
        raise RuntimeError(f"unsupported SkillOpt entry point: {matches[0].value}")

    module = import_module(module_name)
    registry = getattr(module, "_ENV_REGISTRY", None)
    if not isinstance(registry, dict):
        raise RuntimeError(
            f"SkillOpt {SKILLOPT_VERSION} no longer exposes the expected environment registry"
        )
    return module


def register_environment(module: ModuleType, name: str, adapter: type[Any]) -> None:
    """Register one local adapter against the checked private compatibility seam."""

    module._ENV_REGISTRY[name] = adapter
