"""Helpers for recording installed inference-engine versions."""

import importlib.metadata


def get_package_version(distribution_name: str) -> str:
    """Return an installed distribution's version, or ``"unknown"`` if absent."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
