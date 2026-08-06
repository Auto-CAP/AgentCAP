import importlib.metadata

from agent_cap.utils.package_version import get_package_version


def test_get_package_version_returns_installed_version(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")

    assert get_package_version("vllm") == "1.2.3"


def test_get_package_version_returns_unknown_when_distribution_is_absent(monkeypatch):
    def missing_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)

    assert get_package_version("sglang") == "unknown"
