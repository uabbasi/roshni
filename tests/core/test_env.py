"""Tests for roshni.core.env."""

from roshni.core.env import is_cloud_environment, is_laptop


def test_default_is_laptop(monkeypatch):
    for var in ("IN_DOCKER", "CLOUD_RUN", "K_SERVICE", "FUNCTION_NAME"):
        monkeypatch.delenv(var, raising=False)
    assert is_laptop() is True
    assert is_cloud_environment() is False


def test_cloud_markers(monkeypatch):
    for var in ("IN_DOCKER", "CLOUD_RUN", "K_SERVICE", "FUNCTION_NAME"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("IN_DOCKER", "1")
    assert is_cloud_environment() is True
    assert is_laptop() is False


def test_cloud_run_marker(monkeypatch):
    for var in ("IN_DOCKER", "CLOUD_RUN", "K_SERVICE", "FUNCTION_NAME"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("K_SERVICE", "my-service")
    assert is_cloud_environment() is True
