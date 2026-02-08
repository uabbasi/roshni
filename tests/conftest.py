"""Shared test fixtures for roshni."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def tmp_config_file(tmp_dir):
    """Create a temporary YAML config file."""
    import yaml

    config_data = {
        "paths": {
            "data_dir": os.path.join(tmp_dir, "data"),
            "cache_dir": os.path.join(tmp_dir, "cache"),
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        },
    }
    config_path = os.path.join(tmp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path
