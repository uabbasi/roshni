"""
Environment detection utilities.

Detects whether code is running on a local machine or in a cloud environment
(Docker / Cloud Run / Lambda). Enables environment-specific behavior without
changing business logic.
"""

import os


def is_cloud_environment() -> bool:
    """
    Check if running in a cloud environment (Docker/Cloud Run).

    Checks for IN_DOCKER, CLOUD_RUN, K_SERVICE (Cloud Run/Functions),
    and FUNCTION_NAME (Cloud Functions) environment variables.
    """
    cloud_markers = ("IN_DOCKER", "CLOUD_RUN", "K_SERVICE", "FUNCTION_NAME")
    return any(os.environ.get(var) for var in cloud_markers)


def is_laptop() -> bool:
    """Check if running on a local machine (not in cloud)."""
    return not is_cloud_environment()
