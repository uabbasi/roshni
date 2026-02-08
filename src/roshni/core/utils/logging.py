"""
Logging configuration using loguru.

Provides a simple setup function that configures loguru with sensible defaults.
Consumers can call setup_logging() at app startup, or just use loguru directly.
"""

import sys

from loguru import logger


def setup_logging(
    level: str = "WARNING",
    log_file: str | None = None,
    fmt: str = "<level>[{level.name}]</level> {message}",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru with console and optional file output.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. If None, only logs to stderr.
        fmt: Loguru format string.
        rotation: Log file rotation size.
        retention: How long to keep rotated logs.
    """
    logger.remove()
    logger.add(sys.stderr, level=level, format=fmt)

    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation=rotation,
            retention=retention,
        )
