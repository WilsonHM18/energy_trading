"""Logging configuration using Loguru.

Call ``configure_logging()`` once at the entry point of any script or CLI.
All library modules use ``from loguru import logger`` directly no further
setup is required inside library code.

Usage::

    from energy_trading.utils.logging import configure_logging
    configure_logging(level="INFO")
"""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "<level>{message}</level>"
)


def configure_logging(level: str = "INFO") -> None:
    """Configure the global Loguru logger.

    This function is idempotent: subsequent calls with the same level are
    no-ops.  Calling with a different level reconfigures the handler.

    Args:
        level: A valid Loguru log level string (e.g. ``"DEBUG"``, ``"INFO"``).
    """
    global _CONFIGURED

    logger.remove()  # Remove the default handler.
    logger.add(
        sys.stderr,
        format=_FORMAT,
        level=level.upper(),
        colorize=True,
        enqueue=True,  # Thread-safe async sink.
    )
    _CONFIGURED = True
    logger.debug("Logging configured at level={}", level.upper())
