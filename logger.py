"""
utils/logger.py — Centralised logging setup.

Every module obtains its logger via get_logger(__name__) so the format
and level are controlled in one place and driven by Settings.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

from app.config import get_settings


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


@lru_cache(maxsize=1)
def _configure_root() -> None:
    """Configure the root logger once."""
    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring root is configured first."""
    _configure_root()
    return logging.getLogger(name)
