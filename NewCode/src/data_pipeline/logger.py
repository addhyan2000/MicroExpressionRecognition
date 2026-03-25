# -*- coding: utf-8 -*-
"""Centralized logging configuration for the preprocessing pipeline.

Usage::

    from logger import get_pipeline_logger

    logger = get_pipeline_logger(log_file="preprocessing.log")
    logger.info("Pipeline started.")
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union


_LOGGER_NAME: str = "mer_preprocessing"
_LOCK = threading.Lock()


def get_pipeline_logger(
    log_file: Optional[Union[str, Path]] = None,
    logger_name: str = _LOGGER_NAME,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Create (or retrieve) the canonical pipeline logger.

    On the first call the logger is configured with:
    * A **console** handler at ``console_level`` (default ``INFO``).
    * A **rotating file** handler (RotatingFileHandler) at ``file_level`` (default ``DEBUG``) 
      using the provided ``max_bytes`` and ``backup_count``.

    This function is thread-safe. Subsequent calls with the same ``logger_name`` 
    return the already-configured logger without adding duplicate handlers.

    Args:
        log_file: Path to the log file (string or Path object). If ``None``, 
            only the console handler is attached.
        logger_name: Name used for ``logging.getLogger``. Defaults to
            ``"mer_preprocessing"``.
        console_level: Minimum severity routed to ``stderr``.
        file_level: Minimum severity written to the log file.
        max_bytes: Maximum size of a single log file before rotation.
        backup_count: Number of historical log files to retain.

    Returns:
        A fully configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(logger_name)

    with _LOCK:
        # Guard against duplicate handler attachment on repeated calls.
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter.
        logger.propagate = False

        # ── Console handler ───────────────────────────────────────────────
        console_fmt = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # ── Rotating File handler ─────────────────────────────────────────
        if log_file is not None:
            try:
                log_path = Path(log_file).resolve()
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_fmt = logging.Formatter(
                    fmt=(
                        "%(asctime)s │ %(levelname)-8s │ %(name)s │ "
                        "%(filename)s:%(lineno)d │ %(funcName)s │ %(message)s"
                    ),
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                
                file_handler = RotatingFileHandler(
                    filename=str(log_path),
                    mode="a",
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(file_level)
                file_handler.setFormatter(file_fmt)
                logger.addHandler(file_handler)
                
            except (OSError, PermissionError) as e:
                # Thread-Safe Warning: Log to console if file setup fails.
                logger.warning(
                    f"Failed to initialize rotating file handler at '{log_file}': {e}. "
                    "Only console logging will be active."
                )

    return logger
