"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import PROJECT_ROOT


def setup_logger(level: str = "INFO") -> None:
    """Configure project-wide logging."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.remove()

    # Console output
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # File output with rotation
    logger.add(
        log_dir / "project.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} - {message}",
    )


def get_logger(name: str):
    """Get a named logger instance."""
    return logger.bind(name=name)
