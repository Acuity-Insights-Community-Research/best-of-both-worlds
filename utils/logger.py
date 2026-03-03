"""
Logging utility for SJT AI Rating System.
Provides consistent logging setup with file and console handlers.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__ or script name)
        log_dir: Directory for log files. Defaults to 'logs' in project root.
        level: Logging level (default: INFO)
        console: Whether to also log to console (default: True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create log directory if it doesn't exist
    if log_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_filename = f"{name}_{timestamp}.log"
    log_path = log_dir / log_filename
    
    # File handler with detailed formatting
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with simpler formatting
    if console:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_path}")
    return logger

