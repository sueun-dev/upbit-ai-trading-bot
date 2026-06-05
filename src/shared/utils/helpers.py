"""Shared logging and datetime helpers."""

import datetime
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Tuple


def setup_logging() -> None:
    """Configure logging settings for the application.

    Creates a logs directory and attaches a console handler plus a daily-rotating
    file handler (rotates at midnight) so a long-running bot does not keep writing
    to a single start-day file forever.
    """
    os.makedirs("logs", exist_ok=True)

    file_handler = TimedRotatingFileHandler(
        "logs/trading.log", when="midnight", backupCount=30, encoding="utf-8"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler,
        ],
    )


def get_formatted_datetime() -> Tuple[str, str]:
    """Return the current date and time in standard formats.

    Returns:
        tuple: (date string YYYY-MM-DD, time string HH:MM UTC)
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M UTC")
    return date_str, time_str
