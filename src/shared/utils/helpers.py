import datetime
import logging
import os
import sys


def setup_logging() -> None:
    """Configure logging settings for the application.
    
    Creates logs directory and sets up both console and file logging handlers.
    """
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/{datetime.datetime.now().strftime("%Y-%m-%d")}.log')
        ]
    )


def get_formatted_datetime() -> tuple:
    """Return the current date and time in standard formats.

    Returns:
        tuple: (date string YYYY-MM-DD, time string HH:MM UTC)
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M UTC")
    return date_str, time_str