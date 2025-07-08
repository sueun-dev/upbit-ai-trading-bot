import datetime
import json
import os
from decimal import Decimal, ROUND_DOWN


def get_formatted_datetime() -> tuple:
    """Return the current date and time in standard formats.

    Returns:
        tuple: (date string YYYY-MM-DD, time string HH:MM UTC)
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M UTC")
    return date_str, time_str


def truncate_float(val: float, decimal: int = 6) -> float:
    """Truncate a float to a fixed number of decimal places.

    Args:
        val: The float value to truncate.
        decimal: Number of decimal places to keep.

    Returns:
        The truncated float value, where any digits beyond the specified
        decimal places are removed (no rounding), 
        e.g. truncate_float(1.23456789, 4) -> 1.2345.
    """
    d = Decimal(str(val))
    quantize_str = '1.' + '0' * decimal
    return float(d.quantize(Decimal(quantize_str), rounding=ROUND_DOWN))


def load_json_file(path: str) -> dict:
    """Load and return JSON data from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a dict, or empty dict if file does not exist.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: dict) -> None:
    """Save a dict as JSON to the specified file path.

    Args:
        path: Destination file path.
        data: Dictionary to serialize to JSON.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
