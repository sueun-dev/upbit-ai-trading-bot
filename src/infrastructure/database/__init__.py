"""Database infrastructure module.

This module provides database utilities and helpers for the trading system.
"""

import os
from pathlib import Path


def get_db_path(db_name: str) -> str:
    """Get the full path to a database file.
    
    Args:
        db_name: Name of the database file (e.g., 'pattern_learning.db')
        
    Returns:
        Full path to the database file
    """
    # Get the directory where this module is located
    module_dir = Path(__file__).parent
    
    # Create data directory if it doesn't exist
    data_dir = module_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Return the full path to the database
    return str(data_dir / db_name)


# Export the function
__all__ = ['get_db_path']