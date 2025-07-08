"""Scraper package for news collection and processing.

This package contains modules for scraping cryptocurrency news
from various sources.
"""

import logging

# Suppress feedparser's internal logging to reduce verbosity
logging.getLogger('feedparser').setLevel(logging.ERROR)

__all__ = [] 