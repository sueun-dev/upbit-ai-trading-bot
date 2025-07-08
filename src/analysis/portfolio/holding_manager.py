"""Simple holding manager for tracking purchase records."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

HOLDINGS_FILE = "holdings.json"


def record_purchase(symbol: str, price: float, amount_krw: float = None) -> None:
    """Record a purchase for tracking purposes.
    
    Args:
        symbol: Cryptocurrency symbol
        price: Purchase price
        amount_krw: Amount in KRW (optional)
    """
    try:
        # Load existing holdings
        holdings = load_holdings()
        
        # Add new purchase record
        if symbol not in holdings:
            holdings[symbol] = []
        
        purchase_record = {
            "timestamp": datetime.now().isoformat(),
            "price": price,
            "amount_krw": amount_krw,
            "type": "buy"
        }
        
        holdings[symbol].append(purchase_record)
        
        # Save updated holdings
        save_holdings(holdings)
        
        logger.info("Recorded purchase: %s at %s KRW", symbol, price)
        
    except Exception as e:
        logger.error("Failed to record purchase for %s: %s", symbol, e)


def load_holdings() -> Dict[str, List[Dict]]:
    """Load holdings from file.
    
    Returns:
        Dictionary of holdings by symbol
    """
    try:
        if os.path.exists(HOLDINGS_FILE):
            with open(HOLDINGS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error("Failed to load holdings: %s", e)
        return {}


def save_holdings(holdings: Dict[str, List[Dict]]) -> None:
    """Save holdings to file.
    
    Args:
        holdings: Holdings dictionary to save
    """
    try:
        with open(HOLDINGS_FILE, 'w') as f:
            json.dump(holdings, f, indent=2)
    except Exception as e:
        logger.error("Failed to save holdings: %s", e)
