"""Portfolio manager with comprehensive error handling and AI analysis."""

import logging
from typing import Dict, Any, Optional

import pyupbit
from src.shared.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


# USED
def get_portfolio_status(upbit, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get portfolio status with comprehensive error handling and AI analysis."""
    try:
        # Get balances from Upbit
        balances = upbit.get_balances()
        
        # Initialize portfolio data
        krw_balance = 0.0
        crypto_assets = {}
        
        # Process each balance
        for item in balances:
            currency = item.get('currency', '')
            balance = float(item.get('balance', 0))
            
            if balance == 0:
                continue
            
            if currency == 'KRW':
                krw_balance = balance
            else:
                # Get price info
                avg_buy_price = float(item.get('avg_buy_price', 0))
                market_price = pyupbit.get_current_price(f'KRW-{currency}')
                current_price = float(market_price) if market_price else 0.0
                
                # Calculate value
                current_value_krw = balance * (current_price if current_price > 0 else avg_buy_price)
                
                crypto_assets[currency] = {
                    'balance': balance,
                    'avg_buy_price': avg_buy_price,
                    'current_price': current_price,
                    'current_value': current_value_krw,
                    'value_krw': current_value_krw,
                    'percent_of_total': 0.0
                }
        
        # Calculate total balance
        total_balance = krw_balance + sum(asset['value_krw'] for asset in crypto_assets.values())
        
        # Calculate percentages
        if total_balance > 0:
            for asset in crypto_assets.values():
                asset['percent_of_total'] = (asset['value_krw'] / total_balance) * 100
        
        # Initialize AI analysis result
        ai_analysis = {
            'risk_level': 'unknown',
            'diversification_score': 0.0,
            'insights': [],
            'suggested_actions': []
        }
        
        # Perform AI analysis - always required
        if not api_key:
            raise ValueError("OpenAI API key is required for portfolio analysis")
            
        if crypto_assets:
            openai_client = OpenAIClient(api_key=api_key)
            
            # Prepare portfolio summary
            cash_ratio = (krw_balance / total_balance * 100) if total_balance > 0 else 100
            holdings = [{
                'symbol': symbol,
                'percent_of_total': asset['percent_of_total'],
                'value_krw': asset['value_krw']
            } for symbol, asset in crypto_assets.items()]
            
            # Create AI prompt
            ai_prompt = f"""Analyze this cryptocurrency portfolio:

                Portfolio Overview:
                - Total Value: {total_balance:,.0f} KRW
                - Cash Ratio: {cash_ratio:.1f}%
                - Holdings: {len(holdings)}

                Holdings Detail:
                {chr(10).join([f"- {h['symbol']}: {h['percent_of_total']:.1f}% ({h['value_krw']:,.0f} KRW)" for h in holdings])}

                Return analysis in this exact JSON format:
                {{
                    "risk_level": "low/medium/high",
                    "diversification_score": 0.0-1.0,
                    "insights": ["portfolio observation", "market insight"],
                    "suggested_actions": ["specific action 1", "specific action 2", "specific action 3"]
                }}

                Example actions:
                - "Reduce BTC from 65% to 40%"
                - "Increase cash to 20%"
                - "Set -5% stop-loss on all positions"
                - "Take 30% profits on ETH"
            """

            result = openai_client.analyze_with_prompt(
                prompt=ai_prompt,
                system_message="You are a cryptocurrency portfolio analyst. Provide analysis in the exact JSON format requested."
            )
            
            ai_analysis.update(result)
        
        return {
            'total_balance': total_balance,
            'total_assets': total_balance,  # Add for compatibility
            'krw_balance': krw_balance,
            'available_krw': krw_balance,  # Add for compatibility
            'assets': crypto_assets,
            'ai_portfolio_analysis': ai_analysis,
            'last_updated': '2025-07-03T02:00:00Z',
            'metrics': {
                'total_value': total_balance,
                'asset_count': len(crypto_assets),
                'largest_holding': max([asset['percent_of_total'] for asset in crypto_assets.values()]) if crypto_assets else 0,
                'cash_ratio': (krw_balance / total_balance * 100) if total_balance > 0 else 100
            }
        }
        
    except Exception as e:
        logger.error(f"Portfolio status failed: {e}")
        raise