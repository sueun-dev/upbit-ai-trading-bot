import logging
import time

from src.analysis.ai_learning.ai_learning_system import AILearningSystem
from src.core.orchestrator.trading_orchestrator import TradingOrchestrator
from src.infrastructure.config.settings import (
    CHECK_INTERVAL_SECONDS,
    OPENAI_API_KEY,
    UPBIT_ACCESS_KEY,
    UPBIT_SECRET_KEY,
)
from src.shared.utils.helpers import setup_logging

MAX_CONSECUTIVE_ERRORS = 5
AI_LEARNING_DAYS_BACK = 30

logger = logging.getLogger(__name__)

def run_main_trading_loop(orchestrator: TradingOrchestrator) -> None:
    """Execute the main trading loop with error handling.
    
    Args:
        orchestrator: Trading orchestrator instance.
    """
    consecutive_errors = 0
    
    logger.info(f"⏰ Main trading cycle runs every {CHECK_INTERVAL_SECONDS} seconds")
    
    while True:
        try:
            orchestrator.run_trading_cycle()
            consecutive_errors = 0  # Reset on success
            
            logger.info("😴 Waiting %ds before next trading cycle...", CHECK_INTERVAL_SECONDS)
            
            time.sleep(CHECK_INTERVAL_SECONDS)
                
        except Exception as exc:
            consecutive_errors += 1
            logger.error(f"Error {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}: {exc}")
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical("❌ Too many errors. Stopping.")
                break
            
            time.sleep(min(CHECK_INTERVAL_SECONDS * consecutive_errors, 3600))

if __name__ == "__main__":
    try:
        setup_logging()
        
        trade_analyzer = AILearningSystem()
        insights = trade_analyzer.analyze_historical_trades(days_back=AI_LEARNING_DAYS_BACK)
        logger.info(f"📊 Trade analyzer: {len(insights)} insights")
        
        for insight in insights:
            if insight['type'] == 'success_rate' and insight.get('metrics', {}).get('total_trades', 0) > 0:
                m = insight['metrics']
                logger.info(f"📈 {m['success_rate']}% success ({m['successful_trades']}/{m['total_trades']})")
            elif insight['type'] == 'failure_analysis' and insight.get('metrics', {}).get('total_failures', 0) > 0:
                logger.info(f"⚠️ Failure: {insight.get('most_common_reason', 'Unknown')}")
        
        orchestrator = TradingOrchestrator(
            access_key=UPBIT_ACCESS_KEY,
            secret_key=UPBIT_SECRET_KEY,
            openai_api_key=OPENAI_API_KEY,
            trade_analyzer=trade_analyzer
        )
        
        run_main_trading_loop(orchestrator)
        
    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}", exc_info=True)
        raise SystemExit(1)
