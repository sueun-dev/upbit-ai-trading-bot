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

# Constants
MAX_CONSECUTIVE_ERRORS = 5
AI_LEARNING_DAYS_BACK = 30

# Module-level logger
logger = logging.getLogger(__name__)

def initialize_trade_history_analyzer() -> AILearningSystem:
    """Initialize trade history analyzer with past data.
    
    Returns:
        AILearningSystem instance. On first run (empty database):
        - lessons_learned: []
        - pattern_statistics: {}
        
        With trade history:
        - lessons_learned: List with 2 insights (success_rate, failure_analysis)
        - pattern_statistics: Dict with per-symbol success rates
    """
    try:
        analyzer = AILearningSystem()
        insights = analyzer.analyze_historical_trades(days_back=AI_LEARNING_DAYS_BACK)
        
        # Log initial insights
        logger.info(f"📊 Trade history analyzer initialized with {len(insights)} insights")
        
        for insight in insights:
            if insight['type'] == 'success_rate':
                metrics = insight['metrics']
                logger.info(f"📈 Overall: {metrics['success_rate']}% success ({metrics['successful_trades']}/{metrics['total_trades']} trades)")
                if metrics['total_trades'] > 0:
                    logger.info(f"💰 Avg profit: {metrics.get('avg_profit', 0)}%, Avg loss: {metrics.get('avg_loss', 0)}%")
            
            elif insight['type'] == 'failure_analysis':
                metrics = insight['metrics']
                if metrics['total_failures'] > 0:
                    logger.info(f"⚠️ Failure rate: {metrics['failure_rate']}% - Most common: {insight['most_common_reason']}")
        
        return analyzer
        
    except Exception as e:
        logger.critical(f"❌ Trade history analyzer initialization FAILED: {e}")
        raise

def run_main_trading_loop(orchestrator: TradingOrchestrator) -> None:
    """Execute the main trading loop with error handling.
    
    Args:
        orchestrator: Trading orchestrator instance.
    """
    consecutive_errors = 0
    
    logger.info(f"⏰ Main trading cycle runs every {CHECK_INTERVAL_SECONDS} seconds")
    logger.info("✅ All systems operational - starting main trading loop")
    
    while True:
        try:
            orchestrator.run_trading_cycle()
            consecutive_errors = 0  # Reset on success
            
            logger.info("😴 Waiting %ds before next trading cycle...", CHECK_INTERVAL_SECONDS)
            
            # Simple sleep
            time.sleep(CHECK_INTERVAL_SECONDS)
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading loop stopped by user")
            break
            
        except Exception as exc:
            consecutive_errors += 1
            logger.error(
                "Unexpected error in main trading loop (%d/%d): %s",
                consecutive_errors,
                MAX_CONSECUTIVE_ERRORS,
                exc,
                exc_info=True
            )
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical(
                    "❌ Too many consecutive errors (%d). Stopping trading loop.",
                    consecutive_errors
                )
                break
            
            # Exponential backoff
            wait_time = min(CHECK_INTERVAL_SECONDS * (2 ** (consecutive_errors - 1)), 3600)
            logger.info("Waiting %ds before retry...", wait_time)
            
            # Simple sleep
            time.sleep(wait_time)

if __name__ == "__main__":
    try:
        setup_logging()
        
        logger.info(f"[START] Trading interval: {CHECK_INTERVAL_SECONDS} seconds")
        
        # Initialize trade history analyzer
        trade_analyzer = initialize_trade_history_analyzer()
        
        orchestrator = TradingOrchestrator(
            access_key=UPBIT_ACCESS_KEY,
            secret_key=UPBIT_SECRET_KEY,
            openai_api_key=OPENAI_API_KEY,
            trade_analyzer=trade_analyzer
        )
        
        logger.info("🚀 Starting Advanced AI Auto Trading System")
        
        # Run main trading loop
        run_main_trading_loop(orchestrator)
        
    except Exception as e:
            logger.critical(f"❌ Fatal error: {e}", exc_info=True)
            raise SystemExit(1)
