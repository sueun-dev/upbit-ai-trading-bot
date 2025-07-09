"""Multi-source news scraper that aggregates from multiple news sources."""
import logging
from typing import List, Dict, Any

from .enhanced_rss_scraper import enhanced_rss_aggregator

# RSS collection constants
RSS_MULTIPLIER = 2  # Collect more articles from RSS
RSS_SUFFICIENT_THRESHOLD_DIVISOR = 2  # Sufficient RSS articles as fraction of max

# Default settings
DEFAULT_MAX_ARTICLES = 30

logger = logging.getLogger(__name__)


class MultiSourceNewsScraper:
    """Aggregates news from multiple cryptocurrency news sources using RSS."""
    
    def __init__(self) -> None:
        """Initialize the multi-source news scraper."""
        pass
    
    # USED
    def fetch_all_news(self, max_total_articles: int = DEFAULT_MAX_ARTICLES) -> List[Dict[str, Any]]:
        """Fetch news from all available sources.
        
        Uses enhanced RSS aggregator as the primary (and only) source.
        
        Args:
            max_total_articles: Maximum total articles to return.
            
        Returns:
            List of news articles from all sources.
        """        
        try:
            rss_articles = enhanced_rss_aggregator.fetch_all_news(
                max_total_articles * RSS_MULTIPLIER
            )
            
            if rss_articles and len(rss_articles) >= max_total_articles // RSS_SUFFICIENT_THRESHOLD_DIVISOR:
                logger.info("✅ RSS 수집 성공: %d개 기사", len(rss_articles))
                return rss_articles
            
            logger.info("⚠️ RSS 수집 부족(%d개)", len(rss_articles))
            return rss_articles
                
        except Exception as e:
            logger.error("❌ 뉴스 수집 전체 실패: %s", e)
            return []

# Global instance for backward compatibility
multi_source_scraper = MultiSourceNewsScraper()