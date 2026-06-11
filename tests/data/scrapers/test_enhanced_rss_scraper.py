"""Tests for source scoring and freshness filters in the RSS scraper."""

import unittest
from datetime import datetime, timedelta, timezone

from src.data.scrapers.enhanced_rss_scraper import EnhancedRSSAggregator


class TestEnhancedRSSAggregatorQuality(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregator = EnhancedRSSAggregator()
        self.feed = {"name": "CoinDesk", "weight": 0.95}

    def test_build_article_scores_fresh_market_relevant_news(self) -> None:
        article = self.aggregator._build_article(
            title="Bitcoin ETF volume jumps as BTC breaks resistance",
            summary="Fresh market catalyst with BTC volume and ETF flows.",
            link="https://www.coindesk.com/markets/btc-etf?utm_source=test",
            published="",
            published_dt=datetime.now(timezone.utc) - timedelta(hours=1),
            feed_config=self.feed,
        )

        self.assertIsNotNone(article)
        assert article is not None
        self.assertGreater(article["quality_score"], 0.7)
        self.assertEqual(article["url"], "https://www.coindesk.com/markets/btc-etf")

    def test_stale_article_is_dropped(self) -> None:
        article = self.aggregator._build_article(
            title="Ethereum upgrade analysis for ETH traders",
            summary="Old but otherwise crypto related.",
            link="https://www.coindesk.com/markets/eth-upgrade",
            published="",
            published_dt=datetime.now(timezone.utc) - timedelta(days=4),
            feed_config=self.feed,
        )

        self.assertIsNone(article)

    def test_deduplicate_articles_uses_canonical_url(self) -> None:
        first = self.aggregator._build_article(
            title="Solana SOL mainnet upgrade drives trading volume",
            summary="SOL mainnet upgrade and exchange volume.",
            link="https://www.coindesk.com/markets/sol-upgrade?utm_campaign=a",
            published="",
            published_dt=datetime.now(timezone.utc),
            feed_config=self.feed,
        )
        second = self.aggregator._build_article(
            title="Solana SOL mainnet upgrade drives trading volume",
            summary="Duplicate article with tracking URL.",
            link="https://www.coindesk.com/markets/sol-upgrade?utm_campaign=b",
            published="",
            published_dt=datetime.now(timezone.utc),
            feed_config=self.feed,
        )

        assert first is not None
        assert second is not None
        unique = self.aggregator._deduplicate_articles([first, second])

        self.assertEqual(len(unique), 1)


if __name__ == "__main__":
    unittest.main()
