"""Enhanced RSS news scraper with robust error handling and fallback mechanisms.

This module provides a more resilient RSS feed scraper that can handle:
- Malformed XML
- Encoding issues
- Partial feed failures
- Various RSS/Atom formats
"""

import concurrent.futures
import logging
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import chardet
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress chardet's verbose debug logs; we only want warnings/errors.
logging.getLogger("chardet").setLevel(logging.WARNING)
logging.getLogger("chardet.charsetprober").setLevel(logging.WARNING)
logging.getLogger("chardet.universaldetector").setLevel(logging.WARNING)

# HTTP Configuration
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.3

# Parallel processing
MAX_WORKERS = 6

# Article limits
MAX_ARTICLES_PER_FEED = 10
DEFAULT_MAX_ARTICLES = 20
MAX_ARTICLE_AGE_HOURS = 36
MIN_ARTICLE_QUALITY_SCORE = 0.38
MIN_TITLE_LENGTH = 12

# Source quality scores are intentionally conservative. Lower-tier sources can
# still enter the analysis, but only when they are fresh and clearly relevant.
SOURCE_QUALITY_SCORES = {
    "CoinDesk": 1.0,
    "The Block": 0.98,
    "CoinTelegraph": 0.94,
    "Decrypt": 0.9,
    "Blockworks": 0.9,
    "CryptoSlate": 0.82,
    "BeInCrypto": 0.78,
    "The Daily Hodl": 0.72,
    "NewsBTC": 0.68,
    "Bitcoinist": 0.66,
    "CryptoPotato": 0.64,
    "CryptoNews": 0.64,
    "AMBCrypto": 0.58,
    "U.Today": 0.58,
    "Coin Journal": 0.55,
    "CryptoNewsZ": 0.5,
}

MARKET_MOVING_KEYWORDS = {
    "listing": 0.22,
    "listed": 0.22,
    "delisting": 0.25,
    "hack": 0.2,
    "exploit": 0.2,
    "etf": 0.18,
    "sec": 0.16,
    "lawsuit": 0.15,
    "regulation": 0.15,
    "partnership": 0.12,
    "upgrade": 0.12,
    "mainnet": 0.12,
    "airdrop": 0.1,
    "tokenomics": 0.1,
    "whale": 0.08,
    "volume": 0.08,
}

# Cryptocurrency keywords
CRYPTO_KEYWORDS = [
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "crypto",
    "cryptocurrency",
    "blockchain",
    "defi",
    "nft",
    "web3",
    "altcoin",
    "trading",
    "binance",
    "coinbase",
    "upbit",
    "exchange",
    "wallet",
    "mining",
    "staking",
    "yield",
    "dapp",
    "smart contract",
    "ripple",
    "xrp",
    "cardano",
    "ada",
    "solana",
    "sol",
    "polkadot",
    "dot",
    "chainlink",
    "link",
    "dogecoin",
    "doge",
    "market",
    "price",
    "bull",
    "bear",
    "sec",
    "regulation",
]

logger = logging.getLogger(__name__)


class EnhancedRSSAggregator:
    """Enhanced RSS aggregator with robust error handling."""

    def __init__(self) -> None:
        """Initialize enhanced RSS aggregator."""
        self.rss_feeds = self._initialize_feeds()
        self.session = self._create_resilient_session()
        self._user_agent_index = 0

    def _initialize_feeds(self) -> Dict[str, Dict[str, Any]]:
        """Initialize RSS feeds with enhanced configurations."""
        return {
            "cointelegraph": {
                "url": "https://cointelegraph.com/rss",
                "name": "CoinTelegraph",
                "weight": 1.0,
                "encoding_hint": "utf-8",
                "parser_hints": {"lenient": True},
            },
            "coindesk": {
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "name": "CoinDesk",
                "weight": 0.95,
                "encoding_hint": "utf-8",
                "parser_hints": {"lenient": True, "sanitize": True},
                "fallback_url": "https://www.coindesk.com/feed/",
            },
            "decrypt": {
                "url": "https://decrypt.co/feed",
                "name": "Decrypt",
                "weight": 0.9,
                "encoding_hint": "utf-8",
                "parser_hints": {"lenient": True},
            },
            "theblock": {
                "url": "https://www.theblock.co/rss.xml",
                "name": "The Block",
                "weight": 0.9,
                "encoding_hint": "utf-8",
            },
            "bitcoinist": {
                "url": "https://bitcoinist.com/feed/",
                "name": "Bitcoinist",
                "weight": 0.85,
                "encoding_hint": "utf-8",
            },
            "cryptoslate": {
                "url": "https://cryptoslate.com/feed/",
                "name": "CryptoSlate",
                "weight": 0.85,
                "encoding_hint": "utf-8",
            },
            "dailyhodl": {
                "url": "https://dailyhodl.com/feed/",
                "name": "The Daily Hodl",
                "weight": 0.8,
                "encoding_hint": "utf-8",
            },
            "cryptopotato": {
                "url": "https://cryptopotato.com/feed/",
                "name": "CryptoPotato",
                "weight": 0.75,
                "encoding_hint": "utf-8",
                "parser_hints": {"lenient": True},
            },
            "beincrypto": {
                "url": "https://beincrypto.com/feed/",
                "name": "BeInCrypto",
                "weight": 0.75,
                "encoding_hint": "utf-8",
            },
            "coinjournal": {
                "url": "https://coinjournal.net/news/feed/",  # Updated URL
                "name": "Coin Journal",
                "weight": 0.7,
                "encoding_hint": "utf-8",
            },
            "newsbtc": {
                "url": "https://www.newsbtc.com/feed/",
                "name": "NewsBTC",
                "weight": 0.7,
                "encoding_hint": "utf-8",
            },
            "blockworks": {
                "url": "https://blockworks.co/feed",
                "name": "Blockworks",
                "weight": 0.85,
                "encoding_hint": "utf-8",  # Force UTF-8
                "parser_hints": {"sanitize": True, "encoding_override": "utf-8"},
            },
            "cryptonews": {
                "url": "https://cryptonews.com/news/feed/",
                "name": "CryptoNews",
                "weight": 0.8,
                "encoding_hint": "utf-8",
            },
            "ambcrypto": {
                "url": "https://ambcrypto.com/feed/",
                "name": "AMBCrypto",
                "weight": 0.7,
                "encoding_hint": "utf-8",
            },
            "u_today": {
                "url": "https://u.today/rss",
                "name": "U.Today",
                "weight": 0.75,
                "encoding_hint": "utf-8",
            },
            "cryptonewsz": {
                "url": "https://www.cryptonewsz.com/feed/",
                "name": "CryptoNewsZ",
                "weight": 0.65,
                "encoding_hint": "utf-8",
                "parser_hints": {"lenient": True},
            },
        }

    def _create_resilient_session(self) -> requests.Session:
        """Create a resilient HTTP session with retry logic."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_next_user_agent(self) -> str:
        """Rotate user agents to avoid blocking."""
        agent = USER_AGENTS[self._user_agent_index]
        self._user_agent_index = (self._user_agent_index + 1) % len(USER_AGENTS)
        return agent

    def fetch_all_news(
        self, max_articles: int = DEFAULT_MAX_ARTICLES
    ) -> List[Dict[str, Any]]:
        """Fetch news from all RSS feeds with enhanced error handling."""
        logger.info("🚀 Starting enhanced RSS news collection...")
        all_articles = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_feed = {}

            # Submit all feed fetch tasks
            for feed_key, feed_config in self.rss_feeds.items():
                future = executor.submit(
                    self._try_fetch_feed, feed_config["url"], feed_config
                )
                future_to_feed[future] = feed_key

            # Collect results
            for future in concurrent.futures.as_completed(future_to_feed):
                feed_key = future_to_feed[future]
                feed_name = self.rss_feeds[feed_key]["name"]

                try:
                    articles = future.result()
                    if articles:
                        logger.info(
                            f"✅ {feed_name}: {len(articles)} articles collected"
                        )
                        all_articles.extend(articles)
                    else:
                        logger.warning(f"⚠️ {feed_name}: No articles collected")
                except Exception as e:
                    logger.error(f"❌ {feed_name}: Collection failed - {e}")

        # Process, deduplicate and rank by source quality, freshness and relevance.
        unique_articles = self._deduplicate_articles(all_articles)
        sorted_articles = self._rank_articles(unique_articles)

        final_articles = sorted_articles[:max_articles]
        logger.info(f"✅ Total articles collected: {len(final_articles)}")

        return final_articles

    def _try_fetch_feed(
        self, url: str, feed_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Try to fetch and parse a feed URL."""
        try:
            # Fetch raw content
            headers = {
                "User-Agent": self._get_next_user_agent(),
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }

            response = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            # Detect and fix encoding
            raw_content = response.content
            encoding_hint = feed_config.get("encoding_hint", "utf-8")

            # Try to detect encoding (confidence is None for empty/undetectable content)
            detected = chardet.detect(raw_content)
            confident = (detected.get("confidence") or 0) > 0.7
            encoding = detected["encoding"] if confident else encoding_hint

            content = raw_content.decode(encoding or "utf-8", errors="replace")

            # Fix common XML issues
            content = self._sanitize_xml(content)

            # Try parsing with feedparser first
            articles = self._parse_with_feedparser(content, feed_config)
            if articles:
                return articles

            # Fallback to custom XML parser
            articles = self._parse_with_xml(content, feed_config)
            if articles:
                return articles

            # Last resort: regex extraction
            articles = self._parse_with_regex(content, feed_config)
            return articles

        except Exception as e:
            logger.error(f"Error processing {feed_config['name']}: {e}")
            return []

    def _sanitize_xml(self, text: str) -> str:
        """Sanitize XML content to fix common issues."""
        # Remove BOM
        text = text.lstrip("\ufeff")

        # Fix CDATA sections
        text = re.sub(
            r"<!\[CDATA\[(.*?)\]\]>",
            lambda m: unescape(m.group(1)),
            text,
            flags=re.DOTALL,
        )

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

        # Fix mismatched tags (basic)
        text = re.sub(r"<([^/>]+)>([^<]*)</\1(?:[^>]*)>", r"<\1>\2</\1>", text)

        return text

    def _parse_with_feedparser(
        self, content: str, feed_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse feed with feedparser library."""
        # Parse with feedparser
        parser_hints = feed_config.get("parser_hints", {})

        if parser_hints.get("sanitize"):
            content = self._sanitize_xml(content)

        # Suppress feedparser warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feed = feedparser.parse(content)

        # Always try to extract articles even if bozo
        if hasattr(feed, "entries") and feed.entries:
            articles = []
            for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
                article = self._extract_article_from_entry(entry, feed_config)
                if article:
                    articles.append(article)

            if articles:
                return articles

        # No articles found
        return []

    def _parse_with_xml(
        self, content: str, feed_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse feed with XML ElementTree as fallback."""
        try:
            # Try to parse as XML
            root = ET.fromstring(content)

            # Find all item elements (RSS)
            items = root.findall(".//item")

            articles = []
            for item in items[:MAX_ARTICLES_PER_FEED]:
                article = self._extract_article_from_xml(item, feed_config)
                if article:
                    articles.append(article)

            return articles

        except ET.ParseError:
            logger.debug(
                f"XML parsing failed for {feed_config['name']}, trying regex extraction..."
            )
            return []

    def _parse_with_regex(
        self, content: str, feed_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Last resort: extract articles with regex."""
        try:
            articles = []

            # Find all item blocks
            item_pattern = r"<item>(.*?)</item>"
            items = re.findall(item_pattern, content, re.DOTALL | re.IGNORECASE)

            for item_content in items[:MAX_ARTICLES_PER_FEED]:
                article = self._extract_article_from_regex(item_content, feed_config)
                if article:
                    articles.append(article)

            return articles

        except Exception as e:
            logger.debug(f"Regex extraction also failed for {feed_config['name']}: {e}")
            return []

    def _extract_article_from_entry(
        self, entry: Any, feed_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract article from feedparser entry."""
        try:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            summary = entry.get("summary", "").strip()

            published = entry.get("published", "")
            published_dt = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_parts = entry.published_parsed
                published_dt = datetime(
                    published_parts[0],
                    published_parts[1],
                    published_parts[2],
                    published_parts[3],
                    published_parts[4],
                    published_parts[5],
                    tzinfo=timezone.utc,
                )

            return self._build_article(
                title=title,
                summary=summary,
                link=link,
                published=published,
                feed_config=feed_config,
                published_dt=published_dt,
            )

        except Exception as e:
            logger.debug(f"Error extracting article: {e}")
            return None

    def _extract_article_from_xml(
        self, item: ET.Element, feed_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract article from XML element."""
        try:
            # Extract RSS fields
            title = self._get_xml_text(item, ["title"])
            link = self._get_xml_text(item, ["link", "guid"])
            summary = self._get_xml_text(item, ["description", "summary"])
            published = self._get_xml_text(item, ["pubDate", "published"])

            return self._build_article(
                title=title,
                summary=summary,
                link=link,
                published=published,
                feed_config=feed_config,
            )

        except Exception as e:
            logger.debug(f"Error extracting from XML: {e}")
            return None

    def _extract_article_from_regex(
        self, item_content: str, feed_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract article using regex patterns."""
        try:
            # Extract fields with regex
            title = self._extract_with_regex(item_content, r"<title>(.*?)</title>")
            link = self._extract_with_regex(item_content, r"<link>(.*?)</link>")
            summary = self._extract_with_regex(
                item_content, r"<description>(.*?)</description>"
            )
            published = self._extract_with_regex(
                item_content, r"<pubDate>(.*?)</pubDate>"
            )

            return self._build_article(
                title=title,
                summary=summary,
                link=link,
                published=published,
                feed_config=feed_config,
            )

        except Exception as e:
            logger.debug(f"Error extracting with regex: {e}")
            return None

    def _get_xml_text(self, element: ET.Element, tags: List[str]) -> str:
        """Get text from XML element trying multiple tag names."""
        for tag in tags:
            elem = element.find(tag)
            if elem is not None:
                if tag.endswith("link") and elem.get("href"):
                    return elem.get("href", "")
                return elem.text or ""
        return ""

    def _extract_with_regex(self, text: str, pattern: str) -> str:
        """Extract text using regex pattern."""
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _is_crypto_related(self, text: str) -> bool:
        """Check if text contains cryptocurrency keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in CRYPTO_KEYWORDS)

    def _build_article(
        self,
        title: str,
        summary: str,
        link: str,
        published: str,
        feed_config: Dict[str, Any],
        published_dt: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a normalized, scored article or drop low-quality items."""
        clean_title = self._clean_text(title)
        clean_summary = self._clean_text(summary)
        clean_url = self._canonicalize_url(link)

        if len(clean_title) < MIN_TITLE_LENGTH or not clean_url:
            return None

        combined_text = f"{clean_title} {clean_summary}"
        if not self._is_crypto_related(combined_text):
            return None

        parsed_dt = published_dt or self._parse_published_datetime(published)
        if parsed_dt and self._article_age_hours(parsed_dt) > MAX_ARTICLE_AGE_HOURS:
            return None

        source = feed_config["name"]
        source_score = self._source_quality_score(source, feed_config)
        freshness_score = self._freshness_score(parsed_dt)
        relevance_score = self._relevance_score(combined_text)
        url_score = 1.0 if self._url_domain(clean_url) else 0.0

        quality_score = (
            source_score * 0.42
            + freshness_score * 0.28
            + relevance_score * 0.25
            + url_score * 0.05
        )
        if quality_score < MIN_ARTICLE_QUALITY_SCORE:
            return None

        return {
            "title": clean_title,
            "summary": clean_summary,
            "url": clean_url,
            "source": source,
            "published_date": self._format_published_datetime(parsed_dt, published),
            "published_ts": parsed_dt.timestamp() if parsed_dt else 0.0,
            "weight": feed_config.get("weight", 0.5),
            "source_score": round(source_score, 3),
            "freshness_score": round(freshness_score, 3),
            "relevance_score": round(relevance_score, 3),
            "quality_score": round(quality_score, 3),
            "url_domain": self._url_domain(clean_url),
        }

    def _parse_published_datetime(self, value: str) -> Optional[datetime]:
        """Parse RSS/Atom date strings into UTC datetimes."""
        if not value:
            return None

        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _format_published_datetime(
        self, value: Optional[datetime], fallback: str
    ) -> str:
        """Return normalized ISO datetime when possible."""
        if value:
            return value.isoformat()
        return fallback

    def _article_age_hours(self, published: datetime) -> float:
        """Return article age in hours."""
        now = datetime.now(timezone.utc)
        return max(0.0, (now - published).total_seconds() / 3600)

    def _freshness_score(self, published: Optional[datetime]) -> float:
        """Score recent articles higher without dropping undated feeds entirely."""
        if not published:
            return 0.45
        age_hours = self._article_age_hours(published)
        if age_hours <= 3:
            return 1.0
        if age_hours <= 12:
            return 0.82
        if age_hours <= 24:
            return 0.64
        if age_hours <= MAX_ARTICLE_AGE_HOURS:
            return 0.42
        return 0.0

    def _source_quality_score(self, source: str, feed_config: Dict[str, Any]) -> float:
        """Return source reliability score, falling back to feed weight."""
        return float(SOURCE_QUALITY_SCORES.get(source, feed_config.get("weight", 0.5)))

    def _relevance_score(self, text: str) -> float:
        """Score crypto and market-moving relevance from title/summary text."""
        text_lower = text.lower()
        keyword_hits = sum(1 for keyword in CRYPTO_KEYWORDS if keyword in text_lower)
        base_score = min(0.65, keyword_hits * 0.08)
        event_score = sum(
            weight
            for keyword, weight in MARKET_MOVING_KEYWORDS.items()
            if keyword in text_lower
        )
        ticker_bonus = 0.12 if re.search(r"\(([A-Z0-9]{2,10})\)", text) else 0.0
        return float(min(1.0, base_score + event_score + ticker_bonus))

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Unescape HTML entities
        text = unescape(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text.strip()

    def _canonicalize_url(self, url: str) -> str:
        """Normalize URLs for dedupe while preserving the article identity."""
        if not url:
            return ""

        parsed = urlparse(unescape(url).strip())
        if not parsed.scheme or not parsed.netloc:
            return url.strip()

        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
            and key.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}
        ]
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                "",
                urlencode(query),
                "",
            )
        )

    def _url_domain(self, url: str) -> str:
        """Return URL host without leading www."""
        parsed = urlparse(url)
        return parsed.netloc.lower().removeprefix("www.")

    def _deduplicate_articles(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on canonical URL and title."""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []

        for article in articles:
            url = article.get("url", "")
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

            # Create normalized title for comparison
            normalized_title = re.sub(r"[^a-z0-9]", "", article["title"].lower())

            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_articles.append(article)

        return unique_articles

    def _rank_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank articles by quality, then recency."""
        return sorted(
            articles,
            key=lambda article: (
                article.get("quality_score", 0),
                article.get("published_ts", 0),
                article.get("source_score", 0),
            ),
            reverse=True,
        )


# Create global instance
enhanced_rss_aggregator = EnhancedRSSAggregator()
