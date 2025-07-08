"""Enhanced RSS news scraper with robust error handling and fallback mechanisms.

This module provides a more resilient RSS feed scraper that can handle:
- Malformed XML
- Encoding issues
- Partial feed failures
- Various RSS/Atom formats
"""

import logging
import re
import time
import concurrent.futures
import warnings
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from html import unescape

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import chardet

# Suppress chardet debug messages
# chardet produces verbose debug logs during encoding detection that appear as errors
# but are actually just part of its normal detection process. We only want warnings/errors.
logging.getLogger('chardet').setLevel(logging.WARNING)
logging.getLogger('chardet.charsetprober').setLevel(logging.WARNING)
logging.getLogger('chardet.universaldetector').setLevel(logging.WARNING)

import feedparser

# HTTP Configuration
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.3

# Parallel processing
MAX_WORKERS = 6

# Article limits
MAX_ARTICLES_PER_FEED = 10
DEFAULT_MAX_ARTICLES = 20

# Time constraints
RECENT_HOURS = 24  # More lenient time window

# Cryptocurrency keywords
CRYPTO_KEYWORDS = [
    'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
    'blockchain', 'defi', 'nft', 'web3', 'altcoin', 'trading',
    'binance', 'coinbase', 'upbit', 'exchange', 'wallet',
    'mining', 'staking', 'yield', 'dapp', 'smart contract',
    'ripple', 'xrp', 'cardano', 'ada', 'solana', 'sol',
    'polkadot', 'dot', 'chainlink', 'link', 'dogecoin', 'doge',
    'market', 'price', 'bull', 'bear', 'sec', 'regulation'
]

logger = logging.getLogger(__name__)


class EnhancedRSSAggregator:
    """Enhanced RSS aggregator with robust error handling."""
    
    # USED
    def __init__(self):
        """Initialize enhanced RSS aggregator."""
        self.rss_feeds = self._initialize_feeds()
        self.session = self._create_resilient_session()
        self._user_agent_index = 0
    
    # USED
    def _initialize_feeds(self) -> Dict[str, Dict[str, Any]]:
        """Initialize RSS feeds with enhanced configurations."""
        return {
            'cointelegraph': {
                'url': 'https://cointelegraph.com/rss',
                'name': 'CoinTelegraph',
                'weight': 1.0,
                'encoding_hint': 'utf-8',
                'parser_hints': {'lenient': True}
            },
            'coindesk': {
                'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'name': 'CoinDesk',
                'weight': 0.95,
                'encoding_hint': 'utf-8',
                'parser_hints': {'lenient': True, 'sanitize': True},
                'fallback_url': 'https://www.coindesk.com/feed/'
            },
            'decrypt': {
                'url': 'https://decrypt.co/feed',
                'name': 'Decrypt',
                'weight': 0.9,
                'encoding_hint': 'utf-8',
                'parser_hints': {'lenient': True}
            },
            'theblock': {
                'url': 'https://www.theblock.co/rss.xml',
                'name': 'The Block',
                'weight': 0.9,
                'encoding_hint': 'utf-8'
            },
            'bitcoinist': {
                'url': 'https://bitcoinist.com/feed/',
                'name': 'Bitcoinist',
                'weight': 0.85,
                'encoding_hint': 'utf-8'
            },
            'cryptoslate': {
                'url': 'https://cryptoslate.com/feed/',
                'name': 'CryptoSlate',
                'weight': 0.85,
                'encoding_hint': 'utf-8'
            },
            'dailyhodl': {
                'url': 'https://dailyhodl.com/feed/',
                'name': 'The Daily Hodl',
                'weight': 0.8,
                'encoding_hint': 'utf-8'
            },
            'cryptopotato': {
                'url': 'https://cryptopotato.com/feed/',
                'name': 'CryptoPotato',
                'weight': 0.75,
                'encoding_hint': 'utf-8',
                'parser_hints': {'lenient': True}
            },
            'beincrypto': {
                'url': 'https://beincrypto.com/feed/',
                'name': 'BeInCrypto',
                'weight': 0.75,
                'encoding_hint': 'utf-8'
            },
            'coinjournal': {
                'url': 'https://coinjournal.net/news/feed/',  # Updated URL
                'name': 'Coin Journal',
                'weight': 0.7,
                'encoding_hint': 'utf-8'
            },
            'newsbtc': {
                'url': 'https://www.newsbtc.com/feed/',
                'name': 'NewsBTC',
                'weight': 0.7,
                'encoding_hint': 'utf-8'
            },
            'blockworks': {
                'url': 'https://blockworks.co/feed',
                'name': 'Blockworks',
                'weight': 0.85,
                'encoding_hint': 'utf-8',  # Force UTF-8
                'parser_hints': {'sanitize': True, 'encoding_override': 'utf-8'}
            },
            'cryptonews': {
                'url': 'https://cryptonews.com/news/feed/',
                'name': 'CryptoNews',
                'weight': 0.8,
                'encoding_hint': 'utf-8'
            },
            'ambcrypto': {
                'url': 'https://ambcrypto.com/feed/',
                'name': 'AMBCrypto',
                'weight': 0.7,
                'encoding_hint': 'utf-8'
            },
            'u_today': {
                'url': 'https://u.today/rss',
                'name': 'U.Today',
                'weight': 0.75,
                'encoding_hint': 'utf-8'
            },
            'cryptonewsz': {
                'url': 'https://www.cryptonewsz.com/feed/',
                'name': 'CryptoNewsZ',
                'weight': 0.65,
                'encoding_hint': 'utf-8',
                'parser_hints': {'lenient': True}
            }
        }
    
    # USED
    def _create_resilient_session(self) -> requests.Session:
        """Create a resilient HTTP session with retry logic."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    # USED
    def _get_next_user_agent(self) -> str:
        """Rotate user agents to avoid blocking."""
        agent = USER_AGENTS[self._user_agent_index]
        self._user_agent_index = (self._user_agent_index + 1) % len(USER_AGENTS)
        return agent
    
    # USED
    def fetch_all_news(self, max_articles: int = DEFAULT_MAX_ARTICLES) -> List[Dict[str, Any]]:
        """Fetch news from all RSS feeds with enhanced error handling."""
        logger.info("🚀 Starting enhanced RSS news collection...")
        all_articles = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_feed = {}
            
            # Submit all feed fetch tasks
            for feed_key, feed_config in self.rss_feeds.items():
                future = executor.submit(self._try_fetch_feed, feed_config['url'], feed_config)
                future_to_feed[future] = feed_key
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_feed):
                feed_key = future_to_feed[future]
                feed_name = self.rss_feeds[feed_key]['name']
                
                try:
                    articles = future.result()
                    if articles:
                        logger.info(f"✅ {feed_name}: {len(articles)} articles collected")
                        all_articles.extend(articles)
                    else:
                        logger.warning(f"⚠️ {feed_name}: No articles collected")
                except Exception as e:
                    logger.error(f"❌ {feed_name}: Collection failed - {e}")
        
        # Process and deduplicate
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Sort by date (newest first)
        sorted_articles = sorted(
            unique_articles,
            key=lambda x: x.get('published_date', ''),
            reverse=True
        )
        
        final_articles = sorted_articles[:max_articles]
        logger.info(f"✅ Total articles collected: {len(final_articles)}")
        
        return final_articles
    
    # USED
    def _try_fetch_feed(self, url: str, feed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Try to fetch and parse a feed URL."""
        try:
            # Fetch raw content
            headers = {
                'User-Agent': self._get_next_user_agent(),
                'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache'
            }
            
            response = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Detect and fix encoding
            raw_content = response.content
            encoding_hint = feed_config.get('encoding_hint', 'utf-8')
            
            # Try to detect encoding
            detected = chardet.detect(raw_content)
            encoding = detected['encoding'] if detected['confidence'] > 0.7 else encoding_hint
            
            content = raw_content.decode(encoding)

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
    
    # USED
    def _sanitize_xml(self, text: str) -> str:
        """Sanitize XML content to fix common issues."""
        # Remove BOM
        text = text.lstrip('\ufeff')
        
        # Fix CDATA sections
        text = re.sub(r'<!\[CDATA\[(.*?)\]\]>', lambda m: unescape(m.group(1)), text, flags=re.DOTALL)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Fix mismatched tags (basic)
        text = re.sub(r'<([^/>]+)>([^<]*)</\1(?:[^>]*)>', r'<\1>\2</\1>', text)
        
        return text
    
    # USED
    def _parse_with_feedparser(self, content: str, feed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse feed with feedparser library."""
        # Parse with feedparser
        parser_hints = feed_config.get('parser_hints', {})
        
        if parser_hints.get('sanitize'):
            content = self._sanitize_xml(content)
        
        # Suppress feedparser warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feed = feedparser.parse(content)
        
        # Always try to extract articles even if bozo
        if hasattr(feed, 'entries') and feed.entries:
            articles = []
            for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
                article = self._extract_article_from_entry(entry, feed_config)
                if article:
                    articles.append(article)
            
            if articles:
                return articles
        
        # No articles found
        return []
    
    # USED
    def _parse_with_xml(self, content: str, feed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse feed with XML ElementTree as fallback."""
        try:
            # Try to parse as XML
            root = ET.fromstring(content)
            
            # Find all item elements (RSS)
            items = root.findall('.//item')
            
            articles = []
            for item in items[:MAX_ARTICLES_PER_FEED]:
                article = self._extract_article_from_xml(item, feed_config)
                if article:
                    articles.append(article)
            
            return articles
            
        except ET.ParseError as e:
            logger.debug(f"XML parsing failed for {feed_config['name']}, trying regex extraction...")
            return []
    
    # USED
    def _parse_with_regex(self, content: str, feed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Last resort: extract articles with regex."""
        try:
            articles = []
            
            # Find all item blocks
            item_pattern = r'<item>(.*?)</item>'
            items = re.findall(item_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for item_content in items[:MAX_ARTICLES_PER_FEED]:
                article = self._extract_article_from_regex(item_content, feed_config)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.debug(f"Regex extraction also failed for {feed_config['name']}: {e}")
            return []
    
    # USED
    def _extract_article_from_entry(self, entry: Any, feed_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract article from feedparser entry."""
        try:
            title = entry.get('title', '').strip()
            link = entry.get('link', '').strip()
            summary = entry.get('summary', '').strip()
            
            # Get publish date
            published = ''
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6]).isoformat()
            elif hasattr(entry, 'published'):
                published = entry.published
            
            # Check if crypto-related
            if not self._is_crypto_related(title + ' ' + summary):
                return None
            
            return {
                'title': self._clean_text(title),
                'summary': self._clean_text(summary),
                'url': link,
                'source': feed_config['name'],
                'published_date': published,
                'weight': feed_config.get('weight', 0.5)
            }
            
        except Exception as e:
            logger.debug(f"Error extracting article: {e}")
            return None
    
    # USED
    def _extract_article_from_xml(self, item: ET.Element, feed_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract article from XML element."""
        try:
            # Extract RSS fields
            title = self._get_xml_text(item, ['title'])
            link = self._get_xml_text(item, ['link', 'guid'])
            summary = self._get_xml_text(item, ['description', 'summary'])
            published = self._get_xml_text(item, ['pubDate', 'published'])
            
            # Check if crypto-related
            if not self._is_crypto_related(title + ' ' + summary):
                return None
            
            return {
                'title': self._clean_text(title),
                'summary': self._clean_text(summary),
                'url': link,
                'source': feed_config['name'],
                'published_date': published,
                'weight': feed_config.get('weight', 0.5)
            }
            
        except Exception as e:
            logger.debug(f"Error extracting from XML: {e}")
            return None
    
    # USED
    def _extract_article_from_regex(self, item_content: str, feed_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract article using regex patterns."""
        try:
            # Extract fields with regex
            title = self._extract_with_regex(item_content, r'<title>(.*?)</title>')
            link = self._extract_with_regex(item_content, r'<link>(.*?)</link>')
            summary = self._extract_with_regex(item_content, r'<description>(.*?)</description>')
            published = self._extract_with_regex(item_content, r'<pubDate>(.*?)</pubDate>')
            
            # Check if crypto-related
            if not self._is_crypto_related(title + ' ' + summary):
                return None
            
            return {
                'title': self._clean_text(title),
                'summary': self._clean_text(summary),
                'url': link,
                'source': feed_config['name'],
                'published_date': published,
                'weight': feed_config.get('weight', 0.5)
            }
            
        except Exception as e:
            logger.debug(f"Error extracting with regex: {e}")
            return None
    
    # USED
    def _get_xml_text(self, element: ET.Element, tags: List[str]) -> str:
        """Get text from XML element trying multiple tag names."""
        for tag in tags:
            elem = element.find(tag)
            if elem is not None:
                if tag.endswith('link') and elem.get('href'):
                    return elem.get('href', '')
                return elem.text or ''
        return ''
    
    # USED
    def _extract_with_regex(self, text: str, pattern: str) -> str:
        """Extract text using regex pattern."""
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ''
    
    # USED
    def _is_crypto_related(self, text: str) -> bool:
        """Check if text contains cryptocurrency keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in CRYPTO_KEYWORDS)
    
    # USED
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Unescape HTML entities
        text = unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    # USED
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # Create normalized title for comparison
            normalized_title = re.sub(r'[^a-z0-9]', '', article['title'].lower())
            
            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles

# Create global instance
enhanced_rss_aggregator = EnhancedRSSAggregator()

