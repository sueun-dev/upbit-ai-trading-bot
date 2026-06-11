"""Pattern Learner for identifying and learning from trading patterns.

This module analyzes historical trading data to identify successful and
unsuccessful patterns, enabling the system to learn and improve over time.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.infrastructure.database import get_db_path
from src.shared.openai_client import OpenAIClient

# Pattern thresholds
MIN_OCCURRENCES_FOR_LESSON = 10
HIGH_SUCCESS_THRESHOLD = 0.7
LOW_SUCCESS_THRESHOLD = 0.3

# Confidence thresholds
MIN_LESSON_CONFIDENCE = 0.7
MIN_LESSON_APPLICATIONS = 5
MIN_LESSON_SUCCESS_RATE = 0.6
NEUTRAL_SUCCESS_RATE = 0.5

# Market categorization thresholds
TREND_STRONG_UP_THRESHOLD = 5.0
TREND_UP_THRESHOLD = 2.0
TREND_STRONG_DOWN_THRESHOLD = -5.0
TREND_DOWN_THRESHOLD = -2.0

VOLUME_VERY_HIGH_THRESHOLD = 2.0
VOLUME_HIGH_THRESHOLD = 1.5
VOLUME_LOW_THRESHOLD = 0.5

RSI_OVERBOUGHT_THRESHOLD = 70
RSI_OVERSOLD_THRESHOLD = 30

VOLATILITY_HIGH_THRESHOLD = 0.15
VOLATILITY_LOW_THRESHOLD = 0.05

# AI generation settings
LESSON_GENERATION_TEMPERATURE = 0.3

# Database limits
MAX_LESSONS_FOR_PROMPT = 10

logger = logging.getLogger(__name__)


class PatternLearner:
    """Pattern learning system for trading strategy improvement.

    Features:
    - Success/failure pattern identification
    - Market condition correlation
    - Trading lesson generation
    - Pattern-based recommendations
    - Performance tracking
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the pattern learner.

        Args:
            api_key: Backward-compatible placeholder; local OAuth bridge is used.
        """
        self.openai_client = OpenAIClient(api_key=api_key)
        self.db_path = get_db_path("pattern_learning.db")

        # Initialize pattern learning database
        with sqlite3.connect(self.db_path) as conn:
            self._create_patterns_table(conn)
            self._create_lessons_table(conn)
            self._create_outcomes_table(conn)

    def _create_patterns_table(self, conn: sqlite3.Connection) -> None:
        """Create learned patterns table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_rate REAL NOT NULL,
                occurrence_count INTEGER NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_lessons_table(self, conn: sqlite3.Connection) -> None:
        """Create trading lessons table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson_type TEXT NOT NULL,
                lesson_content TEXT NOT NULL,
                confidence REAL NOT NULL,
                application_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _create_outcomes_table(self, conn: sqlite3.Connection) -> None:
        """Create pattern outcomes table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pattern_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                profit_loss REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (pattern_id) REFERENCES learned_patterns (id)
            )
        """)

    def analyze_trade_patterns(
        self,
        symbol: str,
        action: str,
        market_conditions: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        """Analyze and record trading patterns.

        Args:
            symbol: Trading symbol.
            action: Trading action taken.
            market_conditions: Market conditions at time of trade.
            outcome: Trade outcome (profit/loss, etc.).
        """
        try:
            # Extract pattern features
            pattern = self._extract_pattern_features(symbol, action, market_conditions)

            # Record pattern outcome
            self._record_pattern_outcome(pattern, outcome)

            # Update pattern statistics
            self._update_pattern_statistics(pattern)

            # Generate new lessons if threshold met
            if self._should_generate_lesson(pattern):
                self._generate_trading_lesson(pattern)

        except Exception as e:
            logger.error(f"Failed to analyze trade pattern: {e}")

    def get_trading_lessons_for_prompt(self) -> List[str]:
        """Get relevant trading lessons for AI prompts.

        Returns:
            List of trading lessons to include in prompts.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                lessons = self._fetch_high_confidence_lessons(conn)
                return self._filter_successful_lessons(lessons)

        except Exception as e:
            logger.error(f"Failed to get trading lessons: {e}")
            return []

    def _extract_pattern_features(
        self, symbol: str, action: str, market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relevant features from trading conditions.

        Args:
            symbol: Trading symbol.
            action: Trading action.
            market_conditions: Market conditions.

        Returns:
            Extracted pattern features.
        """
        features = {
            "symbol": symbol,
            "action": action,
            "price_trend": self._categorize_trend(
                market_conditions.get("price_24h_change", 0)
            ),
            "volume_level": self._categorize_volume(
                market_conditions.get("volume_ratio", 1)
            ),
            "rsi_level": self._categorize_rsi(
                market_conditions.get("rsi", {}).get("rsi_14", 50)
            ),
            "volatility": self._categorize_volatility(
                market_conditions.get("volatility", 0.1)
            ),
            "market_regime": market_conditions.get("regime", "neutral"),
        }

        # Create pattern key
        pattern_key = self._create_pattern_key(
            action, features["price_trend"], features["rsi_level"]
        )
        features["pattern_key"] = pattern_key

        return features

    def _categorize_trend(self, price_change: float) -> str:
        """Categorize price trend.

        Args:
            price_change: 24h price change percentage.

        Returns:
            Trend category.
        """
        if price_change > TREND_STRONG_UP_THRESHOLD:
            return "strong_up"
        elif price_change > TREND_UP_THRESHOLD:
            return "up"
        elif price_change < TREND_STRONG_DOWN_THRESHOLD:
            return "strong_down"
        elif price_change < TREND_DOWN_THRESHOLD:
            return "down"
        else:
            return "sideways"

    def _categorize_volume(self, volume_ratio: float) -> str:
        """Categorize volume level.

        Args:
            volume_ratio: Volume ratio compared to average.

        Returns:
            Volume category.
        """
        if volume_ratio > VOLUME_VERY_HIGH_THRESHOLD:
            return "very_high"
        elif volume_ratio > VOLUME_HIGH_THRESHOLD:
            return "high"
        elif volume_ratio < VOLUME_LOW_THRESHOLD:
            return "low"
        else:
            return "normal"

    def _categorize_rsi(self, rsi: float) -> str:
        """Categorize RSI level.

        Args:
            rsi: RSI value.

        Returns:
            RSI category.
        """
        if rsi > RSI_OVERBOUGHT_THRESHOLD:
            return "overbought"
        elif rsi < RSI_OVERSOLD_THRESHOLD:
            return "oversold"
        else:
            return "neutral"

    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility level.

        Args:
            volatility: Volatility value.

        Returns:
            Volatility category.
        """
        if volatility > VOLATILITY_HIGH_THRESHOLD:
            return "high"
        elif volatility < VOLATILITY_LOW_THRESHOLD:
            return "low"
        else:
            return "medium"

    def _create_pattern_key(self, action: str, trend: str, rsi: str) -> str:
        """Create unique pattern key.

        Args:
            action: Trading action.
            trend: Price trend.
            rsi: RSI level.

        Returns:
            Pattern key string.
        """
        return f"{action}_{trend}_{rsi}"

    def _record_pattern_outcome(
        self, pattern: Dict[str, Any], outcome: Dict[str, Any]
    ) -> None:
        """Record pattern outcome in database.

        Args:
            pattern: Pattern features.
            outcome: Trade outcome.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                pattern_id = self._get_or_create_pattern(conn, pattern)
                self._insert_pattern_outcome(conn, pattern_id, outcome)

        except Exception as e:
            logger.error(f"Failed to record pattern outcome: {e}")

    def _get_or_create_pattern(
        self, conn: sqlite3.Connection, pattern: Dict[str, Any]
    ) -> int:
        """Get existing pattern ID or create new pattern.

        Args:
            conn: Database connection.
            pattern: Pattern features.

        Returns:
            Pattern ID.
        """
        cursor = conn.execute(
            """
            SELECT id FROM learned_patterns
            WHERE pattern_type = ?
            LIMIT 1
        """,
            (pattern["pattern_key"],),
        )

        row = cursor.fetchone()
        if row:
            return int(row[0])

        # Create new pattern
        cursor = conn.execute(
            """
            INSERT INTO learned_patterns
            (pattern_type, pattern_data, success_rate, occurrence_count, last_seen)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                pattern["pattern_key"],
                json.dumps(pattern),
                NEUTRAL_SUCCESS_RATE,
                0,
                datetime.now(),
            ),
        )
        return cursor.lastrowid or -1

    def _insert_pattern_outcome(
        self, conn: sqlite3.Connection, pattern_id: int, outcome: Dict[str, Any]
    ) -> None:
        """Insert pattern outcome record.

        Args:
            conn: Database connection.
            pattern_id: Pattern ID.
            outcome: Trade outcome.
        """
        outcome_type = "success" if outcome.get("profit_loss", 0) > 0 else "failure"

        conn.execute(
            """
            INSERT INTO pattern_outcomes
            (pattern_id, trade_id, outcome, profit_loss, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                pattern_id,
                outcome.get("trade_id", ""),
                outcome_type,
                outcome.get("profit_loss", 0),
                datetime.now(),
            ),
        )

    def _update_pattern_statistics(self, pattern: Dict[str, Any]) -> None:
        """Update pattern statistics based on outcomes.

        Args:
            pattern: Pattern features.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = self._calculate_pattern_statistics(conn, pattern)
                if stats:
                    self._update_pattern_record(conn, stats)

        except Exception as e:
            logger.error(f"Failed to update pattern statistics: {e}")

    def _calculate_pattern_statistics(
        self, conn: sqlite3.Connection, pattern: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Calculate pattern statistics from outcomes.

        Args:
            conn: Database connection.
            pattern: Pattern features.

        Returns:
            Statistics dictionary or None.
        """
        cursor = conn.execute(
            """
            SELECT
                p.id,
                COUNT(po.id) as total_outcomes,
                SUM(CASE WHEN po.outcome = 'success' THEN 1 ELSE 0 END) as successes
            FROM learned_patterns p
            JOIN pattern_outcomes po ON p.id = po.pattern_id
            WHERE p.pattern_type = ?
            GROUP BY p.id
        """,
            (pattern["pattern_key"],),
        )

        row = cursor.fetchone()
        if row:
            pattern_id, total, successes = row
            success_rate = successes / total if total > 0 else NEUTRAL_SUCCESS_RATE

            return {
                "pattern_id": pattern_id,
                "success_rate": success_rate,
                "total": total,
            }
        return None

    def _update_pattern_record(
        self, conn: sqlite3.Connection, stats: Dict[str, Any]
    ) -> None:
        """Update pattern record with new statistics.

        Args:
            conn: Database connection.
            stats: Pattern statistics.
        """
        conn.execute(
            """
            UPDATE learned_patterns
            SET success_rate = ?, occurrence_count = ?, last_seen = ?
            WHERE id = ?
        """,
            (
                stats["success_rate"],
                stats["total"],
                datetime.now(),
                stats["pattern_id"],
            ),
        )

    def _should_generate_lesson(self, pattern: Dict[str, Any]) -> bool:
        """Determine if pattern warrants generating a lesson.

        Args:
            pattern: Pattern features.

        Returns:
            True if lesson should be generated.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT occurrence_count, success_rate
                    FROM learned_patterns
                    WHERE pattern_type = ?
                """,
                    (pattern["pattern_key"],),
                )

                row = cursor.fetchone()
                if row:
                    occurrences, success_rate = row
                    return self._meets_lesson_criteria(occurrences, success_rate)

                return False

        except Exception as e:
            logger.error(f"Failed to check lesson generation: {e}")
            return False

    def _meets_lesson_criteria(self, occurrences: int, success_rate: float) -> bool:
        """Check if pattern meets criteria for lesson generation.

        Args:
            occurrences: Number of pattern occurrences.
            success_rate: Pattern success rate.

        Returns:
            True if criteria are met.
        """
        return occurrences >= MIN_OCCURRENCES_FOR_LESSON and (
            success_rate > HIGH_SUCCESS_THRESHOLD
            or success_rate < LOW_SUCCESS_THRESHOLD
        )

    def _generate_trading_lesson(self, pattern: Dict[str, Any]) -> None:
        """Generate trading lesson from pattern using AI.

        Args:
            pattern: Pattern features.
        """
        try:
            pattern_stats = self._get_pattern_stats_for_lesson(pattern)
            if not pattern_stats:
                return

            lesson_content = self._generate_lesson_with_ai(pattern, pattern_stats)

            if lesson_content:
                self._store_trading_lesson(lesson_content, pattern_stats)

        except Exception as e:
            logger.error(f"Failed to generate trading lesson: {e}")

    def _get_pattern_stats_for_lesson(
        self, pattern: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get pattern statistics for lesson generation.

        Args:
            pattern: Pattern features.

        Returns:
            Pattern statistics or None.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT pattern_data, success_rate, occurrence_count
                FROM learned_patterns
                WHERE pattern_type = ?
            """,
                (pattern["pattern_key"],),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "pattern_data": json.loads(row[0]),
                    "success_rate": row[1],
                    "occurrences": row[2],
                }
        return None

    def _generate_lesson_with_ai(
        self, pattern: Dict[str, Any], pattern_stats: Dict[str, Any]
    ) -> str:
        """Generate lesson content using AI.

        Args:
            pattern: Pattern features.
            pattern_stats: Pattern statistics.

        Returns:
            Generated lesson content or empty string.
        """
        system_message = self._get_lesson_generation_system_message()
        prompt = self._create_lesson_generation_prompt(pattern, pattern_stats)

        result = self.openai_client.analyze_with_prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=LESSON_GENERATION_TEMPERATURE,
        )

        return str(result.get("response", ""))

    def _get_lesson_generation_system_message(self) -> str:
        """Get system message for lesson generation.

        Returns:
            System message string.
        """
        return """You are a trading strategy analyst. Generate a concise,
        actionable trading lesson from pattern analysis. Focus on when to apply
        or avoid certain actions based on market conditions."""

    def _create_lesson_generation_prompt(
        self, pattern: Dict[str, Any], pattern_stats: Dict[str, Any]
    ) -> str:
        """Create prompt for lesson generation.

        Args:
            pattern: Pattern features.
            pattern_stats: Pattern statistics.

        Returns:
            Prompt string.
        """
        pattern_data = pattern_stats["pattern_data"]
        success_rate = pattern_stats["success_rate"]
        occurrences = pattern_stats["occurrences"]

        focus = "what works" if success_rate > NEUTRAL_SUCCESS_RATE else "what to avoid"

        return f"""Analyze this trading pattern and generate a lesson:

Pattern: {pattern['pattern_key']}
Success Rate: {success_rate:.1%}
Occurrences: {occurrences}

Pattern Details:
- Action: {pattern_data.get('action')}
- Price Trend: {pattern_data.get('price_trend')}
- RSI Level: {pattern_data.get('rsi_level')}
- Volume: {pattern_data.get('volume_level')}
- Volatility: {pattern_data.get('volatility')}

Generate a one-sentence trading lesson that can be applied in future decisions.
Focus on {focus}.
"""

    def _store_trading_lesson(
        self, lesson_content: str, pattern_stats: Dict[str, Any]
    ) -> None:
        """Store generated trading lesson.

        Args:
            lesson_content: Lesson text.
            pattern_stats: Pattern statistics.
        """
        success_rate = pattern_stats["success_rate"]
        lesson_type = "success" if success_rate > NEUTRAL_SUCCESS_RATE else "failure"
        confidence = abs(success_rate - NEUTRAL_SUCCESS_RATE) * 2

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trading_lessons
                (lesson_type, lesson_content, confidence)
                VALUES (?, ?, ?)
            """,
                (lesson_type, lesson_content, confidence),
            )

        logger.info(f"Generated new trading lesson: {lesson_content}")

    def _fetch_high_confidence_lessons(self, conn: sqlite3.Connection) -> List[tuple]:
        """Fetch high-confidence lessons from database.

        Args:
            conn: Database connection.

        Returns:
            List of lesson tuples.
        """
        cursor = conn.execute(
            """
            SELECT lesson_content, confidence, success_count, application_count
            FROM trading_lessons
            WHERE confidence > ?
            AND application_count > ?
            ORDER BY (success_count * 1.0 / application_count) DESC
            LIMIT ?
        """,
            (MIN_LESSON_CONFIDENCE, MIN_LESSON_APPLICATIONS, MAX_LESSONS_FOR_PROMPT),
        )

        return cursor.fetchall()

    def _filter_successful_lessons(self, lessons: List[tuple]) -> List[str]:
        """Filter lessons based on success rate.

        Args:
            lessons: List of lesson tuples.

        Returns:
            List of successful lesson contents.
        """
        successful_lessons = []

        for row in lessons:
            content, _, success_count, application_count = row
            success_rate = (
                success_count / application_count if application_count > 0 else 0
            )

            if success_rate > MIN_LESSON_SUCCESS_RATE:
                successful_lessons.append(content)

        return successful_lessons


# Note: pattern_learner instance will be created by TradingOrchestrator with API key
