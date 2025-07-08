"""Pattern Learner for identifying and learning from trading patterns.

This module analyzes historical trading data to identify successful and
unsuccessful patterns, enabling the system to learn and improve over time.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.infrastructure.database import get_db_path
from src.shared.openai_client import OpenAIClient

# Pattern thresholds
MIN_OCCURRENCES_FOR_LESSON = 10
HIGH_SUCCESS_THRESHOLD = 0.7
LOW_SUCCESS_THRESHOLD = 0.3
MIN_OCCURRENCES_FOR_MATCHING = 5
FAILURE_PATTERN_THRESHOLD = 0.4
HIGH_RISK_SUCCESS_THRESHOLD = 0.2
HIGH_RISK_MIN_OCCURRENCES = 10
COMMON_FACTOR_THRESHOLD = 0.5
HIGH_FAILURE_RATE_THRESHOLD = 0.6

# Confidence thresholds
MIN_LESSON_CONFIDENCE = 0.7
MIN_LESSON_APPLICATIONS = 5
MIN_LESSON_SUCCESS_RATE = 0.6
NEUTRAL_SUCCESS_RATE = 0.5
DEFAULT_CONFIDENCE = 0.0

# Success rate thresholds
SUCCESS_ACTION_THRESHOLD = 0.6
FAILURE_ACTION_THRESHOLD = 0.4

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
MAX_FAILURE_PATTERNS = 10
MAX_LESSONS_FOR_PROMPT = 10
MAX_RECOMMENDATIONS = 5
MAX_ACTION_RECOMMENDATIONS = 3
MAX_HIGH_RISK_CONDITIONS = 3

# Default values
DEFAULT_DAYS_BACK = 7

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
    
    def __init__(self) -> None:
        """Initialize the pattern learner."""
        self.openai_client = None  # Will be set by initialize()
        self.db_path = get_db_path('pattern_learning.db')
        self._init_database()
        self.pattern_cache = {}
        logger.info("Pattern Learner initialized")
    
    # USED
    def initialize(self, api_key: str) -> None:
        """Initialize with API key.
        
        Args:
            api_key: OpenAI API key
        """
        self.openai_client = OpenAIClient(api_key=api_key)
    
    def _init_database(self) -> None:
        """Initialize pattern learning database."""
        with sqlite3.connect(self.db_path) as conn:
            self._create_patterns_table(conn)
            self._create_lessons_table(conn)
            self._create_outcomes_table(conn)
    
    def _create_patterns_table(self, conn: sqlite3.Connection) -> None:
        """Create learned patterns table."""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                success_rate REAL NOT NULL,
                occurrence_count INTEGER NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    def _create_lessons_table(self, conn: sqlite3.Connection) -> None:
        """Create trading lessons table."""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trading_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson_type TEXT NOT NULL,
                lesson_content TEXT NOT NULL,
                confidence REAL NOT NULL,
                application_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    def _create_outcomes_table(self, conn: sqlite3.Connection) -> None:
        """Create pattern outcomes table."""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pattern_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                profit_loss REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (pattern_id) REFERENCES learned_patterns (id)
            )
        ''')
    
    def analyze_trade_patterns(
        self,
        symbol: str,
        action: str,
        market_conditions: Dict[str, Any],
        outcome: Dict[str, Any]
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
            pattern = self._extract_pattern_features(
                symbol, action, market_conditions
            )
            
            # Record pattern outcome
            self._record_pattern_outcome(pattern, outcome)
            
            # Update pattern statistics
            self._update_pattern_statistics(pattern)
            
            # Generate new lessons if threshold met
            if self._should_generate_lesson(pattern):
                self._generate_trading_lesson(pattern)
                
        except Exception as e:
            logger.error(f"Failed to analyze trade pattern: {e}")
    
    def get_pattern_recommendations(
        self,
        symbol: str,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get pattern-based recommendations for current conditions.
        
        Args:
            symbol: Trading symbol.
            market_conditions: Current market conditions.
            
        Returns:
            Pattern-based recommendations.
        """
        try:
            # Find matching patterns
            matching_patterns = self._find_matching_patterns(
                symbol, market_conditions
            )
            
            if not matching_patterns:
                return self._get_empty_recommendations()
            
            # Aggregate recommendations
            recommendations = self._aggregate_pattern_recommendations(
                matching_patterns
            )
            
            return {
                'has_recommendations': True,
                'confidence': recommendations['confidence'],
                'recommendations': recommendations['actions'],
                'supporting_patterns': len(matching_patterns)
            }
            
        except Exception as e:
            logger.error(f"Failed to get pattern recommendations: {e}")
            return self._get_empty_recommendations()
    
    def analyze_failure_patterns(
        self,
        days_back: int = DEFAULT_DAYS_BACK
    ) -> Dict[str, Any]:
        """Analyze recent failure patterns to identify issues.
        
        Args:
            days_back: Number of days to analyze.
            
        Returns:
            Failure pattern analysis.
        """
        try:
            failure_patterns = self._get_recent_failure_patterns(days_back)
            
            # Analyze common failure factors
            failure_analysis = self._analyze_failure_factors(failure_patterns)
            
            # Generate recommendations
            recommendations = self._generate_failure_recommendations(
                failure_analysis
            )
            
            return {
                'failure_rate': failure_analysis.get('overall_failure_rate', 0),
                'common_factors': failure_analysis.get('common_factors', []),
                'high_risk_conditions': failure_analysis.get('high_risk_conditions', []),
                'recommendations': recommendations,
                'pattern_count': len(failure_patterns)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze failure patterns: {e}")
            return self._get_empty_failure_analysis()
    
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
        self,
        symbol: str,
        action: str,
        market_conditions: Dict[str, Any]
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
            'symbol': symbol,
            'action': action,
            'price_trend': self._categorize_trend(
                market_conditions.get('price_change_24h', 0)
            ),
            'volume_level': self._categorize_volume(
                market_conditions.get('volume_ratio', 1)
            ),
            'rsi_level': self._categorize_rsi(
                market_conditions.get('rsi', {}).get('rsi_14', 50)
            ),
            'volatility': self._categorize_volatility(
                market_conditions.get('volatility', 0.1)
            ),
            'market_regime': market_conditions.get('regime', 'neutral')
        }
        
        # Create pattern key
        pattern_key = self._create_pattern_key(
            action, features['price_trend'], features['rsi_level']
        )
        features['pattern_key'] = pattern_key
        
        return features
    
    def _categorize_trend(self, price_change: float) -> str:
        """Categorize price trend.
        
        Args:
            price_change: 24h price change percentage.
            
        Returns:
            Trend category.
        """
        if price_change > TREND_STRONG_UP_THRESHOLD:
            return 'strong_up'
        elif price_change > TREND_UP_THRESHOLD:
            return 'up'
        elif price_change < TREND_STRONG_DOWN_THRESHOLD:
            return 'strong_down'
        elif price_change < TREND_DOWN_THRESHOLD:
            return 'down'
        else:
            return 'sideways'
    
    def _categorize_volume(self, volume_ratio: float) -> str:
        """Categorize volume level.
        
        Args:
            volume_ratio: Volume ratio compared to average.
            
        Returns:
            Volume category.
        """
        if volume_ratio > VOLUME_VERY_HIGH_THRESHOLD:
            return 'very_high'
        elif volume_ratio > VOLUME_HIGH_THRESHOLD:
            return 'high'
        elif volume_ratio < VOLUME_LOW_THRESHOLD:
            return 'low'
        else:
            return 'normal'
    
    def _categorize_rsi(self, rsi: float) -> str:
        """Categorize RSI level.
        
        Args:
            rsi: RSI value.
            
        Returns:
            RSI category.
        """
        if rsi > RSI_OVERBOUGHT_THRESHOLD:
            return 'overbought'
        elif rsi < RSI_OVERSOLD_THRESHOLD:
            return 'oversold'
        else:
            return 'neutral'
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility level.
        
        Args:
            volatility: Volatility value.
            
        Returns:
            Volatility category.
        """
        if volatility > VOLATILITY_HIGH_THRESHOLD:
            return 'high'
        elif volatility < VOLATILITY_LOW_THRESHOLD:
            return 'low'
        else:
            return 'medium'
    
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
        self,
        pattern: Dict[str, Any],
        outcome: Dict[str, Any]
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
        self,
        conn: sqlite3.Connection,
        pattern: Dict[str, Any]
    ) -> int:
        """Get existing pattern ID or create new pattern.
        
        Args:
            conn: Database connection.
            pattern: Pattern features.
            
        Returns:
            Pattern ID.
        """
        cursor = conn.execute('''
            SELECT id FROM learned_patterns
            WHERE pattern_type = ?
            LIMIT 1
        ''', (pattern['pattern_key'],))
        
        row = cursor.fetchone()
        if row:
            return row[0]
        
        # Create new pattern
        cursor = conn.execute('''
            INSERT INTO learned_patterns 
            (pattern_type, pattern_data, success_rate, occurrence_count, last_seen)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            pattern['pattern_key'],
            json.dumps(pattern),
            NEUTRAL_SUCCESS_RATE,
            0,
            datetime.now()
        ))
        return cursor.lastrowid
    
    def _insert_pattern_outcome(
        self,
        conn: sqlite3.Connection,
        pattern_id: int,
        outcome: Dict[str, Any]
    ) -> None:
        """Insert pattern outcome record.
        
        Args:
            conn: Database connection.
            pattern_id: Pattern ID.
            outcome: Trade outcome.
        """
        outcome_type = 'success' if outcome.get('profit_loss', 0) > 0 else 'failure'
        
        conn.execute('''
            INSERT INTO pattern_outcomes
            (pattern_id, trade_id, outcome, profit_loss, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            pattern_id,
            outcome.get('trade_id', ''),
            outcome_type,
            outcome.get('profit_loss', 0),
            datetime.now()
        ))
    
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
        self,
        conn: sqlite3.Connection,
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate pattern statistics from outcomes.
        
        Args:
            conn: Database connection.
            pattern: Pattern features.
            
        Returns:
            Statistics dictionary or None.
        """
        cursor = conn.execute('''
            SELECT 
                p.id,
                COUNT(po.id) as total_outcomes,
                SUM(CASE WHEN po.outcome = 'success' THEN 1 ELSE 0 END) as successes
            FROM learned_patterns p
            JOIN pattern_outcomes po ON p.id = po.pattern_id
            WHERE p.pattern_type = ?
            GROUP BY p.id
        ''', (pattern['pattern_key'],))
        
        row = cursor.fetchone()
        if row:
            pattern_id, total, successes = row
            success_rate = successes / total if total > 0 else NEUTRAL_SUCCESS_RATE
            
            return {
                'pattern_id': pattern_id,
                'success_rate': success_rate,
                'total': total
            }
        return None
    
    def _update_pattern_record(
        self,
        conn: sqlite3.Connection,
        stats: Dict[str, Any]
    ) -> None:
        """Update pattern record with new statistics.
        
        Args:
            conn: Database connection.
            stats: Pattern statistics.
        """
        conn.execute('''
            UPDATE learned_patterns
            SET success_rate = ?, occurrence_count = ?, last_seen = ?
            WHERE id = ?
        ''', (
            stats['success_rate'],
            stats['total'],
            datetime.now(),
            stats['pattern_id']
        ))
    
    def _should_generate_lesson(self, pattern: Dict[str, Any]) -> bool:
        """Determine if pattern warrants generating a lesson.
        
        Args:
            pattern: Pattern features.
            
        Returns:
            True if lesson should be generated.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT occurrence_count, success_rate
                    FROM learned_patterns
                    WHERE pattern_type = ?
                ''', (pattern['pattern_key'],))
                
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
        return (
            occurrences >= MIN_OCCURRENCES_FOR_LESSON and
            (success_rate > HIGH_SUCCESS_THRESHOLD or 
             success_rate < LOW_SUCCESS_THRESHOLD)
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
        self,
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get pattern statistics for lesson generation.
        
        Args:
            pattern: Pattern features.
            
        Returns:
            Pattern statistics or None.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT pattern_data, success_rate, occurrence_count
                FROM learned_patterns
                WHERE pattern_type = ?
            ''', (pattern['pattern_key'],))
            
            row = cursor.fetchone()
            if row:
                return {
                    'pattern_data': json.loads(row[0]),
                    'success_rate': row[1],
                    'occurrences': row[2]
                }
        return None
    
    def _generate_lesson_with_ai(
        self,
        pattern: Dict[str, Any],
        pattern_stats: Dict[str, Any]
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
            temperature=LESSON_GENERATION_TEMPERATURE
        )
        
        return result.get('response', '')
    
    def _get_lesson_generation_system_message(self) -> str:
        """Get system message for lesson generation.
        
        Returns:
            System message string.
        """
        return """You are a trading strategy analyst. Generate a concise, 
        actionable trading lesson from pattern analysis. Focus on when to apply 
        or avoid certain actions based on market conditions."""
    
    def _create_lesson_generation_prompt(
        self,
        pattern: Dict[str, Any],
        pattern_stats: Dict[str, Any]
    ) -> str:
        """Create prompt for lesson generation.
        
        Args:
            pattern: Pattern features.
            pattern_stats: Pattern statistics.
            
        Returns:
            Prompt string.
        """
        pattern_data = pattern_stats['pattern_data']
        success_rate = pattern_stats['success_rate']
        occurrences = pattern_stats['occurrences']
        
        focus = 'what works' if success_rate > NEUTRAL_SUCCESS_RATE else 'what to avoid'
        
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
        self,
        lesson_content: str,
        pattern_stats: Dict[str, Any]
    ) -> None:
        """Store generated trading lesson.
        
        Args:
            lesson_content: Lesson text.
            pattern_stats: Pattern statistics.
        """
        success_rate = pattern_stats['success_rate']
        lesson_type = 'success' if success_rate > NEUTRAL_SUCCESS_RATE else 'failure'
        confidence = abs(success_rate - NEUTRAL_SUCCESS_RATE) * 2
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trading_lessons
                (lesson_type, lesson_content, confidence)
                VALUES (?, ?, ?)
            ''', (lesson_type, lesson_content, confidence))
        
        logger.info(f"Generated new trading lesson: {lesson_content}")
    
    def _find_matching_patterns(
        self,
        symbol: str,
        market_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find patterns matching current conditions.
        
        Args:
            symbol: Trading symbol.
            market_conditions: Current market conditions.
            
        Returns:
            List of matching patterns.
        """
        try:
            # Extract current features
            current_features = self._extract_pattern_features(
                symbol, 'unknown', market_conditions
            )
            
            # Find similar patterns
            with sqlite3.connect(self.db_path) as conn:
                all_patterns = self._fetch_eligible_patterns(conn)
                return self._filter_matching_patterns(all_patterns, current_features)
                
        except Exception as e:
            logger.error(f"Failed to find matching patterns: {e}")
            return []
    
    def _fetch_eligible_patterns(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Fetch patterns eligible for matching.
        
        Args:
            conn: Database connection.
            
        Returns:
            List of eligible patterns.
        """
        cursor = conn.execute('''
            SELECT pattern_type, pattern_data, success_rate, occurrence_count
            FROM learned_patterns
            WHERE occurrence_count >= ?
            ORDER BY occurrence_count DESC
        ''', (MIN_OCCURRENCES_FOR_MATCHING,))
        
        patterns = []
        for row in cursor:
            patterns.append({
                'type': row[0],
                'data': json.loads(row[1]),
                'success_rate': row[2],
                'occurrences': row[3]
            })
        return patterns
    
    def _filter_matching_patterns(
        self,
        patterns: List[Dict[str, Any]],
        current_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter patterns that match current features.
        
        Args:
            patterns: List of patterns.
            current_features: Current market features.
            
        Returns:
            List of matching patterns.
        """
        matching = []
        for pattern in patterns:
            if self._patterns_match(current_features, pattern['data']):
                matching.append(pattern)
        return matching
    
    def _patterns_match(
        self,
        current: Dict[str, Any],
        pattern: Dict[str, Any]
    ) -> bool:
        """Check if patterns match based on key features.
        
        Args:
            current: Current features.
            pattern: Pattern features.
            
        Returns:
            True if patterns match.
        """
        return (
            current.get('price_trend') == pattern.get('price_trend') and
            current.get('rsi_level') == pattern.get('rsi_level') and
            current.get('volatility') == pattern.get('volatility')
        )
    
    def _aggregate_pattern_recommendations(
        self,
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate recommendations from multiple patterns.
        
        Args:
            patterns: List of matching patterns.
            
        Returns:
            Aggregated recommendations.
        """
        action_scores = self._calculate_action_scores(patterns)
        
        if not action_scores:
            return {'confidence': DEFAULT_CONFIDENCE, 'actions': []}
        
        sorted_actions = self._sort_actions_by_score(action_scores)
        confidence = self._calculate_recommendation_confidence(
            sorted_actions, sum(action_scores.values())
        )
        
        return {
            'confidence': confidence,
            'actions': [action for action, _ in sorted_actions[:MAX_ACTION_RECOMMENDATIONS]]
        }
    
    def _calculate_action_scores(
        self,
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate scores for each action based on patterns.
        
        Args:
            patterns: List of patterns.
            
        Returns:
            Dictionary of action scores.
        """
        action_scores = defaultdict(float)
        
        for pattern in patterns:
            weight = self._calculate_pattern_weight(pattern)
            action = pattern['data'].get('action', 'hold')
            
            if pattern['success_rate'] > SUCCESS_ACTION_THRESHOLD:
                action_scores[action] += weight
            elif pattern['success_rate'] < FAILURE_ACTION_THRESHOLD:
                # Recommend opposite action for failed patterns
                opposite_action = self._get_opposite_action(action)
                action_scores[opposite_action] += weight
        
        return dict(action_scores)
    
    def _calculate_pattern_weight(self, pattern: Dict[str, Any]) -> float:
        """Calculate weight for pattern based on occurrences and success rate.
        
        Args:
            pattern: Pattern data.
            
        Returns:
            Pattern weight.
        """
        return pattern['occurrences'] * abs(pattern['success_rate'] - NEUTRAL_SUCCESS_RATE)
    
    def _get_opposite_action(self, action: str) -> str:
        """Get opposite trading action.
        
        Args:
            action: Original action.
            
        Returns:
            Opposite action.
        """
        opposites = {
            'buy': 'hold',
            'sell': 'hold',
            'hold': 'hold'
        }
        return opposites.get(action, 'hold')
    
    def _sort_actions_by_score(
        self,
        action_scores: Dict[str, float]
    ) -> List[tuple]:
        """Sort actions by score in descending order.
        
        Args:
            action_scores: Dictionary of action scores.
            
        Returns:
            Sorted list of (action, score) tuples.
        """
        return sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_recommendation_confidence(
        self,
        sorted_actions: List[tuple],
        total_weight: float
    ) -> float:
        """Calculate confidence for recommendations.
        
        Args:
            sorted_actions: Sorted action list.
            total_weight: Total weight of all actions.
            
        Returns:
            Confidence score between 0 and 1.
        """
        if not sorted_actions or total_weight == 0:
            return DEFAULT_CONFIDENCE
        
        top_score = sorted_actions[0][1]
        return min(top_score / total_weight, 1.0)
    
    def _get_recent_failure_patterns(self, days_back: int) -> List[Dict[str, Any]]:
        """Get recent failure patterns from database.
        
        Args:
            days_back: Number of days to look back.
            
        Returns:
            List of failure patterns.
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT pattern_type, pattern_data, success_rate, occurrence_count
                FROM learned_patterns
                WHERE success_rate < ?
                AND last_seen > ?
                ORDER BY occurrence_count DESC
                LIMIT ?
            ''', (FAILURE_PATTERN_THRESHOLD, cutoff_date, MAX_FAILURE_PATTERNS))
            
            failure_patterns = []
            for row in cursor:
                failure_patterns.append({
                    'type': row[0],
                    'data': json.loads(row[1]),
                    'success_rate': row[2],
                    'occurrences': row[3]
                })
        
        return failure_patterns
    
    def _analyze_failure_factors(
        self,
        failure_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze common factors in failure patterns.
        
        Args:
            failure_patterns: List of failure patterns.
            
        Returns:
            Analysis results.
        """
        if not failure_patterns:
            return {'overall_failure_rate': 0, 'common_factors': []}
        
        factor_counts = self._count_failure_factors(failure_patterns)
        common_factors = self._identify_common_factors(factor_counts, len(failure_patterns))
        high_risk_conditions = self._identify_high_risk_conditions(failure_patterns)
        overall_failure_rate = self._calculate_overall_failure_rate(failure_patterns)
        
        return {
            'overall_failure_rate': overall_failure_rate,
            'common_factors': common_factors,
            'high_risk_conditions': high_risk_conditions
        }
    
    def _count_failure_factors(
        self,
        failure_patterns: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count occurrences of each factor in failure patterns.
        
        Args:
            failure_patterns: List of failure patterns.
            
        Returns:
            Factor count dictionary.
        """
        factor_counts = defaultdict(int)
        
        for pattern in failure_patterns:
            data = pattern['data']
            factor_counts[f"trend_{data.get('price_trend')}"] += 1
            factor_counts[f"rsi_{data.get('rsi_level')}"] += 1
            factor_counts[f"vol_{data.get('volatility')}"] += 1
            factor_counts[f"action_{data.get('action')}"] += 1
        
        return dict(factor_counts)
    
    def _identify_common_factors(
        self,
        factor_counts: Dict[str, int],
        total_failures: int
    ) -> List[str]:
        """Identify factors common to many failures.
        
        Args:
            factor_counts: Factor occurrence counts.
            total_failures: Total number of failures.
            
        Returns:
            List of common factors.
        """
        return [
            factor for factor, count in factor_counts.items()
            if count / total_failures > COMMON_FACTOR_THRESHOLD
        ]
    
    def _identify_high_risk_conditions(
        self,
        failure_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify high-risk trading conditions.
        
        Args:
            failure_patterns: List of failure patterns.
            
        Returns:
            List of high-risk conditions.
        """
        high_risk = []
        for pattern in failure_patterns:
            if (pattern['success_rate'] < HIGH_RISK_SUCCESS_THRESHOLD and 
                pattern['occurrences'] > HIGH_RISK_MIN_OCCURRENCES):
                high_risk.append({
                    'condition': pattern['type'],
                    'success_rate': pattern['success_rate']
                })
        return high_risk
    
    def _calculate_overall_failure_rate(
        self,
        failure_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate weighted overall failure rate.
        
        Args:
            failure_patterns: List of failure patterns.
            
        Returns:
            Overall failure rate.
        """
        total_occurrences = sum(p['occurrences'] for p in failure_patterns)
        if total_occurrences == 0:
            return 0
        
        weighted_failures = sum(
            (1 - p['success_rate']) * p['occurrences'] 
            for p in failure_patterns
        )
        
        return weighted_failures / total_occurrences
    
    def _generate_failure_recommendations(
        self,
        failure_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on failure analysis.
        
        Args:
            failure_analysis: Failure analysis results.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Add factor-based recommendations
        recommendations.extend(
            self._get_factor_recommendations(failure_analysis.get('common_factors', []))
        )
        
        # Add high-risk condition recommendations
        recommendations.extend(
            self._get_high_risk_recommendations(failure_analysis.get('high_risk_conditions', []))
        )
        
        # Add general recommendations
        failure_rate = failure_analysis.get('overall_failure_rate', 0)
        if failure_rate > HIGH_FAILURE_RATE_THRESHOLD:
            recommendations.append("Consider more conservative trading approach")
        
        return recommendations[:MAX_RECOMMENDATIONS]
    
    def _get_factor_recommendations(self, common_factors: List[str]) -> List[str]:
        """Get recommendations based on common failure factors.
        
        Args:
            common_factors: List of common factors.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        factor_recommendations = {
            'trend_strong_down': "Avoid buying during strong downtrends",
            'rsi_overbought': "Avoid buying when RSI indicates overbought",
            'vol_high': "Reduce position sizes during high volatility",
            'action_buy': "Review buy criteria - high failure rate detected"
        }
        
        for factor in common_factors:
            for key, recommendation in factor_recommendations.items():
                if key in factor:
                    recommendations.append(recommendation)
                    break
        
        return recommendations
    
    def _get_high_risk_recommendations(
        self,
        high_risk_conditions: List[Dict[str, Any]]
    ) -> List[str]:
        """Get recommendations for high-risk conditions.
        
        Args:
            high_risk_conditions: List of high-risk conditions.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        for condition in high_risk_conditions[:MAX_HIGH_RISK_CONDITIONS]:
            recommendations.append(
                f"Avoid {condition['condition']} "
                f"(success rate: {condition['success_rate']:.1%})"
            )
        
        return recommendations
    
    def _get_empty_recommendations(self) -> Dict[str, Any]:
        """Get empty recommendations structure.
        
        Returns:
            Empty recommendations dictionary.
        """
        return {
            'has_recommendations': False,
            'confidence': DEFAULT_CONFIDENCE,
            'recommendations': []
        }
    
    def _get_empty_failure_analysis(self) -> Dict[str, Any]:
        """Get empty failure analysis structure.
        
        Returns:
            Empty failure analysis dictionary.
        """
        return {
            'failure_rate': 0,
            'common_factors': [],
            'high_risk_conditions': [],
            'recommendations': [],
            'pattern_count': 0
        }
    
    def _fetch_high_confidence_lessons(
        self,
        conn: sqlite3.Connection
    ) -> List[tuple]:
        """Fetch high-confidence lessons from database.
        
        Args:
            conn: Database connection.
            
        Returns:
            List of lesson tuples.
        """
        cursor = conn.execute('''
            SELECT lesson_content, confidence, success_count, application_count
            FROM trading_lessons
            WHERE confidence > ?
            AND application_count > ?
            ORDER BY (success_count * 1.0 / application_count) DESC
            LIMIT ?
        ''', (MIN_LESSON_CONFIDENCE, MIN_LESSON_APPLICATIONS, MAX_LESSONS_FOR_PROMPT))
        
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
            success_rate = success_count / application_count if application_count > 0 else 0
            
            if success_rate > MIN_LESSON_SUCCESS_RATE:
                successful_lessons.append(content)
        
        return successful_lessons


# Create singleton instance
pattern_learner = PatternLearner()