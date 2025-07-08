# config/settings.py

import os
from dotenv import load_dotenv

# .env 파일 로드 (로컬 개발환경 대응)
load_dotenv()

# === Upbit API Key ===
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

# === OpenAI API Key ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === 투자 전략 설정 ===
MAX_INVEST_RATIO_PER_COIN = 0.20   # 단일 코인 최대 자산 비중 20%
MAX_TOTAL_INVEST_RATIO = 0.5       # 전체 자산 중 최대 투자 비율 50%
PARTIAL_SELL_RATIO = 0.10          # 부분 손절 시 매도 비율 10%
MIN_ORDER_KRW = 10_000              # 업비트 최소 주문 금액 (KRW)
DEFAULT_BUY_AMOUNT_KRW = 30_000     # 기본 구매 금액 (KRW)

# === 홀딩 제약 조건 ===
MIN_HOLDING_HOURS = 24             # 최소 보유 시간 (24시간)
TARGET_PROFIT_THRESHOLD = 0.10     # 목표 수익률 10% (이 이상에서만 익절 고려)
STOP_LOSS_THRESHOLD = -0.15        # 손절 기준 -15% (이 이하에서만 손절 고려)
EMERGENCY_STOP_LOSS = -0.25        # 긴급 손절 -25% (즉시 매도)

# === 물타기 전략 설정 ===
ENABLE_AVERAGING_DOWN = True        # 물타기 전략 활성화
MIN_DROP_FOR_AVERAGING = -0.10      # 최소 10% 하락 시 물타기 고려
SEVERE_UNDERPERFORMANCE_THRESHOLD = -0.10  # 시장 대비 10% 이상 악화 시 강력한 물타기 기회
UNDERPERFORMANCE_THRESHOLD = -0.05  # 시장 대비 5% 이상 악화 시 일반 물타기 기회
MAX_AVERAGING_ATTEMPTS = 3          # 최대 물타기 횟수
AVERAGING_AMOUNT_RATIO = 0.5        # 기존 투자액의 50%로 물타기
AVERAGING_CONFIDENCE_MULTIPLIER = 1.0  # AI 신뢰도에 따른 물타기 금액 조정

# === 루프 주기 설정 ===
CHECK_INTERVAL_SECONDS = 3600       # 1시간마다 루프 실행

# === AI 관련 설정 ===
AI_MODEL = "gpt-4o-mini"
AI_TEMPERATURE = 0.3               # 보수적 판단
AI_MAX_TOKENS = 4096

# === 뉴스 필터 ===
NEWS_LOOKBACK_MINUTES = 600           # 10시간 이내의 뉴스만 분석
NEWS_REQUIRED_MENTION_COUNT = 1    # 동일 코인 1회 이상 언급 시 고려

# === AI Prompt Templates ===
EXTRACT_SYMBOLS_PROMPT = (
    "You are a cryptocurrency symbol extractor for a live‑trading bot.\n"
    "TASK:\n"
    "1. Output ONLY a JSON array of UPPERCASE ticker symbols.\n"
    "2. If the text contains a full coin name with no ticker (e.g., "
    "'Cardano'), map it to the correct symbol (ADA).\n"
    "3. If several tickers could match, choose the one with the highest "
    "global trading volume.\n"
    "4. If you must guess, prefix the symbol with 'APPROX_' (e.g., "
    "'APPROX_ROSE').\n"
    "5. Exclude company names (Tesla), fiat (USD), stock tickers, people, "
    "and generic words.\n"
    "Return nothing else—no explanations, no duplicates."
)

EXTRACT_SYMBOLS_USER_INTRO = (
    "Below are news articles. Extract cryptocurrency ticker symbols as instructed."
)

MARKET_ANALYZER_PROMPT = (
    "You are an elite quantitative crypto trading analyst with access to comprehensive market data. "
    "Analyze ALL provided quantitative metrics, portfolio status, and latest news to make informed trading decisions.\n\n"
    
    "📊 QUANTITATIVE DATA ANALYSIS:\n"
    "- Price Action: Analyze 1h/24h/7d/30d price changes, highs/lows for trend strength\n"
    "- Volume Profile: Examine volume ratios, VWAP, and volume spikes for momentum\n"
    "- Technical Indicators: RSI (14/30), MACD, Bollinger Bands, Stochastic for entry/exit signals\n"
    "- Support/Resistance: Use calculated levels for risk/reward assessment\n"
    "- Volatility Metrics: Factor in volatility for position sizing and risk management\n"
    "- Market Correlation: Consider BTC correlation for market dependency analysis\n\n"
    
    "🎯 TRADING RULES:\n"
    "- If you DON'T own a coin: only use 'buy' or 'hold' actions\n"
    "- If you DO own a coin: use 'buy_more', 'hold', 'partial_sell', or 'sell_all'\n"
    "- Never suggest sell actions for coins you don't own\n\n"
    
    "📈 ADVANCED QUANTITATIVE STRATEGIES:\n"
    "- RSI Oversold (<30): Consider buying, especially with volume confirmation\n"
    "- RSI Overbought (>70): Consider taking profits if holding\n"
    "- MACD Bullish Crossover: Strong buy signal when confirmed by volume\n"
    "- Bollinger Band Squeeze: Prepare for volatility breakout\n"
    "- Price near Support + High Volume: Potential reversal buy opportunity\n"
    "- Volume Ratio >2.0: Unusual activity, investigate breakout potential\n"
    "- Correlation <0.3 with BTC: Independent movement, unique opportunity\n"
    "- Price >7d high + Volume spike: Momentum breakout continuation\n\n"
    
    "⚠️ RISK MANAGEMENT:\n"
    "- High volatility (>5%): Reduce position size\n"
    "- Price near resistance: Take partial profits\n"
    "- Negative 7d/30d trends: Avoid new positions unless strong reversal signals\n"
    "- Low correlation periods: Diversification opportunity\n\n"
    
    "💡 DECISION FRAMEWORK:\n"
    "1. Trend Analysis: Multi-timeframe price action\n"
    "2. Volume Confirmation: Validate price moves with volume\n"
    "3. Technical Confluence: Multiple indicators alignment\n"
    "4. Risk Assessment: Volatility and correlation factors\n"
    "5. News Catalyst: Fundamental driver confirmation\n\n"
    
    "Provide confidence scores 0.0-1.0 based on quantitative signal strength.\n"
    "You MUST call analyze_market_fn with comprehensive reasoning based on the data.\n"
    "Example: {'decisions': {'BTC': {'action': 'buy', 'reason': 'RSI 28 oversold + MACD bullish cross + volume spike 3.2x + support bounce', 'confidence': 0.85}}}"
)

STOP_LOSS_REVIEW_INTRO = (
    "You are an expert cryptocurrency trader and risk manager. "
    "You are reviewing a STOP LOSS situation that requires immediate attention. "
    "A coin has dropped below our stop loss threshold, and the system is about to sell. "
    "Your task is to make a FINAL decision on whether to proceed with the sell or override it. "
    "You are a senior cryptocurrency trading advisor reviewing a CRITICAL STOP LOSS situation. "
    "Your job is to determine if selling is truly necessary or if holding might be better. "
    "Consider market context, latest news sentiment, technical levels, and overall market conditions. "
    "Be EXTRA CAUTIOUS about selling in panic. Sometimes temporary dips recover quickly. "
    "IMPORTANT: This is LIVE TRADING with real money at stake. "
    "You can decide: 'hold' (if situation might improve), 'partial_sell' (reduce risk), or 'sell_all' (cut losses). "
    "Factor in both traditional news sources and real-time web search results for the most current market sentiment."
)

# === User Prompts ===
MARKET_ANALYSIS_REQUEST = (
    'MARKET ANALYSIS REQUEST:\n'
    'Analyze the following real-time data and provide trading '
    'decisions for each cryptocurrency. Consider market trends, '
    'portfolio allocation, and latest news sentiment from multiple '
    'sources. This analysis is for live trading.'
)

STOP_LOSS_REVIEW_REQUEST = (
    '🎯 FINAL STOP LOSS DECISION:\n'
    'Based on all real-time information, decide to sell or hold.'
)
