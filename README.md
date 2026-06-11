# Upbit AI Trading Bot

AI-assisted automated trading system for the Upbit exchange: news collection, market analysis, AI decision making, and trade execution with layered safety systems.

AI 기반 업비트 암호화폐 자동 거래 시스템 — 뉴스 수집, 시장 분석, AI 의사결정, 거래 실행을 안전장치와 함께 자동화합니다.

> ⚠️ **실제 자금이 거래되는 라이브 트레이딩 봇입니다.** 암호화폐 거래에는 손실 위험이 따릅니다. 본인 책임 하에 사용하세요. This is live trading with real funds — use at your own risk.

## 주요 기능 (Features)

- **AI 의사결정**: 로컬 OAuth bridge(`http://127.0.0.1:8787/v1`)를 통해 시장 데이터·뉴스·포트폴리오를 분석해 매수/매도/보유를 결정
- **뉴스 수집**: 암호화폐 RSS 피드를 수집한 뒤 소스 신뢰도, 신선도, 관련성, URL/제목 중복을 기준으로 필터링·정렬
- **기술적 분석**: RSI, MACD, Bollinger Bands, ADX/DI, ATR%, EMA 정렬, Donchian 위치, 거래대금/상대 거래량 등 정량 지표 산출
- **다중 AI 검증**: 동일 모델로 독립적인 교차 검증 단계를 거쳐 의사결정 신뢰도 보강
- **적응형 리스크 관리**: 포지션 사이징, 손절매 추천, 포트폴리오 노출 한도 적용
- **물타기(Averaging Down) 전략**: 설정 가능한 조건에서 추가 매수
- **AI 학습 시스템**: 과거 거래를 분석해 성공률·실패 패턴 인사이트를 도출
- **패턴 학습**: 거래 패턴을 SQLite에 저장하고 향후 의사결정에 교훈으로 반영
- **안전장치**: Circuit Breaker, 실시간 리스크 모니터링, 자동 손절매, 거래 후 분석

## 요구 사항 (Prerequisites)

- Python 3.11 이상
- [Poetry](https://python-poetry.org/) (의존성 관리, 권장)
- Upbit Open API 키 (Access / Secret)
- 로컬 ChatGPT/Codex OAuth bridge 실행 환경 (`http://127.0.0.1:8787/v1`)

## 설치 (Installation)

```bash
# 저장소 클론
git clone https://github.com/sueun-dev/upbit-ai-trading-bot.git
cd upbit-ai-trading-bot

# 의존성 설치 (Poetry)
poetry install
```

Poetry를 사용하지 않는 경우 `pyproject.toml`에 명시된 런타임 의존성(`pyupbit`, `requests`, `python-dotenv`, `lxml`, `pandas`, `feedparser`, `chardet` 등)을 직접 설치하세요.

## 환경 변수 설정 (Configuration)

업비트 API 키와 OAuth bridge 설정은 환경 변수 또는 프로젝트 루트의 `.env` 파일로 제공합니다. (`python-dotenv`로 자동 로드되며, `.env`는 `.gitignore`에 포함되어 있습니다.)

```bash
# .env
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key

# Optional: defaults shown
AI_BRIDGE_BASE_URL=http://127.0.0.1:8787/v1
AI_MODEL=gpt-5.5
AI_REASONING_EFFORT=xhigh
```

`UPBIT_ACCESS_KEY` 또는 `UPBIT_SECRET_KEY`가 없으면 프로그램이 시작 시 명확한 메시지와 함께 종료됩니다. AI 호출은 OpenAI Platform API 키를 직접 쓰지 않고 로컬 OAuth bridge로 전송됩니다. bridge 로그인 토큰이 만료되었거나 서버가 꺼져 있으면 AI 요청 단계에서 명확한 오류가 납니다.

기타 전략·루프 설정값은 `src/infrastructure/config/settings.py`에서 조정할 수 있습니다 (예: `CHECK_INTERVAL_SECONDS`, 투자 비율, 손절 기준, 사용 AI 모델 등).

## 실행 (Usage)

```bash
# Poetry 환경에서 실행
poetry run python main.py

# 또는 콘솔 스크립트 사용 (poetry install 이후)
poetry run upbit-trader

# Poetry 없이 직접 실행
python main.py
```

실행하면 메인 루프가 시작되어 `CHECK_INTERVAL_SECONDS`(기본 3600초 = 1시간)마다 거래 사이클을 수행합니다. 연속 에러가 `MAX_CONSECUTIVE_ERRORS`(5회)를 초과하면 시스템이 안전하게 정지합니다.

## 테스트 (Testing)

```bash
# 개발 의존성 포함 설치
poetry install

# 테스트 실행
poetry run pytest

# 커버리지 포함
poetry run pytest --cov=src
```

코드 품질 도구도 dev 의존성으로 제공됩니다:

```bash
poetry run black src tests      # 코드 포매팅
poetry run isort src tests      # import 정렬
poetry run flake8 src tests     # 린팅
poetry run mypy src             # 타입 체크
```

## 프로젝트 구조 (Project Layout)

```
.
├── main.py                          # 진입점: AI 학습 초기화 후 메인 거래 루프 실행
├── pyproject.toml                   # Poetry 패키지·의존성·도구 설정
├── src/
│   ├── core/
│   │   ├── orchestrator/            # 거래 사이클 오케스트레이션
│   │   └── clients/                 # Upbit 트레이더, Circuit Breaker
│   ├── analysis/                    # AI 분석, 다중 AI 검증, 리스크/포트폴리오/패턴 학습
│   ├── data/
│   │   ├── collectors/              # 시장 데이터·기술적 지표 수집
│   │   └── scrapers/                # RSS 기반 뉴스 수집
│   ├── infrastructure/
│   │   ├── config/                  # 설정값·프롬프트 템플릿
│   │   └── database/                # DB 경로 헬퍼
│   └── shared/                      # 공통 상수, OpenAI 클라이언트, 유틸/데이터 스토어
└── tests/                           # pytest 테스트 스위트
```

## 시스템 전체 로직 플로우

### 1. 메인 프로그램 시작 (main.py)

#### 1.1 초기화 단계
```python
# 로깅 설정
setup_logging()

# AI 학습 시스템 초기화
trade_analyzer = AILearningSystem()

# 과거 거래 분석 (30일, AI_LEARNING_DAYS_BACK)
insights = trade_analyzer.analyze_historical_trades(days_back=30)
# Return 예시:
# [
#   {
#     "type": "success_rate",
#     "metrics": {
#       "total_trades": 50,
#       "successful_trades": 35,
#       "success_rate": 70.0,
#       "avg_profit": 5.2,
#       "avg_loss": -3.1
#     },
#     "performance": "excellent",
#     "recommendation": "Continue with current strategy"
#   },
#   {
#     "type": "failure_analysis",
#     "metrics": {
#       "total_failures": 15,
#       "most_common_reason": "stop_loss_triggered"
#     }
#   }
# ]
```

#### 1.2 Trading Orchestrator 생성
```python
orchestrator = TradingOrchestrator(
    access_key=UPBIT_ACCESS_KEY,
    secret_key=UPBIT_SECRET_KEY,
    trade_analyzer=trade_analyzer
)
```

#### 1.3 메인 거래 루프 실행
```python
run_main_trading_loop(orchestrator)
# CHECK_INTERVAL_SECONDS(기본 3600초)마다 거래 사이클 실행
# 최대 연속 에러 5회(MAX_CONSECUTIVE_ERRORS)까지 허용
```

### 2. 거래 사이클 실행 (TradingOrchestrator.run_trading_cycle)

#### 2.1 데이터 수집 및 검증 (_collect_and_validate_data)

##### 2.1.1 포트폴리오 상태 확인
```python
portfolio = self.trader.get_portfolio_status()
# Return 예시:
# {
#   "total_balance": 1000000,
#   "total_krw": 1000000,
#   "total_investment": 950000,
#   "available_krw": 500000,
#   "assets": {
#     "KRW": {
#       "balance": 500000,
#       "locked": 0
#     },
#     "BTC": {
#       "balance": 0.001234,
#       "avg_buy_price": 65000000,
#       "current_price": 68000000,
#       "krw_value": 83872,
#       "profit_loss": 3872,
#       "profit_loss_percentage": 4.84
#     }
#   },
#   "holdings": {
#     "BTC": {
#       "balance": 0.001234,
#       "avg_buy_price": 65000000
#     }
#   }
# }
```

##### 2.1.2 뉴스 수집
```python
news_list = self.collect_news()
# 다수의 RSS 피드에서 기사 수집
# Return 예시:
# [
#   {
#     "title": "비트코인 급등, 7만 달러 돌파",
#     "summary": "비트코인이 기관 투자자들의 매수세로 급등...",
#     "source": "CoinDesk",
#     "url": "https://example.com/news/123"
#   },
#   {
#     "title": "이더리움 업그레이드 완료",
#     "summary": "이더리움 덴쿤 업그레이드가 성공적으로...",
#     "source": "CoinTelegraph",
#     "url": "https://example.com/news/456"
#   }
# ]
```

##### 2.1.3 뉴스에서 심볼 추출
```python
symbols = self.extract_market_symbols(news_list)
# Return 예시: ["BTC", "ETH", "DOGE"]

# 보유 종목 심볼 추가
portfolio_symbols = list(portfolio['holdings'].keys())
symbols = list(set(symbols + portfolio_symbols))
# 최종 심볼 예시: ["BTC", "ETH", "DOGE", "XRP", "ADA"]
```

##### 2.1.4 시장 데이터 수집
```python
market_data = self.collect_market_data(symbols)
# Return 예시:
# {
#   "BTC": {
#     "symbol": "BTC",
#     "current_price": 68000000,
#     "is_held": True,
#     "price_24h_change": 5.2,
#     "volume_24h": 123456789,
#     "volume_ratio_24h_7d": 1.15,
#     "rsi_14": 65.5,
#     "macd": 150.0,
#     "macd_signal": 120.0,
#     "bb_upper": 69000000,
#     "bb_middle": 67000000,
#     "bb_lower": 65000000,
#     "volatility_7d": 0.045,
#     "support_1": 66500000,
#     "resistance_1": 68500000,
#     "comprehensive_analysis": {
#       "overall_signal": "BUY",
#       "strength": 0.75,
#       "technical_indicators": {...},
#       "volume_analysis": {...}
#     }
#   }
# }
```

#### 2.2 AI 의사결정 생성 (_get_ai_decisions_with_safety)

##### 2.2.1 패턴 학습 교훈 획득
```python
trading_lessons = self._get_pattern_learning_lessons()
# Return 예시:
# [
#   "Avoid buying during low volume periods",
#   "RSI above 70 often leads to short-term corrections",
#   "MACD crossover signals are more reliable in trending markets"
# ]
```

##### 2.2.2 AI 분석을 통한 초기 결정
```python
raw_decisions = self.ai_analyzer.analyze_market_data(
    news_list, market_data, portfolio, trading_lessons
)
# Return 예시:
# {
#   "BTC": {
#     "action": "hold",
#     "reason": "RSI가 65로 중립적이며, 현재 4.84% 수익 중. 추가 상승 모멘텀 대기",
#     "confidence": 0.7
#   },
#   "ETH": {
#     "action": "buy",
#     "reason": "덴쿤 업그레이드 완료로 긍정적 전망. RSI 35로 과매도 구간",
#     "confidence": 0.85,
#     "amount_krw": 30000
#   }
# }
```

##### 2.2.3 과거 거래 인사이트 적용
```python
self._apply_trade_history_insights(raw_decisions)
# 과거 성공률이 낮은 종목의 confidence 감소
# 예: DOGE 성공률 30% → confidence *= 0.5
```

##### 2.2.4 다중 AI 검증
```python
validated_decisions = self.multi_ai_validator.cross_validate_multiple_decisions(
    raw_decisions, market_data, portfolio, news_list
)
# 동일 모델로 독립적인 교차 검증 단계를 수행
# Return 예시:
# {
#   "ETH": {
#     "action": "buy",
#     "reason": "덴쿤 업그레이드 완료로 긍정적 전망. RSI 35로 과매도 구간",
#     "confidence": 0.85,
#     "amount_krw": 30000,
#     "multi_ai_approved": True,
#     "risk_level": "medium",
#     "consensus_score": 0.8
#   }
# }
```

##### 2.2.5 적응형 리스크 관리 적용
```python
final_decisions = self._apply_adaptive_risk_management(
    validated_decisions, portfolio, market_data
)
# Return 예시:
# {
#   "ETH": {
#     "action": "buy",
#     "reason": "덴쿤 업그레이드 완료로 긍정적 전망. RSI 35로 과매도 구간",
#     "confidence": 0.85,
#     "recommended_amount_krw": 30000,
#     "position_sizing": {
#       "recommended_size": 30000,
#       "max_allowed": 200000,
#       "portfolio_percentage": 3.0
#     },
#     "stop_loss_recommendation": {
#       "price": 2100000,
#       "percentage": 0.05,
#       "reason": "AI-optimized stop-loss at 5.0% below current price"
#     }
#   }
# }
```

#### 2.3 리스크 관리 적용 (_apply_risk_management)

##### 2.3.1 포트폴리오 리스크 모니터링
```python
risk_assessment = self.risk_monitor.monitor_active_positions(
    portfolio, market_data, recent_trades
)
# Return 예시:
# {
#   "overall_risk": "medium",
#   "ai_assessment": "Portfolio diversification adequate. Total exposure 50% of capital.",
#   "risk_factors": [
#     {"factor": "concentration", "severity": "low"},
#     {"factor": "volatility", "severity": "medium"}
#   ]
# }
```

##### 2.3.2 손절매 트리거 체크
```python
stop_loss_triggers = self.risk_monitor.check_stop_loss_triggers(portfolio, market_data)
# Return 예시:
# [
#   {
#     "symbol": "DOGE",
#     "current_loss": -0.16,
#     "recommended_action": "sell_all",
#     "reason": "Stop-loss: -16% loss exceeds threshold",
#     "urgency": "high"
#   }
# ]
```

#### 2.4 거래 실행 (execute_trading_decisions)

##### 2.4.1 각 결정에 대한 거래 실행
```python
for symbol, decision in decisions.items():
    executed = self.trader.execute_trade(
        symbol, decision["action"], decision["reason"], decision
    )
    # Return 예시: True (성공) 또는 False (실패)
```

##### 2.4.2 거래 분석 및 기록
```python
# 거래 분석 생성
trade_analysis = TradeAnalysis(
    timestamp="2024-01-20T10:30:00",
    symbol="ETH",
    action="buy",
    action_korean="매수",
    analysis="Executed buy for ETH",
    summary="덴쿤 업그레이드 완료로 긍정적 전망. RSI 35로 과매도 구간",
    confidence=0.85,
    market_context="Price: 2,200,000원"
)

# 거래 기록 생성
trade_record = TradeRecord(
    timestamp="2024-01-20T10:30:00",
    symbol="ETH",
    action="buy",
    confidence=0.85,
    reason="덴쿤 업그레이드 완료로 긍정적 전망. RSI 35로 과매도 구간",
    price=2200000,
    amount_krw=30000,
    quantity=0.0136,
    remaining_quantity=0.0136
)

# 매도 시 수익률 계산
profit_percent, avg_buy_price, used_buy_ids = self.data_store.calculate_sell_profit(
    symbol, current_price, quantity
)
# Return 예시: (12.5, 1950000, [123, 124, 125])
```

#### 2.5 사이클 완료 및 모니터링

##### 2.5.1 분석 결과 기록
```python
analysis_result = AIAnalysisResult(
    timestamp="2024-01-20T10:30:00",
    news_count=15,
    extracted_symbols=["BTC", "ETH", "DOGE"],
    market_data_symbols=["BTC", "ETH", "DOGE", "XRP", "ADA"],
    decisions={...},
    analysis_duration=3.5,
    portfolio_value=1000000,
    circuit_breaker_status="active"
)
```

##### 2.5.2 포트폴리오 상태 모니터링
```python
# 10 사이클마다 상세 리포트
# 📊 PORTFOLIO STATUS REPORT (Detailed)
# ============================================================
# 💰 Total Value: 1,050,000원
# 💵 Total Investment: 1,000,000원
# 📈 Total P&L: +50,000원 (+5.00%)
#
# 🪙 Holding 3 coins:
# ------------------------------------------------------------
# 🟢 BTC: 0.00123400 @ 65,000,000원 → 68,000,000원 | Value: 83,872원 | P&L: +3,872원 (+4.84%)
#    📊 Historical success rate: 75%
# 🟢 ETH: 0.01360000 @ 2,200,000원 → 2,300,000원 | Value: 31,280원 | P&L: +1,360원 (+4.55%)
# 🔴 XRP: 100.00000000 @ 800원 → 750원 | Value: 75,000원 | P&L: -5,000원 (-6.25%)
#
# 💵 Available KRW: 859,848원
# 📈 Last 24h: 7/10 successful trades
```

## 주요 변수 및 상수

### main.py
- `MAX_CONSECUTIVE_ERRORS = 5` - 최대 연속 에러 허용 횟수
- `AI_LEARNING_DAYS_BACK = 30` - AI 학습 분석 기간 (일)

### src/infrastructure/config/settings.py
- `CHECK_INTERVAL_SECONDS = 3600` - 거래 사이클 실행 간격 (초, 기본 1시간)
- `MAX_INVEST_RATIO_PER_COIN = 0.20` - 단일 코인 최대 자산 비중
- `MAX_TOTAL_INVEST_RATIO = 0.5` - 전체 자산 중 최대 투자 비율
- `DEFAULT_BUY_AMOUNT_KRW = 30_000` - 기본 매수 금액
- `MIN_ORDER_KRW = 10_000` - 업비트 최소 주문 금액
- `STOP_LOSS_THRESHOLD = -0.15` - 손절 기준
- `EMERGENCY_STOP_LOSS = -0.25` - 긴급 손절 기준
- `AI_BRIDGE_BASE_URL = "http://127.0.0.1:8787/v1"` - 로컬 OAuth bridge URL
- `AI_MODEL = "gpt-5.5"` - bridge에 요청하는 기본 모델
- `AI_REASONING_EFFORT = "xhigh"` - bridge에 함께 전달하는 추론 강도

### src/core/orchestrator/trading_orchestrator.py
- `DEFAULT_BUY_AMOUNT_KRW = 30_000` - 기본 매수 금액
- `MIN_VIABLE_TRADE_AMOUNT = 10000` - 최소 거래 금액
- `STOP_LOSS_CONFIDENCE = 0.9` - 손절매 신뢰도
- `MAX_NEWS_ARTICLES = 15` - 최대 뉴스 수집 개수

### 거래 액션 타입 (src/shared/constants.py)
- `ACTION_BUY = 'buy'` - 신규 매수
- `ACTION_BUY_MORE = 'buy_more'` - 추가 매수 (물타기)
- `ACTION_SELL_ALL = 'sell_all'` - 전량 매도
- `ACTION_PARTIAL_SELL = 'partial_sell'` - 부분 매도
- `ACTION_HOLD = 'hold'` - 보유 유지

## 시스템 안전장치

1. **Circuit Breaker**: 과도한 거래 방지
2. **다중 AI 검증 (Multi-AI Validation)**: 동일 모델의 독립적인 교차 검증으로 의사결정 보강
3. **Risk Monitor**: 실시간 리스크 모니터링
4. **Stop-Loss Triggers**: 자동 손절매 시스템
5. **Pattern Learning**: 과거 패턴 학습을 통한 개선
6. **Post-Trade Analysis**: 거래 후 분석 및 학습

## 데이터베이스 구조

로컬 SQLite 데이터베이스를 사용합니다.

- `trading_data.db`: 거래 기록, AI 분석 결과, 뉴스 저장 (`DataStore`, `AILearningSystem`)
- `pattern_learning.db`: 학습된 패턴 및 교훈 저장 (`PatternLearner`)

## 에러 처리

- 연속 에러 5회(`MAX_CONSECUTIVE_ERRORS`) 초과 시 시스템 정지
- 각 에러 발생 시 대기 시간 증가 (최대 3600초)
- Circuit breaker를 통한 거래 제한

## 라이선스 (License)

이 프로젝트는 [MIT License](LICENSE)로 배포됩니다.
