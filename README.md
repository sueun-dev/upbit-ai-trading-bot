# AI Upbit Trading System

AI 기반 업비트 암호화폐 자동 거래 시스템

## 시스템 전체 로직 플로우

### 1. 메인 프로그램 시작 (main.py)

#### 1.1 초기화 단계
```python
# 로깅 설정
setup_logging()

# AI 학습 시스템 초기화
trade_analyzer = AILearningSystem()

# 과거 거래 분석 (30일)
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
    openai_api_key=OPENAI_API_KEY,
    trade_analyzer=trade_analyzer
)
```

#### 1.3 메인 거래 루프 실행
```python
run_main_trading_loop(orchestrator)
# CHECK_INTERVAL_SECONDS(60초)마다 거래 사이클 실행
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
# Return 예시:
# [
#   {
#     "title": "비트코인 급등, 7만 달러 돌파",
#     "summary": "비트코인이 기관 투자자들의 매수세로 급등...",
#     "source": "coinness",
#     "url": "https://example.com/news/123"
#   },
#   {
#     "title": "이더리움 업그레이드 완료",
#     "summary": "이더리움 덴쿤 업그레이드가 성공적으로...",
#     "source": "tokenpost",
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

##### 2.2.4 멀티 AI 검증
```python
validated_decisions = self.multi_ai_validator.cross_validate_multiple_decisions(
    raw_decisions, market_data, portfolio, news_list
)
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

## 실행 방법

### 1. 메인 프로그램 시작 (main.py)

Poetry를 사용하는 경우:
```bash
poetry run python3 main.py
```

또는 직접 실행:
```bash
python3 main.py
```

## 주요 변수 및 상수

### main.py
- `MAX_CONSECUTIVE_ERRORS = 5` - 최대 연속 에러 허용 횟수
- `AI_LEARNING_DAYS_BACK = 30` - AI 학습 분석 기간 (일)
- `CHECK_INTERVAL_SECONDS = 60` - 거래 사이클 실행 간격 (초)

### trading_orchestrator.py
- `DEFAULT_BUY_AMOUNT_KRW = 30_000` - 기본 매수 금액
- `MIN_VIABLE_TRADE_AMOUNT = 10000` - 최소 거래 금액
- `STOP_LOSS_CONFIDENCE = 0.9` - 손절매 신뢰도
- `MAX_NEWS_ARTICLES = 15` - 최대 뉴스 수집 개수

### 거래 액션 타입
- `ACTION_BUY = 'buy'` - 신규 매수
- `ACTION_BUY_MORE = 'buy_more'` - 추가 매수 (물타기)
- `ACTION_SELL_ALL = 'sell_all'` - 전량 매도
- `ACTION_PARTIAL_SELL = 'partial_sell'` - 부분 매도
- `ACTION_HOLD = 'hold'` - 보유 유지

## 시스템 안전장치

1. **Circuit Breaker**: 과도한 거래 방지
2. **Multi-AI Validation**: 복수 AI 모델을 통한 교차 검증
3. **Risk Monitor**: 실시간 리스크 모니터링
4. **Stop-Loss Triggers**: 자동 손절매 시스템
5. **Pattern Learning**: 과거 패턴 학습을 통한 개선
6. **Post-Trade Analysis**: 거래 후 분석 및 학습

## 데이터베이스 구조

- `trading_system.db`: 거래 기록, AI 분석 결과, 뉴스 저장
- `pattern_learning.db`: 학습된 패턴 및 교훈 저장

## 에러 처리

- 연속 에러 5회 초과 시 시스템 정지
- 각 에러 발생 시 대기 시간 증가 (최대 3600초)
- Circuit breaker를 통한 거래 제한
EOF < /dev/null