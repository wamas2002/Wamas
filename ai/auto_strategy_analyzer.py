"""
Auto Strategy Analyzer
Continuously analyzes real OKX market data and recommends optimal strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from config import Config

@dataclass
class MarketAnalysis:
    """Market analysis result for a trading pair"""
    symbol: str
    timestamp: datetime
    atr: float
    volume_ratio: float
    volatility: float
    trend_strength: float
    trend_direction: str
    support_levels: List[float]
    resistance_levels: List[float]
    market_regime: str
    recommended_strategy: str
    confidence: float
    risk_score: float

@dataclass
class StrategyRecommendation:
    """Strategy recommendation with reasoning"""
    symbol: str
    recommended_strategy: str
    current_strategy: Optional[str]
    confidence: float
    reasoning: str
    market_conditions: Dict[str, float]
    switch_recommended: bool
    priority: str  # 'low', 'medium', 'high'

class AutoStrategyAnalyzer:
    """Auto strategy analyzer with real-time OKX data integration"""
    
    def __init__(self, db_path: str = "data/strategy_analysis.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Strategy mapping based on market conditions
        self.strategy_mapping = {
            'trending_bullish': {
                'primary': 'breakout',
                'secondary': 'trailing_atr',
                'confidence_threshold': 0.7
            },
            'trending_bearish': {
                'primary': 'dca',
                'secondary': 'grid',
                'confidence_threshold': 0.6
            },
            'ranging': {
                'primary': 'grid',
                'secondary': 'ping_pong',
                'confidence_threshold': 0.8
            },
            'volatile': {
                'primary': 'ping_pong',
                'secondary': 'trailing_atr',
                'confidence_threshold': 0.6
            },
            'low_volume': {
                'primary': 'dca',
                'secondary': 'grid',
                'confidence_threshold': 0.5
            }
        }
        
        # Performance tracking for strategy effectiveness
        self.strategy_performance = {
            'dca': {'win_rate': 0.65, 'avg_return': 0.08, 'max_drawdown': 0.12},
            'grid': {'win_rate': 0.72, 'avg_return': 0.06, 'max_drawdown': 0.08},
            'breakout': {'win_rate': 0.58, 'avg_return': 0.15, 'max_drawdown': 0.18},
            'ping_pong': {'win_rate': 0.68, 'avg_return': 0.09, 'max_drawdown': 0.10},
            'trailing_atr': {'win_rate': 0.62, 'avg_return': 0.12, 'max_drawdown': 0.15}
        }
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database for storing analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_analysis (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    atr REAL,
                    volume_ratio REAL,
                    volatility REAL,
                    trend_strength REAL,
                    trend_direction TEXT,
                    market_regime TEXT,
                    recommended_strategy TEXT,
                    confidence REAL,
                    risk_score REAL,
                    analysis_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_recommendations (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    recommended_strategy TEXT,
                    current_strategy TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    switch_recommended BOOLEAN,
                    priority TEXT,
                    market_conditions TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance_tracking (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    start_time DATETIME,
                    end_time DATETIME,
                    return_pct REAL,
                    max_drawdown REAL,
                    trade_count INTEGER,
                    win_rate REAL
                )
            """)
    
    def analyze_market_conditions(self, okx_data_service, symbol: str) -> MarketAnalysis:
        """Analyze current market conditions for a symbol using real OKX data"""
        try:
            # Get comprehensive market data
            hourly_data = okx_data_service.get_historical_data(symbol, '1h', limit=168)  # 7 days
            daily_data = okx_data_service.get_historical_data(symbol, '1d', limit=30)   # 30 days
            
            if hourly_data.empty or daily_data.empty:
                raise ValueError(f"No market data available for {symbol}")
            
            # Calculate technical indicators
            atr = self._calculate_atr(hourly_data)
            volume_ratio = self._calculate_volume_ratio(hourly_data)
            volatility = self._calculate_volatility(hourly_data)
            trend_strength = self._calculate_trend_strength(daily_data)
            trend_direction = self._determine_trend_direction(daily_data)
            
            # Identify support and resistance levels
            support_levels, resistance_levels = self._identify_key_levels(daily_data)
            
            # Determine market regime
            market_regime = self._classify_market_regime(
                atr, volume_ratio, volatility, trend_strength, trend_direction
            )
            
            # Get strategy recommendation
            recommended_strategy, confidence = self._recommend_strategy(
                market_regime, volatility, volume_ratio, trend_strength
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(volatility, atr, volume_ratio)
            
            analysis = MarketAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                atr=atr,
                volume_ratio=volume_ratio,
                volatility=volatility,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                market_regime=market_regime,
                recommended_strategy=recommended_strategy,
                confidence=confidence,
                risk_score=risk_score
            )
            
            # Store analysis in database
            self._store_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions for {symbol}: {str(e)}")
            raise
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return float(atr)
    
    def _calculate_volume_ratio(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate current volume ratio vs average"""
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=period).mean().iloc[-1]
        
        return float(current_volume / avg_volume) if avg_volume > 0 else 1.0
    
    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=period).std().iloc[-1]
        
        return float(volatility * np.sqrt(24))  # Annualized volatility
    
    def _calculate_trend_strength(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate trend strength using ADX-like calculation"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate directional movement
        dm_plus = np.where(
            (high.diff() > low.diff().abs()) & (high.diff() > 0),
            high.diff(),
            0
        )
        dm_minus = np.where(
            (low.diff().abs() > high.diff()) & (low.diff() < 0),
            low.diff().abs(),
            0
        )
        
        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        di_plus = (pd.Series(dm_plus).rolling(window=period).mean() / atr) * 100
        di_minus = (pd.Series(dm_minus).rolling(window=period).mean() / atr) * 100
        
        # Calculate DX and ADX
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = dx.rolling(window=period).mean().iloc[-1]
        
        return float(adx) / 100.0  # Normalize to 0-1
    
    def _determine_trend_direction(self, data: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        # Use multiple timeframe analysis
        close = data['close']
        
        # Short-term trend (5-day EMA vs 10-day EMA)
        ema_5 = close.ewm(span=5).mean()
        ema_10 = close.ewm(span=10).mean()
        short_trend = ema_5.iloc[-1] > ema_10.iloc[-1]
        
        # Medium-term trend (10-day EMA vs 20-day EMA)
        ema_20 = close.ewm(span=20).mean()
        medium_trend = ema_10.iloc[-1] > ema_20.iloc[-1]
        
        # Long-term trend (price vs 30-day SMA)
        sma_30 = close.rolling(window=30).mean()
        long_trend = close.iloc[-1] > sma_30.iloc[-1]
        
        # Combine trends
        bullish_signals = sum([short_trend, medium_trend, long_trend])
        
        if bullish_signals >= 2:
            return 'bullish'
        elif bullish_signals <= 1:
            return 'bearish'
        else:
            return 'neutral'
    
    def _identify_key_levels(self, data: pd.DataFrame, lookback: int = 20) -> Tuple[List[float], List[float]]:
        """Identify key support and resistance levels"""
        high = data['high']
        low = data['low']
        
        # Find local highs and lows
        resistance_levels = []
        support_levels = []
        
        for i in range(lookback, len(data) - lookback):
            # Check for resistance (local high)
            if high.iloc[i] == high.iloc[i-lookback:i+lookback+1].max():
                resistance_levels.append(float(high.iloc[i]))
            
            # Check for support (local low)
            if low.iloc[i] == low.iloc[i-lookback:i+lookback+1].min():
                support_levels.append(float(low.iloc[i]))
        
        # Return most recent and significant levels
        resistance_levels = sorted(list(set(resistance_levels)))[-3:]  # Top 3
        support_levels = sorted(list(set(support_levels)), reverse=True)[-3:]  # Bottom 3
        
        return support_levels, resistance_levels
    
    def _classify_market_regime(self, atr: float, volume_ratio: float, volatility: float, 
                              trend_strength: float, trend_direction: str) -> str:
        """Classify the current market regime"""
        
        # Normalize values for comparison
        high_volatility = volatility > 0.03  # 3% daily volatility
        high_volume = volume_ratio > 1.5
        strong_trend = trend_strength > 0.25
        
        if strong_trend and trend_direction == 'bullish':
            return 'trending_bullish'
        elif strong_trend and trend_direction == 'bearish':
            return 'trending_bearish'
        elif high_volatility and not strong_trend:
            return 'volatile'
        elif not high_volatility and not strong_trend:
            return 'ranging'
        elif volume_ratio < 0.7:
            return 'low_volume'
        else:
            return 'ranging'  # Default
    
    def _recommend_strategy(self, market_regime: str, volatility: float, 
                          volume_ratio: float, trend_strength: float) -> Tuple[str, float]:
        """Recommend optimal strategy based on market conditions"""
        
        if market_regime not in self.strategy_mapping:
            return 'grid', 0.5  # Default fallback
        
        strategy_config = self.strategy_mapping[market_regime]
        primary_strategy = strategy_config['primary']
        confidence_threshold = strategy_config['confidence_threshold']
        
        # Calculate confidence based on how well conditions match
        confidence = confidence_threshold
        
        # Adjust confidence based on market clarity
        if trend_strength > 0.3:
            confidence += 0.1
        if volatility < 0.02:  # Low volatility increases confidence for ranging strategies
            if market_regime in ['ranging', 'low_volume']:
                confidence += 0.1
        if volume_ratio > 2.0:  # High volume increases confidence
            confidence += 0.05
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        return primary_strategy, confidence
    
    def _calculate_risk_score(self, volatility: float, atr: float, volume_ratio: float) -> float:
        """Calculate risk score (0-1, higher = riskier)"""
        
        # Volatility component (0-0.4)
        vol_score = min(volatility * 10, 0.4)
        
        # ATR component (0-0.3)
        atr_score = min(atr / 1000, 0.3)  # Normalize ATR
        
        # Volume component (0-0.3)
        vol_ratio_score = min(abs(volume_ratio - 1) * 0.3, 0.3)
        
        risk_score = vol_score + atr_score + vol_ratio_score
        return min(risk_score, 1.0)
    
    def generate_strategy_recommendations(self, okx_data_service, 
                                        current_strategies: Dict[str, str]) -> List[StrategyRecommendation]:
        """Generate strategy recommendations for all symbols"""
        recommendations = []
        
        for symbol in Config.SUPPORTED_SYMBOLS:
            try:
                # Analyze current market conditions
                analysis = self.analyze_market_conditions(okx_data_service, symbol)
                
                # Get current strategy
                current_strategy = current_strategies.get(symbol)
                
                # Determine if strategy switch is recommended
                switch_recommended = False
                priority = 'low'
                reasoning = f"Market regime: {analysis.market_regime}"
                
                if current_strategy != analysis.recommended_strategy:
                    # Check if switch is worth it based on confidence and performance
                    current_perf = self.strategy_performance.get(current_strategy, {})
                    recommended_perf = self.strategy_performance.get(analysis.recommended_strategy, {})
                    
                    if analysis.confidence > 0.7:
                        if (recommended_perf.get('win_rate', 0) > current_perf.get('win_rate', 0) or
                            recommended_perf.get('avg_return', 0) > current_perf.get('avg_return', 0)):
                            switch_recommended = True
                            priority = 'high' if analysis.confidence > 0.8 else 'medium'
                            reasoning += f". Recommended strategy shows better performance (WR: {recommended_perf.get('win_rate', 0):.1%})"
                
                recommendation = StrategyRecommendation(
                    symbol=symbol,
                    recommended_strategy=analysis.recommended_strategy,
                    current_strategy=current_strategy,
                    confidence=analysis.confidence,
                    reasoning=reasoning,
                    market_conditions={
                        'volatility': analysis.volatility,
                        'volume_ratio': analysis.volume_ratio,
                        'trend_strength': analysis.trend_strength,
                        'risk_score': analysis.risk_score
                    },
                    switch_recommended=switch_recommended,
                    priority=priority
                )
                
                recommendations.append(recommendation)
                
                # Store recommendation
                self._store_recommendation(recommendation)
                
            except Exception as e:
                self.logger.error(f"Error generating recommendation for {symbol}: {str(e)}")
                continue
        
        return recommendations
    
    def _store_analysis(self, analysis: MarketAnalysis):
        """Store market analysis in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO market_analysis (
                    symbol, timestamp, atr, volume_ratio, volatility, trend_strength,
                    trend_direction, market_regime, recommended_strategy, confidence,
                    risk_score, analysis_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.symbol,
                analysis.timestamp,
                analysis.atr,
                analysis.volume_ratio,
                analysis.volatility,
                analysis.trend_strength,
                analysis.trend_direction,
                analysis.market_regime,
                analysis.recommended_strategy,
                analysis.confidence,
                analysis.risk_score,
                json.dumps({
                    'support_levels': analysis.support_levels,
                    'resistance_levels': analysis.resistance_levels
                })
            ))
    
    def _store_recommendation(self, recommendation: StrategyRecommendation):
        """Store strategy recommendation in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO strategy_recommendations (
                    symbol, timestamp, recommended_strategy, current_strategy,
                    confidence, reasoning, switch_recommended, priority, market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation.symbol,
                datetime.now(),
                recommendation.recommended_strategy,
                recommendation.current_strategy,
                recommendation.confidence,
                recommendation.reasoning,
                recommendation.switch_recommended,
                recommendation.priority,
                json.dumps(recommendation.market_conditions)
            ))
    
    def get_analysis_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get analysis history for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM market_analysis 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            cutoff_date = datetime.now() - timedelta(days=days)
            return pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
    
    def get_recommendation_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get recommendation history for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM strategy_recommendations 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            cutoff_date = datetime.now() - timedelta(days=days)
            return pd.read_sql_query(query, conn, params=(symbol, cutoff_date))
    
    def update_strategy_performance(self, symbol: str, strategy: str, 
                                  return_pct: float, max_drawdown: float,
                                  trade_count: int, win_rate: float):
        """Update strategy performance tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO strategy_performance_tracking (
                    symbol, strategy, start_time, end_time, return_pct,
                    max_drawdown, trade_count, win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, strategy, datetime.now() - timedelta(hours=6),
                datetime.now(), return_pct, max_drawdown, trade_count, win_rate
            ))
    
    def get_strategy_effectiveness_report(self) -> Dict[str, Any]:
        """Get comprehensive strategy effectiveness report"""
        with sqlite3.connect(self.db_path) as conn:
            # Get strategy performance stats
            perf_query = """
                SELECT strategy, AVG(return_pct) as avg_return, 
                       AVG(win_rate) as avg_win_rate, AVG(max_drawdown) as avg_drawdown,
                       COUNT(*) as sample_count
                FROM strategy_performance_tracking 
                WHERE start_time >= ?
                GROUP BY strategy
            """
            cutoff_date = datetime.now() - timedelta(days=30)
            perf_df = pd.read_sql_query(perf_query, conn, params=(cutoff_date,))
            
            # Get recommendation accuracy
            rec_query = """
                SELECT recommended_strategy, AVG(confidence) as avg_confidence,
                       COUNT(*) as recommendation_count
                FROM strategy_recommendations 
                WHERE timestamp >= ?
                GROUP BY recommended_strategy
            """
            rec_df = pd.read_sql_query(rec_query, conn, params=(cutoff_date,))
            
            return {
                'performance_stats': perf_df.to_dict('records'),
                'recommendation_stats': rec_df.to_dict('records'),
                'total_analyses': len(self.get_analysis_history('', days=30)),
                'last_updated': datetime.now().isoformat()
            }