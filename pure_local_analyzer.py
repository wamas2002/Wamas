
"""
Pure Local Trading Analysis
Advanced local analysis without any external API dependencies
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class PureLocalAnalyzer:
    def __init__(self):
        self.analysis_weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volume': 0.2,
            'volatility': 0.15,
            'support_resistance': 0.1
        }
    
    def analyze_market_conditions(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Comprehensive local market analysis"""
        try:
            latest = df.iloc[-1]
            
            # Trend analysis
            trend_score = self._analyze_trend(df)
            
            # Momentum analysis
            momentum_score = self._analyze_momentum(df)
            
            # Volume analysis
            volume_score = self._analyze_volume(df)
            
            # Volatility analysis
            volatility_score = self._analyze_volatility(df)
            
            # Support/Resistance analysis
            sr_score = self._analyze_support_resistance(df)
            
            # Composite score
            composite_score = (
                trend_score * self.analysis_weights['trend'] +
                momentum_score * self.analysis_weights['momentum'] +
                volume_score * self.analysis_weights['volume'] +
                volatility_score * self.analysis_weights['volatility'] +
                sr_score * self.analysis_weights['support_resistance']
            )
            
            # Risk assessment
            risk_level = self._assess_risk(df, composite_score)
            
            return {
                'symbol': symbol,
                'composite_score': round(composite_score, 2),
                'trend_score': round(trend_score, 2),
                'momentum_score': round(momentum_score, 2),
                'volume_score': round(volume_score, 2),
                'volatility_score': round(volatility_score, 2),
                'sr_score': round(sr_score, 2),
                'risk_level': risk_level,
                'signal': 'BUY' if composite_score > 60 else 'SELL' if composite_score < 40 else 'HOLD'
            }
            
        except Exception as e:
            logger.error(f"Local analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'composite_score': 50.0,
                'risk_level': 'medium',
                'signal': 'HOLD'
            }
    
    def _analyze_trend(self, df: pd.DataFrame) -> float:
        """Analyze price trend strength"""
        try:
            close_prices = df['close']
            
            # Multiple timeframe EMAs
            ema_9 = close_prices.ewm(span=9).mean()
            ema_21 = close_prices.ewm(span=21).mean()
            ema_50 = close_prices.ewm(span=50).mean()
            
            current_price = close_prices.iloc[-1]
            
            score = 50.0
            
            # EMA alignment
            if current_price > ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1]:
                score += 20  # Strong uptrend
            elif current_price > ema_9.iloc[-1] > ema_21.iloc[-1]:
                score += 15  # Moderate uptrend
            elif current_price > ema_9.iloc[-1]:
                score += 10  # Weak uptrend
            
            # Price momentum over different periods
            price_change_5 = (current_price - close_prices.iloc[-6]) / close_prices.iloc[-6] * 100
            price_change_10 = (current_price - close_prices.iloc[-11]) / close_prices.iloc[-11] * 100
            
            if price_change_5 > 2:
                score += 10
            elif price_change_5 > 1:
                score += 5
            
            if price_change_10 > 5:
                score += 10
            elif price_change_10 > 2:
                score += 5
            
            return min(score, 95.0)
            
        except:
            return 50.0
    
    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """Analyze momentum indicators"""
        try:
            close_prices = df['close']
            high_prices = df['high']
            low_prices = df['low']
            
            score = 50.0
            
            # RSI analysis
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.01)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            if 30 < current_rsi < 70:
                score += 10  # Healthy range
            elif current_rsi < 30:
                score += 15  # Oversold bounce potential
            elif current_rsi > 70:
                score += 5   # Overbought but can continue
            
            # MACD analysis
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            
            if macd_line.iloc[-1] > macd_signal.iloc[-1]:
                score += 15
            
            # Stochastic analysis
            lowest_low = low_prices.rolling(window=14).min()
            highest_high = high_prices.rolling(window=14).max()
            k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
            
            current_stoch = k_percent.iloc[-1]
            if 20 < current_stoch < 80:
                score += 10
            elif current_stoch < 20:
                score += 15  # Oversold
            
            return min(score, 95.0)
            
        except:
            return 50.0
    
    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """Analyze volume patterns"""
        try:
            volume = df['volume']
            close_prices = df['close']
            
            score = 50.0
            
            # Volume trend
            volume_sma = volume.rolling(window=20).mean()
            current_volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1]
            
            if current_volume_ratio > 1.5:
                score += 20  # High volume
            elif current_volume_ratio > 1.2:
                score += 15
            elif current_volume_ratio > 1.0:
                score += 10
            
            # Price-volume relationship
            price_change = (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]
            volume_change = (volume.iloc[-1] - volume.iloc[-2]) / volume.iloc[-2]
            
            if price_change > 0 and volume_change > 0:
                score += 10  # Price up on volume
            
            return min(score, 95.0)
            
        except:
            return 50.0
    
    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        """Analyze volatility patterns"""
        try:
            close_prices = df['close']
            high_prices = df['high']
            low_prices = df['low']
            
            score = 50.0
            
            # Average True Range
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - close_prices.shift())
            tr3 = abs(low_prices - close_prices.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # Bollinger Bands
            sma_20 = close_prices.rolling(window=20).mean()
            std_20 = close_prices.rolling(window=20).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
            
            current_price = close_prices.iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Optimal volatility range
            if 0.2 < bb_position < 0.8:
                score += 15  # Good range
            elif bb_position < 0.2:
                score += 20  # Near lower band - potential bounce
            elif bb_position > 0.8:
                score += 10  # Near upper band
            
            return min(score, 95.0)
            
        except:
            return 50.0
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> float:
        """Analyze support and resistance levels"""
        try:
            close_prices = df['close']
            high_prices = df['high']
            low_prices = df['low']
            
            score = 50.0
            current_price = close_prices.iloc[-1]
            
            # Pivot points
            recent_highs = high_prices.rolling(window=5).max()
            recent_lows = low_prices.rolling(window=5).min()
            
            # Distance from recent high/low
            distance_from_high = (recent_highs.iloc[-1] - current_price) / current_price
            distance_from_low = (current_price - recent_lows.iloc[-1]) / current_price
            
            # Near support levels
            if distance_from_low < 0.02:  # Within 2% of recent low
                score += 15
            elif distance_from_low < 0.05:  # Within 5% of recent low
                score += 10
            
            # Below resistance
            if distance_from_high > 0.02:  # More than 2% below recent high
                score += 10
            
            return min(score, 95.0)
            
        except:
            return 50.0
    
    def _assess_risk(self, df: pd.DataFrame, score: float) -> str:
        """Assess risk level based on analysis"""
        try:
            close_prices = df['close']
            
            # Volatility check
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 365)  # Annualized volatility
            
            if volatility > 2.0:  # High volatility
                return "high"
            elif volatility > 1.0:  # Medium volatility
                return "medium"
            else:
                return "low"
                
        except:
            return "medium"

# Global instance for use across the system
local_analyzer = PureLocalAnalyzer()
