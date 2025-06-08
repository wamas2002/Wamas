import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class MarketSentimentAnalyzer:
    """Advanced market sentiment analysis combining multiple indicators"""
    
    def __init__(self):
        self.sentiment_history = []
        self.fear_greed_cache = {}
        self.social_sentiment_cache = {}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def analyze_market_sentiment(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive market sentiment analysis"""
        try:
            sentiment_data = {}
            
            # Technical sentiment indicators
            technical_sentiment = self._calculate_technical_sentiment(price_data)
            sentiment_data['technical'] = technical_sentiment
            
            # Volume-based sentiment
            volume_sentiment = self._calculate_volume_sentiment(price_data)
            sentiment_data['volume'] = volume_sentiment
            
            # Price action sentiment
            price_sentiment = self._calculate_price_action_sentiment(price_data)
            sentiment_data['price_action'] = price_sentiment
            
            # Market structure sentiment
            structure_sentiment = self._calculate_market_structure_sentiment(price_data)
            sentiment_data['market_structure'] = structure_sentiment
            
            # Volatility sentiment
            volatility_sentiment = self._calculate_volatility_sentiment(price_data)
            sentiment_data['volatility'] = volatility_sentiment
            
            # Combined sentiment score
            combined_sentiment = self._calculate_combined_sentiment(sentiment_data)
            sentiment_data['combined'] = combined_sentiment
            
            # Generate trading signals based on sentiment
            signals = self._generate_sentiment_signals(sentiment_data)
            
            return {
                'success': True,
                'sentiment_data': sentiment_data,
                'signals': signals,
                'sentiment_score': combined_sentiment['overall_score'],
                'sentiment_label': combined_sentiment['sentiment_label'],
                'confidence': combined_sentiment['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'sentiment_score': 0.5,
                'sentiment_label': 'Neutral'
            }
    
    def _calculate_technical_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment from technical indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
            
            # RSI sentiment
            rsi = self._calculate_rsi(close)
            rsi_sentiment = self._normalize_rsi_sentiment(rsi[-1] if len(rsi) > 0 else 50)
            
            # MACD sentiment
            macd_line, macd_signal = self._calculate_macd(close)
            macd_sentiment = 1.0 if macd_line[-1] > macd_signal[-1] else 0.0
            
            # Bollinger Bands sentiment
            bb_sentiment = self._calculate_bb_sentiment(close)
            
            # Moving average sentiment
            ma_sentiment = self._calculate_ma_sentiment(close)
            
            # Stochastic sentiment
            stoch_sentiment = self._calculate_stochastic_sentiment(high, low, close)
            
            return {
                'rsi_sentiment': rsi_sentiment,
                'macd_sentiment': macd_sentiment,
                'bollinger_sentiment': bb_sentiment,
                'ma_sentiment': ma_sentiment,
                'stochastic_sentiment': stoch_sentiment,
                'overall': np.mean([rsi_sentiment, macd_sentiment, bb_sentiment, ma_sentiment, stoch_sentiment])
            }
            
        except Exception as e:
            return {'overall': 0.5, 'error': str(e)}
    
    def _calculate_volume_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment from volume patterns"""
        try:
            if 'volume' not in df.columns:
                return {'overall': 0.5, 'note': 'Volume data not available'}
            
            volume = df['volume'].values
            close = df['close'].values
            
            # Volume trend
            volume_ma = np.convolve(volume, np.ones(20)/20, mode='valid')
            volume_trend = 1.0 if volume[-1] > volume_ma[-1] else 0.0
            
            # Price-volume correlation
            price_change = np.diff(close[-20:])
            volume_change = np.diff(volume[-20:])
            correlation = np.corrcoef(price_change, volume_change)[0, 1] if len(price_change) > 1 else 0
            volume_correlation_sentiment = (correlation + 1) / 2  # Normalize to 0-1
            
            # Volume spike detection
            volume_std = np.std(volume[-50:]) if len(volume) >= 50 else np.std(volume)
            volume_spike = 1.0 if volume[-1] > (np.mean(volume[-20:]) + 2 * volume_std) else 0.0
            
            return {
                'volume_trend': volume_trend,
                'price_volume_correlation': volume_correlation_sentiment,
                'volume_spike': volume_spike,
                'overall': np.mean([volume_trend, volume_correlation_sentiment, volume_spike])
            }
            
        except Exception as e:
            return {'overall': 0.5, 'error': str(e)}
    
    def _calculate_price_action_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment from price action patterns"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Trend strength
            trend_strength = self._calculate_trend_strength(close)
            
            # Support/resistance sentiment
            sr_sentiment = self._calculate_support_resistance_sentiment(close, high, low)
            
            # Breakout sentiment
            breakout_sentiment = self._calculate_breakout_sentiment(close, high, low)
            
            # Momentum sentiment
            momentum_sentiment = self._calculate_momentum_sentiment(close)
            
            # Candlestick pattern sentiment
            candle_sentiment = self._calculate_candlestick_sentiment(df)
            
            return {
                'trend_strength': trend_strength,
                'support_resistance': sr_sentiment,
                'breakout': breakout_sentiment,
                'momentum': momentum_sentiment,
                'candlestick_patterns': candle_sentiment,
                'overall': np.mean([trend_strength, sr_sentiment, breakout_sentiment, momentum_sentiment, candle_sentiment])
            }
            
        except Exception as e:
            return {'overall': 0.5, 'error': str(e)}
    
    def _calculate_market_structure_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment from market structure"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Higher highs and higher lows
            hh_hl_sentiment = self._calculate_hh_hl_sentiment(high, low)
            
            # Market efficiency
            efficiency_sentiment = self._calculate_market_efficiency_sentiment(close)
            
            # Fractal analysis
            fractal_sentiment = self._calculate_fractal_sentiment(close)
            
            return {
                'higher_highs_lows': hh_hl_sentiment,
                'market_efficiency': efficiency_sentiment,
                'fractal_structure': fractal_sentiment,
                'overall': np.mean([hh_hl_sentiment, efficiency_sentiment, fractal_sentiment])
            }
            
        except Exception as e:
            return {'overall': 0.5, 'error': str(e)}
    
    def _calculate_volatility_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment from volatility patterns"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # ATR-based volatility sentiment
            atr = self._calculate_atr(high, low, close)
            volatility_trend = self._calculate_volatility_trend_sentiment(atr)
            
            # Volatility clustering
            clustering_sentiment = self._calculate_volatility_clustering_sentiment(close)
            
            # Volatility breakout
            breakout_vol_sentiment = self._calculate_volatility_breakout_sentiment(atr)
            
            return {
                'volatility_trend': volatility_trend,
                'volatility_clustering': clustering_sentiment,
                'volatility_breakout': breakout_vol_sentiment,
                'overall': np.mean([volatility_trend, clustering_sentiment, breakout_vol_sentiment])
            }
            
        except Exception as e:
            return {'overall': 0.5, 'error': str(e)}
    
    def _calculate_combined_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined sentiment score"""
        try:
            # Extract overall scores
            scores = []
            weights = []
            
            if 'technical' in sentiment_data and 'overall' in sentiment_data['technical']:
                scores.append(sentiment_data['technical']['overall'])
                weights.append(0.25)
            
            if 'volume' in sentiment_data and 'overall' in sentiment_data['volume']:
                scores.append(sentiment_data['volume']['overall'])
                weights.append(0.20)
            
            if 'price_action' in sentiment_data and 'overall' in sentiment_data['price_action']:
                scores.append(sentiment_data['price_action']['overall'])
                weights.append(0.25)
            
            if 'market_structure' in sentiment_data and 'overall' in sentiment_data['market_structure']:
                scores.append(sentiment_data['market_structure']['overall'])
                weights.append(0.15)
            
            if 'volatility' in sentiment_data and 'overall' in sentiment_data['volatility']:
                scores.append(sentiment_data['volatility']['overall'])
                weights.append(0.15)
            
            # Calculate weighted average
            if scores and weights:
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                overall_score = np.average(scores, weights=weights)
            else:
                overall_score = 0.5
            
            # Determine sentiment label
            if overall_score >= 0.7:
                sentiment_label = "Very Bullish"
                confidence = min(0.95, overall_score + 0.1)
            elif overall_score >= 0.6:
                sentiment_label = "Bullish"
                confidence = overall_score
            elif overall_score >= 0.4:
                sentiment_label = "Neutral"
                confidence = 1.0 - abs(overall_score - 0.5) * 2
            elif overall_score >= 0.3:
                sentiment_label = "Bearish"
                confidence = 1.0 - overall_score
            else:
                sentiment_label = "Very Bearish"
                confidence = min(0.95, 1.0 - overall_score + 0.1)
            
            return {
                'overall_score': overall_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'component_scores': scores,
                'component_weights': weights.tolist() if len(weights) > 0 else []
            }
            
        except Exception as e:
            return {
                'overall_score': 0.5,
                'sentiment_label': 'Neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_sentiment_signals(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on sentiment analysis"""
        try:
            combined = sentiment_data.get('combined', {})
            overall_score = combined.get('overall_score', 0.5)
            confidence = combined.get('confidence', 0.0)
            
            # Signal generation logic
            if overall_score >= 0.7 and confidence >= 0.6:
                signal = "STRONG_BUY"
                strength = min(1.0, overall_score * confidence)
            elif overall_score >= 0.6 and confidence >= 0.5:
                signal = "BUY"
                strength = overall_score * confidence
            elif overall_score <= 0.3 and confidence >= 0.6:
                signal = "STRONG_SELL"
                strength = min(1.0, (1.0 - overall_score) * confidence)
            elif overall_score <= 0.4 and confidence >= 0.5:
                signal = "SELL"
                strength = (1.0 - overall_score) * confidence
            else:
                signal = "HOLD"
                strength = confidence
            
            # Risk assessment
            risk_level = self._assess_sentiment_risk(sentiment_data)
            
            return {
                'signal': signal,
                'strength': strength,
                'risk_level': risk_level,
                'recommendation': self._generate_recommendation(signal, strength, risk_level)
            }
            
        except Exception as e:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'risk_level': 'HIGH',
                'error': str(e)
            }
    
    def _assess_sentiment_risk(self, sentiment_data: Dict[str, Any]) -> str:
        """Assess risk level based on sentiment analysis"""
        try:
            volatility_score = sentiment_data.get('volatility', {}).get('overall', 0.5)
            volume_score = sentiment_data.get('volume', {}).get('overall', 0.5)
            structure_score = sentiment_data.get('market_structure', {}).get('overall', 0.5)
            
            # High volatility increases risk
            risk_score = volatility_score * 0.4
            
            # Low volume increases risk
            risk_score += (1.0 - volume_score) * 0.3
            
            # Poor market structure increases risk
            risk_score += (1.0 - structure_score) * 0.3
            
            if risk_score >= 0.7:
                return "HIGH"
            elif risk_score >= 0.5:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "MEDIUM"
    
    def _generate_recommendation(self, signal: str, strength: float, risk_level: str) -> str:
        """Generate trading recommendation"""
        if signal in ["STRONG_BUY", "BUY"] and risk_level == "LOW":
            return f"Consider {signal.lower().replace('_', ' ')} position with tight risk management"
        elif signal in ["STRONG_SELL", "SELL"] and risk_level == "LOW":
            return f"Consider {signal.lower().replace('_', ' ')} position with profit targets"
        elif signal == "HOLD":
            return "Wait for clearer signals before entering positions"
        else:
            return f"Sentiment suggests {signal.lower().replace('_', ' ')} but exercise caution due to {risk_level.lower()} risk"
    
    # Helper calculation methods
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _normalize_rsi_sentiment(self, rsi_value: float) -> float:
        """Convert RSI to sentiment score"""
        if rsi_value >= 70:
            return 0.8  # Overbought but still bullish
        elif rsi_value >= 50:
            return 0.5 + (rsi_value - 50) / 40  # 0.5 to 1.0
        elif rsi_value >= 30:
            return rsi_value / 60  # 0.5 to 0.0
        else:
            return 0.2  # Oversold but still bearish
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = self._ema(macd_line, signal)
        return macd_line, macd_signal
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _calculate_bb_sentiment(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> float:
        """Calculate Bollinger Bands sentiment"""
        ma = np.convolve(prices, np.ones(period)/period, mode='valid')
        std = np.array([np.std(prices[i:i+period]) for i in range(len(prices)-period+1)])
        
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        current_price = prices[-1]
        current_upper = upper_band[-1]
        current_lower = lower_band[-1]
        
        # Normalize position within bands to sentiment score
        position = (current_price - current_lower) / (current_upper - current_lower)
        return np.clip(position, 0, 1)
    
    def _calculate_ma_sentiment(self, prices: np.ndarray) -> float:
        """Calculate moving average sentiment"""
        ma_short = np.mean(prices[-10:])
        ma_long = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        
        return 1.0 if ma_short > ma_long else 0.0
    
    def _calculate_stochastic_sentiment(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Stochastic sentiment"""
        if len(high) < period:
            return 0.5
        
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        
        if highest_high == lowest_low:
            return 0.5
        
        k_percent = (close[-1] - lowest_low) / (highest_high - lowest_low)
        return k_percent
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength"""
        if len(prices) < 20:
            return 0.5
        
        # Linear regression slope
        x = np.arange(len(prices[-20:]))
        y = prices[-20:]
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to sentiment score
        price_range = np.max(prices[-20:]) - np.min(prices[-20:])
        normalized_slope = slope / (price_range / 20)
        
        return np.clip((normalized_slope + 1) / 2, 0, 1)
    
    def _calculate_support_resistance_sentiment(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate support/resistance sentiment"""
        # Simplified support/resistance calculation
        recent_highs = high[-20:]
        recent_lows = low[-20:]
        current_price = close[-1]
        
        resistance_level = np.max(recent_highs)
        support_level = np.min(recent_lows)
        
        if resistance_level == support_level:
            return 0.5
        
        position = (current_price - support_level) / (resistance_level - support_level)
        return np.clip(position, 0, 1)
    
    def _calculate_breakout_sentiment(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate breakout sentiment"""
        if len(close) < 50:
            return 0.5
        
        # Check for recent breakouts
        resistance = np.max(high[-50:-10])
        support = np.min(low[-50:-10])
        current_price = close[-1]
        
        if current_price > resistance:
            return 0.9  # Bullish breakout
        elif current_price < support:
            return 0.1  # Bearish breakdown
        else:
            return 0.5  # No breakout
    
    def _calculate_momentum_sentiment(self, prices: np.ndarray) -> float:
        """Calculate momentum sentiment"""
        if len(prices) < 20:
            return 0.5
        
        momentum = (prices[-1] - prices[-20]) / prices[-20]
        return np.clip((momentum + 0.1) / 0.2 + 0.5, 0, 1)
    
    def _calculate_candlestick_sentiment(self, df: pd.DataFrame) -> float:
        """Calculate candlestick pattern sentiment"""
        # Simplified candlestick analysis
        if len(df) < 3:
            return 0.5
        
        recent = df.tail(3)
        open_prices = recent['open'].values
        close_prices = recent['close'].values
        high_prices = recent['high'].values
        low_prices = recent['low'].values
        
        # Check for bullish/bearish patterns
        bullish_signals = 0
        bearish_signals = 0
        
        # Hammer pattern
        for i in range(len(recent)):
            body = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            
            if lower_shadow > 2 * body and upper_shadow < body:
                bullish_signals += 1
            elif upper_shadow > 2 * body and lower_shadow < body:
                bearish_signals += 1
        
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return 0.5
        
        return bullish_signals / total_signals
    
    def _calculate_hh_hl_sentiment(self, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate higher highs and higher lows sentiment"""
        if len(high) < 10:
            return 0.5
        
        recent_highs = high[-10:]
        recent_lows = low[-10:]
        
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
        
        total_periods = len(recent_highs) - 1
        hh_hl_ratio = (hh_count + hl_count) / (2 * total_periods)
        
        return hh_hl_ratio
    
    def _calculate_market_efficiency_sentiment(self, prices: np.ndarray) -> float:
        """Calculate market efficiency sentiment"""
        if len(prices) < 20:
            return 0.5
        
        returns = np.diff(np.log(prices[-20:]))
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        
        # Higher autocorrelation suggests less efficiency (trending market)
        return np.clip((autocorr + 1) / 2, 0, 1)
    
    def _calculate_fractal_sentiment(self, prices: np.ndarray) -> float:
        """Calculate fractal sentiment"""
        # Simplified fractal analysis
        if len(prices) < 20:
            return 0.5
        
        # Calculate Hurst exponent approximation
        returns = np.diff(np.log(prices[-20:]))
        var_returns = np.var(returns)
        
        if var_returns == 0:
            return 0.5
        
        # Simple trending vs mean-reverting measure
        cumulative_sum = np.cumsum(returns - np.mean(returns))
        range_value = np.max(cumulative_sum) - np.min(cumulative_sum)
        
        if range_value == 0:
            return 0.5
        
        hurst_approx = np.log(range_value / np.sqrt(var_returns * len(returns))) / np.log(len(returns))
        return np.clip(hurst_approx, 0, 1)
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        
        return atr
    
    def _calculate_volatility_trend_sentiment(self, atr: np.ndarray) -> float:
        """Calculate volatility trend sentiment"""
        if len(atr) < 10:
            return 0.5
        
        recent_atr = atr[-10:]
        atr_trend = np.polyfit(range(len(recent_atr)), recent_atr, 1)[0]
        
        # Decreasing volatility is generally bullish
        return 1.0 if atr_trend < 0 else 0.0
    
    def _calculate_volatility_clustering_sentiment(self, prices: np.ndarray) -> float:
        """Calculate volatility clustering sentiment"""
        if len(prices) < 20:
            return 0.5
        
        returns = np.diff(np.log(prices[-20:]))
        volatility = np.abs(returns)
        
        # Check for volatility clustering (GARCH effects)
        vol_autocorr = np.corrcoef(volatility[:-1], volatility[1:])[0, 1] if len(volatility) > 1 else 0
        
        # High volatility clustering suggests uncertainty
        return 1.0 - np.clip(vol_autocorr, 0, 1)
    
    def _calculate_volatility_breakout_sentiment(self, atr: np.ndarray) -> float:
        """Calculate volatility breakout sentiment"""
        if len(atr) < 20:
            return 0.5
        
        current_atr = atr[-1]
        avg_atr = np.mean(atr[-20:])
        std_atr = np.std(atr[-20:])
        
        # Volatility breakout detection
        if current_atr > avg_atr + 2 * std_atr:
            return 0.8  # High volatility breakout
        elif current_atr < avg_atr - std_atr:
            return 0.3  # Low volatility
        else:
            return 0.5  # Normal volatility
    
    def get_sentiment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent sentiment analysis history"""
        return self.sentiment_history[-limit:] if self.sentiment_history else []
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get sentiment analysis summary"""
        if not self.sentiment_history:
            return {
                'total_analyses': 0,
                'avg_sentiment': 0.5,
                'sentiment_trend': 'No data'
            }
        
        recent_sentiments = [item.get('sentiment_score', 0.5) for item in self.sentiment_history[-10:]]
        
        return {
            'total_analyses': len(self.sentiment_history),
            'avg_sentiment': np.mean(recent_sentiments),
            'sentiment_trend': 'Bullish' if np.mean(recent_sentiments) > 0.6 else 'Bearish' if np.mean(recent_sentiments) < 0.4 else 'Neutral',
            'last_updated': self.sentiment_history[-1].get('timestamp') if self.sentiment_history else None
        }