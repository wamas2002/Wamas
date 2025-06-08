"""
Daily Top Picks Dashboard
Automatically generates top cryptocurrency recommendations based on AI predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import sqlite3
from trading.okx_data_service import OKXDataService
from ai.advisor import AIFinancialAdvisor
from ai.comprehensive_ml_pipeline import TradingMLPipeline
import warnings
warnings.filterwarnings('ignore')

class DailyTopPicks:
    """Generate daily top cryptocurrency picks based on AI analysis"""
    
    def __init__(self):
        self.okx_data_service = OKXDataService()
        self.advisor = AIFinancialAdvisor()
        self.ml_pipeline = TradingMLPipeline()
        
        # Major USDT trading pairs
        self.supported_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'SOLUSDT', 'AVAXUSDT',
            'UNIUSDT', 'MATICUSDT', 'FTMUSDT', 'ATOMUSDT', 'NEARUSDT',
            'SANDUSDT', 'MANAUSDT', 'ALICEUSDT', 'CHZUSDT', 'ENJUSDT'
        ]
        
    def generate_daily_picks(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Generate daily top picks based on comprehensive analysis"""
        
        # Get analysis for all supported symbols
        symbol_analysis = []
        
        for symbol in self.supported_symbols:
            try:
                analysis = self._analyze_symbol_for_picks(symbol)
                if analysis['score'] > 0:  # Only include symbols with positive analysis
                    symbol_analysis.append(analysis)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by composite score and return top picks
        symbol_analysis.sort(key=lambda x: x['score'], reverse=True)
        
        # Add historical performance for top picks
        top_picks = symbol_analysis[:top_n]
        for pick in top_picks:
            pick.update(self._get_historical_performance(pick['symbol']))
        
        return top_picks
    
    def _analyze_symbol_for_picks(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive analysis for daily picks selection"""
        
        # Get current market data
        current_data = self.okx_data_service.get_historical_data(symbol, '1h', limit=100)
        
        if current_data.empty:
            return {'symbol': symbol, 'score': 0}
        
        # Calculate individual metrics
        predicted_gain = self._calculate_predicted_gain(symbol, current_data)
        volatility_score = self._calculate_volatility_score(current_data)
        volume_score = self._calculate_volume_score(current_data)
        technical_score = self._calculate_technical_momentum(current_data)
        ml_confidence = self._get_ml_confidence(symbol, current_data)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(
            predicted_gain, volatility_score, volume_score, 
            technical_score, ml_confidence
        )
        
        return {
            'symbol': symbol,
            'score': composite_score,
            'predicted_gain': predicted_gain,
            'volatility_score': volatility_score,
            'volume_score': volume_score,
            'technical_score': technical_score,
            'ml_confidence': ml_confidence,
            'current_price': current_data['close'].iloc[-1],
            'price_change_24h': self._calculate_24h_change(current_data),
            'analysis_timestamp': datetime.now()
        }
    
    def _calculate_predicted_gain(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate predicted gain using ML models"""
        try:
            # Get ML prediction
            prediction = self.ml_pipeline.predict(symbol, data.tail(1))
            
            # Convert signal to expected return
            signal = prediction.get('signal', 0.5)
            confidence = prediction.get('confidence', 0.5)
            
            # Estimate potential gain based on signal strength
            if signal > 0.7:
                base_gain = 0.05  # 5% potential gain for strong buy signal
            elif signal > 0.6:
                base_gain = 0.03  # 3% for moderate buy
            elif signal < 0.3:
                base_gain = -0.03  # Negative for sell signals
            elif signal < 0.4:
                base_gain = -0.01  # Small negative for weak sell
            else:
                base_gain = 0.01  # Small positive for hold
            
            # Adjust by confidence
            return base_gain * confidence
            
        except Exception:
            return 0.01  # Small positive default
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """Calculate volatility score (lower volatility = higher score)"""
        try:
            if len(data) < 20:
                return 0.5
            
            # Calculate 20-period volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Convert to score (inverse relationship - lower volatility is better)
            if volatility < 0.02:  # Very low volatility
                return 0.9
            elif volatility < 0.05:  # Low volatility
                return 0.7
            elif volatility < 0.08:  # Medium volatility
                return 0.5
            elif volatility < 0.12:  # High volatility
                return 0.3
            else:  # Very high volatility
                return 0.1
                
        except Exception:
            return 0.5
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume score based on recent volume trends"""
        try:
            if len(data) < 20:
                return 0.5
            
            # Calculate volume moving average
            volume_ma = data['volume'].rolling(20).mean()
            current_volume = data['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Convert to score
            if volume_ratio > 2.0:  # Very high volume
                return 0.9
            elif volume_ratio > 1.5:  # High volume
                return 0.8
            elif volume_ratio > 1.2:  # Above average volume
                return 0.7
            elif volume_ratio > 0.8:  # Normal volume
                return 0.6
            else:  # Low volume
                return 0.3
                
        except Exception:
            return 0.5
    
    def _calculate_technical_momentum(self, data: pd.DataFrame) -> float:
        """Calculate technical momentum score"""
        try:
            if len(data) < 50:
                return 0.5
            
            score = 0.5  # Start neutral
            
            # Price vs moving averages
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            current_price = data['close'].iloc[-1]
            
            if current_price > sma_20.iloc[-1]:
                score += 0.15
            if current_price > sma_50.iloc[-1]:
                score += 0.15
            if sma_20.iloc[-1] > sma_50.iloc[-1]:  # Golden cross
                score += 0.1
            
            # RSI momentum
            rsi = self._calculate_rsi(data['close'])
            if not rsi.empty:
                rsi_value = rsi.iloc[-1]
                if 40 <= rsi_value <= 60:  # Neutral zone is good for momentum
                    score += 0.1
                elif rsi_value < 30:  # Oversold - potential bounce
                    score += 0.05
            
            # Price momentum (recent trend)
            if len(data) >= 5:
                recent_change = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]
                if recent_change > 0.02:  # Positive momentum
                    score += 0.1
                elif recent_change < -0.02:  # Negative momentum
                    score -= 0.1
            
            return max(0, min(1, score))
            
        except Exception:
            return 0.5
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_ml_confidence(self, symbol: str, data: pd.DataFrame) -> float:
        """Get ML model confidence for the symbol"""
        try:
            prediction = self.ml_pipeline.predict(symbol, data.tail(1))
            return prediction.get('confidence', 0.5)
        except Exception:
            return 0.5
    
    def _calculate_24h_change(self, data: pd.DataFrame) -> float:
        """Calculate 24-hour price change percentage"""
        try:
            if len(data) < 24:  # Less than 24 hours of hourly data
                current_price = data['close'].iloc[-1]
                old_price = data['close'].iloc[0]
            else:
                current_price = data['close'].iloc[-1]
                old_price = data['close'].iloc[-24]  # 24 hours ago
            
            return (current_price - old_price) / old_price * 100
        except Exception:
            return 0.0
    
    def _calculate_composite_score(self, predicted_gain: float, volatility_score: float,
                                 volume_score: float, technical_score: float, 
                                 ml_confidence: float) -> float:
        """Calculate weighted composite score for ranking"""
        
        # Define weights for different factors
        weights = {
            'predicted_gain': 0.3,
            'ml_confidence': 0.25,
            'technical_score': 0.2,
            'volume_score': 0.15,
            'volatility_score': 0.1
        }
        
        # Normalize predicted gain to 0-1 scale
        gain_score = max(0, min(1, (predicted_gain + 0.05) / 0.1))  # -5% to +5% range
        
        # Calculate weighted score
        composite_score = (
            gain_score * weights['predicted_gain'] +
            ml_confidence * weights['ml_confidence'] +
            technical_score * weights['technical_score'] +
            volume_score * weights['volume_score'] +
            volatility_score * weights['volatility_score']
        )
        
        return composite_score
    
    def _get_historical_performance(self, symbol: str) -> Dict[str, Any]:
        """Get historical performance statistics"""
        try:
            # This would ideally connect to a database with historical signals
            # For now, return placeholder structure that can be populated
            return {
                'win_rate_7d': 0.65,  # Placeholder - would calculate from historical data
                'avg_return_7d': 0.024,  # Placeholder - 2.4% average return
                'total_signals_7d': 12,  # Placeholder - number of signals
                'last_signal': 'BUY',  # Placeholder - last signal direction
                'last_signal_time': datetime.now() - timedelta(hours=4)
            }
        except Exception:
            return {
                'win_rate_7d': 0.0,
                'avg_return_7d': 0.0,
                'total_signals_7d': 0,
                'last_signal': 'NONE',
                'last_signal_time': None
            }
    
    def get_picks_by_strategy(self, strategy: str = 'balanced') -> List[Dict[str, Any]]:
        """Get picks optimized for specific strategy"""
        
        all_picks = self.generate_daily_picks(top_n=20)
        
        if strategy == 'high_gain':
            # Sort by predicted gain
            return sorted(all_picks, key=lambda x: x['predicted_gain'], reverse=True)[:10]
        
        elif strategy == 'low_risk':
            # Sort by combination of low volatility and high confidence
            return sorted(all_picks, 
                         key=lambda x: x['volatility_score'] * x['ml_confidence'], 
                         reverse=True)[:10]
        
        elif strategy == 'momentum':
            # Sort by technical momentum
            return sorted(all_picks, key=lambda x: x['technical_score'], reverse=True)[:10]
        
        else:  # balanced
            return all_picks[:10]
    
    def get_picks_summary(self) -> Dict[str, Any]:
        """Get summary statistics for daily picks"""
        picks = self.generate_daily_picks()
        
        if not picks:
            return {
                'total_picks': 0,
                'avg_score': 0,
                'buy_signals': 0,
                'hold_signals': 0,
                'high_confidence': 0
            }
        
        return {
            'total_picks': len(picks),
            'avg_score': np.mean([p['score'] for p in picks]),
            'avg_predicted_gain': np.mean([p['predicted_gain'] for p in picks]),
            'high_confidence_picks': len([p for p in picks if p['ml_confidence'] > 0.7]),
            'low_volatility_picks': len([p for p in picks if p['volatility_score'] > 0.7]),
            'generated_at': datetime.now()
        }