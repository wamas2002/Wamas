"""
Market Regime Detector - Context-Aware Trading Environment Analysis
Detects bull/bear/sideways markets to adjust signal confidence dynamically
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Detect market regime using ADX, MACD slope, and EMA angles"""
    
    def __init__(self):
        self.regime_history = []
        self.confidence_adjustments = {
            'bull': 0.0,      # No adjustment in bull market
            'bear': -0.20,    # Reduce BUY confidence by 20% in bear market
            'sideways': 0.0   # No adjustment but filter low confidence
        }
    
    def calculate_ema_angle(self, prices: pd.Series, period: int = 50) -> float:
        """Calculate EMA angle to determine trend direction"""
        try:
            ema = ta.ema(prices, length=period)
            if len(ema) < 5:
                return 0.0
            
            # Calculate angle using last 5 periods
            recent_ema = ema.iloc[-5:].values
            x = np.arange(len(recent_ema))
            
            # Linear regression to find slope
            slope = np.polyfit(x, recent_ema, 1)[0]
            
            # Convert to angle (normalized)
            angle = np.degrees(np.arctan(slope / recent_ema[-1] * 1000))
            return float(angle)
            
        except Exception as e:
            logger.warning(f"EMA angle calculation failed: {e}")
            return 0.0
    
    def calculate_macd_slope(self, prices: pd.Series) -> float:
        """Calculate MACD line slope for trend momentum"""
        try:
            macd = ta.macd(prices)
            if 'MACD_12_26_9' not in macd.columns or len(macd) < 5:
                return 0.0
            
            macd_line = macd['MACD_12_26_9'].iloc[-5:].values
            if len(macd_line) < 5:
                return 0.0
            
            x = np.arange(len(macd_line))
            slope = np.polyfit(x, macd_line, 1)[0]
            return float(slope)
            
        except Exception as e:
            logger.warning(f"MACD slope calculation failed: {e}")
            return 0.0
    
    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """Detect current market regime using multiple indicators"""
        try:
            if len(df) < 50:
                return {
                    'regime': 'unknown',
                    'confidence': 0.0,
                    'bull_score': 0,
                    'bear_score': 0,
                    'sideways_score': 0,
                    'details': 'Insufficient data'
                }
            
            # Calculate ADX for trend strength
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            current_adx = adx['ADX_14'].iloc[-1] if 'ADX_14' in adx.columns else 20
            
            # Calculate EMA angles
            ema20_angle = self.calculate_ema_angle(df['close'], 20)
            ema50_angle = self.calculate_ema_angle(df['close'], 50)
            
            # Calculate MACD slope
            macd_slope = self.calculate_macd_slope(df['close'])
            
            # Price position relative to EMAs
            current_price = df['close'].iloc[-1]
            ema20 = ta.ema(df['close'], length=20).iloc[-1]
            ema50 = ta.ema(df['close'], length=50).iloc[-1]
            
            price_above_ema20 = current_price > ema20
            price_above_ema50 = current_price > ema50
            ema20_above_ema50 = ema20 > ema50
            
            # Scoring system
            bull_score = 0
            bear_score = 0
            sideways_score = 0
            
            # ADX contribution (trend strength)
            if current_adx > 25:  # Strong trend
                if ema20_angle > 2 and ema50_angle > 1:
                    bull_score += 30
                elif ema20_angle < -2 and ema50_angle < -1:
                    bear_score += 30
            else:  # Weak trend (sideways)
                sideways_score += 25
            
            # MACD slope contribution
            if macd_slope > 0.001:
                bull_score += 20
            elif macd_slope < -0.001:
                bear_score += 20
            else:
                sideways_score += 15
            
            # Price vs EMA contribution
            if price_above_ema20 and price_above_ema50 and ema20_above_ema50:
                bull_score += 25
            elif not price_above_ema20 and not price_above_ema50 and not ema20_above_ema50:
                bear_score += 25
            else:
                sideways_score += 20
            
            # EMA angle contribution
            if ema20_angle > 3 and ema50_angle > 1.5:
                bull_score += 15
            elif ema20_angle < -3 and ema50_angle < -1.5:
                bear_score += 15
            else:
                sideways_score += 10
            
            # Additional volatility check for sideways
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            if volatility < 0.02:  # Low volatility suggests sideways
                sideways_score += 10
            
            # Determine regime
            max_score = max(bull_score, bear_score, sideways_score)
            
            if max_score == bull_score and bull_score > 50:
                regime = 'bull'
                confidence = min(bull_score / 100, 0.95)
            elif max_score == bear_score and bear_score > 50:
                regime = 'bear'
                confidence = min(bear_score / 100, 0.95)
            elif max_score == sideways_score and sideways_score > 40:
                regime = 'sideways'
                confidence = min(sideways_score / 100, 0.90)
            else:
                regime = 'unknown'
                confidence = 0.5
            
            regime_data = {
                'regime': regime,
                'confidence': confidence,
                'bull_score': bull_score,
                'bear_score': bear_score,
                'sideways_score': sideways_score,
                'details': {
                    'adx': current_adx,
                    'ema20_angle': ema20_angle,
                    'ema50_angle': ema50_angle,
                    'macd_slope': macd_slope,
                    'price_above_ema20': price_above_ema20,
                    'price_above_ema50': price_above_ema50,
                    'volatility': volatility
                }
            }
            
            # Store in history
            self.regime_history.append(regime_data)
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            return regime_data
            
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'bull_score': 0,
                'bear_score': 0,
                'sideways_score': 0,
                'details': f'Error: {e}'
            }
    
    def apply_market_context_filter(self, signal_confidence: float, 
                                   signal_type: str, regime_data: Dict) -> Tuple[float, bool]:
        """Apply market context penalty to signal confidence"""
        try:
            regime = regime_data['regime']
            adjusted_confidence = signal_confidence
            execute_signal = True
            
            # Apply regime-based adjustments
            if signal_type == 'BUY':
                if regime == 'bear':
                    # Reduce BUY confidence by 20% in bear market
                    adjusted_confidence *= (1 + self.confidence_adjustments['bear'])
                    # Block BUY signals unless confidence > 80% in bear market
                    if adjusted_confidence < 80:
                        execute_signal = False
                        
                elif regime == 'sideways':
                    # Discard signals with confidence < 75% in sideways market
                    if signal_confidence < 75:
                        execute_signal = False
            
            elif signal_type == 'SELL':
                if regime == 'bull':
                    # Be more conservative with SELL signals in bull market
                    if signal_confidence < 75:
                        execute_signal = False
                elif regime == 'bear':
                    # Favor SELL signals in bear market
                    adjusted_confidence *= 1.05
            
            # Ensure confidence stays within bounds
            adjusted_confidence = max(0, min(100, adjusted_confidence))
            
            return adjusted_confidence, execute_signal
            
        except Exception as e:
            logger.error(f"Market context filter failed: {e}")
            return signal_confidence, True
    
    def get_regime_summary(self) -> Dict:
        """Get summary of recent regime detection"""
        if not self.regime_history:
            return {'current_regime': 'unknown', 'stability': 0}
        
        recent_regimes = [r['regime'] for r in self.regime_history[-10:]]
        current_regime = self.regime_history[-1]['regime']
        
        # Calculate regime stability (consistency)
        stability = recent_regimes.count(current_regime) / len(recent_regimes)
        
        return {
            'current_regime': current_regime,
            'stability': stability,
            'recent_confidence': self.regime_history[-1]['confidence'],
            'bull_periods': recent_regimes.count('bull'),
            'bear_periods': recent_regimes.count('bear'),
            'sideways_periods': recent_regimes.count('sideways')
        }