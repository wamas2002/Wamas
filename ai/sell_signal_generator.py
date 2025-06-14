"""
SELL Signal Generator - Bearish Market Detection and Exit Signals
Generates SELL signals using negative MACD cross, RSI conditions, and EMA downward cross
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class SellSignalGenerator:
    """Generate SELL signals using technical indicators for bearish conditions"""
    
    def __init__(self):
        self.min_confidence = 70.0
        self.signal_history = []
        
    def analyze_macd_bearish_cross(self, df: pd.DataFrame) -> Dict:
        """Detect negative MACD crossover (bearish signal)"""
        try:
            macd = ta.macd(df['close'])
            if 'MACD_12_26_9' not in macd.columns or 'MACDs_12_26_9' not in macd.columns:
                return {'detected': False, 'strength': 0, 'details': 'MACD data unavailable'}
            
            macd_line = macd['MACD_12_26_9']
            signal_line = macd['MACDs_12_26_9']
            
            if len(macd_line) < 3:
                return {'detected': False, 'strength': 0, 'details': 'Insufficient data'}
            
            # Check for bearish crossover (MACD crosses below signal line)
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            bearish_cross = (prev_macd >= prev_signal and current_macd < current_signal)
            
            if bearish_cross:
                # Calculate strength based on how far below zero and momentum
                strength = 0
                if current_macd < 0:
                    strength += 30  # Below zero line
                
                # Check momentum (how fast it's declining)
                macd_momentum = current_macd - prev_macd
                if macd_momentum < -0.001:
                    strength += 20
                
                # Check histogram declining
                if 'MACDh_12_26_9' in macd.columns:
                    hist_current = macd['MACDh_12_26_9'].iloc[-1]
                    hist_prev = macd['MACDh_12_26_9'].iloc[-2]
                    if hist_current < hist_prev and hist_current < 0:
                        strength += 15
                
                return {
                    'detected': True,
                    'strength': min(strength, 65),
                    'details': f'MACD bearish cross: {current_macd:.6f} < {current_signal:.6f}'
                }
            
            return {'detected': False, 'strength': 0, 'details': 'No bearish cross detected'}
            
        except Exception as e:
            logger.warning(f"MACD bearish analysis failed: {e}")
            return {'detected': False, 'strength': 0, 'details': f'Error: {e}'}
    
    def analyze_rsi_bearish(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI for bearish conditions (RSI < 40 or overbought reversal)"""
        try:
            rsi = ta.rsi(df['close'], length=14)
            if len(rsi) < 3:
                return {'detected': False, 'strength': 0, 'details': 'Insufficient RSI data'}
            
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            strength = 0
            details = []
            
            # RSI below 40 (bearish momentum)
            if current_rsi < 40:
                strength += 25
                details.append(f'RSI below 40: {current_rsi:.1f}')
            
            # RSI declining from overbought (70+)
            if prev_rsi > 70 and current_rsi < prev_rsi:
                strength += 20
                details.append(f'RSI declining from overbought: {prev_rsi:.1f} → {current_rsi:.1f}')
            
            # RSI showing strong downward momentum
            rsi_momentum = current_rsi - prev_rsi
            if rsi_momentum < -5:
                strength += 15
                details.append(f'Strong RSI decline: {rsi_momentum:.1f}')
            
            # RSI divergence (price up, RSI down) - simplified check
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            rsi_change = current_rsi - rsi.iloc[-5]
            if price_change > 0 and rsi_change < -5:
                strength += 10
                details.append('Bearish RSI divergence detected')
            
            detected = strength > 15
            return {
                'detected': detected,
                'strength': min(strength, 70),
                'details': '; '.join(details) if details else f'RSI: {current_rsi:.1f}'
            }
            
        except Exception as e:
            logger.warning(f"RSI bearish analysis failed: {e}")
            return {'detected': False, 'strength': 0, 'details': f'Error: {e}'}
    
    def analyze_ema_downward_cross(self, df: pd.DataFrame) -> Dict:
        """Detect EMA50 downward cross and bearish EMA alignment"""
        try:
            ema20 = ta.ema(df['close'], length=20)
            ema50 = ta.ema(df['close'], length=50)
            
            if len(ema20) < 3 or len(ema50) < 3:
                return {'detected': False, 'strength': 0, 'details': 'Insufficient EMA data'}
            
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            
            current_ema20 = ema20.iloc[-1]
            current_ema50 = ema50.iloc[-1]
            prev_ema20 = ema20.iloc[-2]
            prev_ema50 = ema50.iloc[-2]
            
            strength = 0
            details = []
            
            # Price below both EMAs (bearish)
            if current_price < current_ema20 and current_price < current_ema50:
                strength += 25
                details.append('Price below both EMAs')
            
            # EMA20 crosses below EMA50 (death cross)
            if prev_ema20 >= prev_ema50 and current_ema20 < current_ema50:
                strength += 30
                details.append('EMA20 death cross below EMA50')
            
            # Both EMAs declining
            ema20_declining = current_ema20 < prev_ema20
            ema50_declining = current_ema50 < prev_ema50
            
            if ema20_declining and ema50_declining:
                strength += 20
                details.append('Both EMAs declining')
            elif ema20_declining:
                strength += 10
                details.append('EMA20 declining')
            
            # Price momentum vs EMA
            price_momentum = (current_price - prev_price) / prev_price
            if price_momentum < -0.02:  # 2% decline
                strength += 15
                details.append('Strong price decline')
            
            detected = strength > 20
            return {
                'detected': detected,
                'strength': min(strength, 90),
                'details': '; '.join(details) if details else 'No significant EMA bearish signals'
            }
            
        except Exception as e:
            logger.warning(f"EMA bearish analysis failed: {e}")
            return {'detected': False, 'strength': 0, 'details': f'Error: {e}'}
    
    def analyze_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """Analyze volume for bearish confirmation"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return {'strength': 0, 'details': 'Volume data unavailable'}
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            strength = 0
            if volume_ratio > 1.5:  # High volume on bearish move
                strength = 15
                details = f'High volume confirmation: {volume_ratio:.1f}x average'
            elif volume_ratio > 1.2:
                strength = 8
                details = f'Elevated volume: {volume_ratio:.1f}x average'
            else:
                details = f'Normal volume: {volume_ratio:.1f}x average'
            
            return {'strength': strength, 'details': details}
            
        except Exception as e:
            logger.warning(f"Volume analysis failed: {e}")
            return {'strength': 0, 'details': 'Volume analysis error'}
    
    def generate_sell_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate comprehensive SELL signal"""
        try:
            if len(df) < 50:
                return None
            
            # Analyze all bearish components
            macd_analysis = self.analyze_macd_bearish_cross(df)
            rsi_analysis = self.analyze_rsi_bearish(df)
            ema_analysis = self.analyze_ema_downward_cross(df)
            volume_analysis = self.analyze_volume_confirmation(df)
            
            # Calculate total confidence
            total_strength = (
                macd_analysis['strength'] +
                rsi_analysis['strength'] +
                ema_analysis['strength'] +
                volume_analysis['strength']
            )
            
            # Check if at least 2 indicators are detected
            detections = sum([
                macd_analysis['detected'],
                rsi_analysis['detected'],
                ema_analysis['detected']
            ])
            
            if detections < 2 or total_strength < self.min_confidence:
                return None
            
            # Additional market structure check
            recent_high = df['high'].rolling(10).max().iloc[-1]
            current_price = df['close'].iloc[-1]
            pullback_ratio = (recent_high - current_price) / recent_high
            
            if pullback_ratio > 0.05:  # 5% pullback from recent high
                total_strength += 10
            
            # Cap confidence at 95%
            confidence = min(total_strength, 95.0)
            
            if confidence < self.min_confidence:
                return None
            
            signal = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'symbol': symbol,
                'signal_type': 'SELL',
                'confidence': confidence,
                'price': float(current_price),
                'components': {
                    'macd_bearish': macd_analysis,
                    'rsi_bearish': rsi_analysis,
                    'ema_bearish': ema_analysis,
                    'volume_confirm': volume_analysis
                },
                'target_price': float(current_price * 0.95),  # 5% below current
                'stop_loss': float(current_price * 1.03),     # 3% above current
                'risk_level': self.assess_risk_level(confidence, pullback_ratio)
            }
            
            # Store in history
            self.signal_history.append(signal)
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
            
            return signal
            
        except Exception as e:
            logger.error(f"SELL signal generation failed for {symbol}: {e}")
            return None
    
    def assess_risk_level(self, confidence: float, pullback_ratio: float) -> str:
        """Assess risk level for SELL signal"""
        if confidence > 85 and pullback_ratio > 0.03:
            return 'low'
        elif confidence > 75:
            return 'medium'
        else:
            return 'high'
    
    def generate_trailing_stop_exit(self, entry_price: float, current_price: float, 
                                   peak_price: float, trailing_pct: float = 0.05) -> Optional[Dict]:
        """Generate trailing stop exit signal for profitable trades"""
        try:
            # Only for profitable trades
            if current_price <= entry_price:
                return None
            
            # Calculate profit and trailing stop
            profit_pct = (current_price - entry_price) / entry_price
            trailing_stop_price = peak_price * (1 - trailing_pct)
            
            # Trigger trailing stop if price falls below trailing level
            if current_price <= trailing_stop_price and profit_pct > 0.02:  # At least 2% profit
                return {
                    'signal_type': 'TRAILING_STOP_SELL',
                    'confidence': 90.0,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'peak_price': peak_price,
                    'profit_pct': profit_pct * 100,
                    'reason': f'Trailing stop triggered: {current_price:.6f} <= {trailing_stop_price:.6f}'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Trailing stop calculation failed: {e}")
            return None
    
    def get_recent_sell_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent SELL signals"""
        return self.signal_history[-limit:] if self.signal_history else []