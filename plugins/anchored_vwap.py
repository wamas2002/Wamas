"""
Anchored VWAP Visual Tool Plugin
TrendSpider-style anchored VWAP with manual/automatic anchoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AnchoredVWAPPlugin:
    """Anchored VWAP calculation and visualization plugin"""
    
    plugin_name = "anchored_vwap"
    
    def __init__(self):
        self.anchor_points = {}
        self.vwap_lines = {}
        
    def detect_swing_points(self, data: pd.DataFrame, lookback: int = 10) -> Dict:
        """Detect swing highs and lows for automatic anchoring"""
        try:
            highs = data['high'].rolling(window=lookback*2+1, center=True).max()
            lows = data['low'].rolling(window=lookback*2+1, center=True).min()
            
            swing_highs = []
            swing_lows = []
            
            for i in range(lookback, len(data) - lookback):
                # Swing high detection
                if data['high'].iloc[i] == highs.iloc[i]:
                    swing_highs.append({
                        'timestamp': data.index[i],
                        'price': data['high'].iloc[i],
                        'type': 'swing_high'
                    })
                
                # Swing low detection
                if data['low'].iloc[i] == lows.iloc[i]:
                    swing_lows.append({
                        'timestamp': data.index[i],
                        'price': data['low'].iloc[i],
                        'type': 'swing_low'
                    })
            
            return {
                'swing_highs': swing_highs[-5:],  # Last 5 swing highs
                'swing_lows': swing_lows[-5:],   # Last 5 swing lows
                'latest_swing': swing_highs[-1] if swing_highs else swing_lows[-1] if swing_lows else None
            }
            
        except Exception as e:
            logger.error(f"Error detecting swing points: {e}")
            return {'swing_highs': [], 'swing_lows': [], 'latest_swing': None}
    
    def detect_breakout_candles(self, data: pd.DataFrame, volume_threshold: float = 1.5) -> List[Dict]:
        """Detect breakout candles based on volume and price action"""
        try:
            breakout_candles = []
            
            # Calculate volume average
            avg_volume = data['volume'].rolling(window=20).mean()
            
            # Calculate price range
            data['range'] = data['high'] - data['low']
            avg_range = data['range'].rolling(window=20).mean()
            
            for i in range(20, len(data)):
                volume_spike = data['volume'].iloc[i] > (avg_volume.iloc[i] * volume_threshold)
                large_range = data['range'].iloc[i] > (avg_range.iloc[i] * 1.2)
                
                if volume_spike and large_range:
                    breakout_candles.append({
                        'timestamp': data.index[i],
                        'price': data['close'].iloc[i],
                        'volume': data['volume'].iloc[i],
                        'type': 'breakout',
                        'strength': (data['volume'].iloc[i] / avg_volume.iloc[i])
                    })
            
            return breakout_candles[-10:]  # Last 10 breakout candles
            
        except Exception as e:
            logger.error(f"Error detecting breakout candles: {e}")
            return []
    
    def calculate_anchored_vwap(self, data: pd.DataFrame, anchor_timestamp: str, anchor_price: float = None) -> Dict:
        """Calculate anchored VWAP from specified anchor point"""
        try:
            # Find anchor index
            anchor_idx = None
            for i, timestamp in enumerate(data.index):
                if str(timestamp) >= anchor_timestamp:
                    anchor_idx = i
                    break
            
            if anchor_idx is None:
                anchor_idx = 0
            
            # Calculate VWAP from anchor point onwards
            anchored_data = data.iloc[anchor_idx:].copy()
            
            # VWAP calculation
            anchored_data['typical_price'] = (anchored_data['high'] + anchored_data['low'] + anchored_data['close']) / 3
            anchored_data['tp_volume'] = anchored_data['typical_price'] * anchored_data['volume']
            
            # Cumulative sums from anchor point
            anchored_data['cum_tp_volume'] = anchored_data['tp_volume'].cumsum()
            anchored_data['cum_volume'] = anchored_data['volume'].cumsum()
            
            # VWAP line
            anchored_data['anchored_vwap'] = anchored_data['cum_tp_volume'] / anchored_data['cum_volume']
            
            # Calculate standard deviation bands
            anchored_data['vwap_variance'] = ((anchored_data['typical_price'] - anchored_data['anchored_vwap']) ** 2) * anchored_data['volume']
            anchored_data['cum_variance'] = anchored_data['vwap_variance'].cumsum()
            anchored_data['vwap_std'] = np.sqrt(anchored_data['cum_variance'] / anchored_data['cum_volume'])
            
            # Standard deviation bands
            anchored_data['vwap_upper_1'] = anchored_data['anchored_vwap'] + anchored_data['vwap_std']
            anchored_data['vwap_lower_1'] = anchored_data['anchored_vwap'] - anchored_data['vwap_std']
            anchored_data['vwap_upper_2'] = anchored_data['anchored_vwap'] + (anchored_data['vwap_std'] * 2)
            anchored_data['vwap_lower_2'] = anchored_data['anchored_vwap'] - (anchored_data['vwap_std'] * 2)
            
            # Prepare result
            result = {
                'anchor_point': {
                    'timestamp': anchor_timestamp,
                    'price': anchor_price or anchored_data['close'].iloc[0],
                    'index': anchor_idx
                },
                'vwap_data': []
            }
            
            for i, (timestamp, row) in enumerate(anchored_data.iterrows()):
                result['vwap_data'].append({
                    'timestamp': str(timestamp),
                    'vwap': float(row['anchored_vwap']),
                    'upper_1': float(row['vwap_upper_1']),
                    'lower_1': float(row['vwap_lower_1']),
                    'upper_2': float(row['vwap_upper_2']),
                    'lower_2': float(row['vwap_lower_2']),
                    'volume': float(row['volume']),
                    'price': float(row['close'])
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating anchored VWAP: {e}")
            return {'anchor_point': None, 'vwap_data': []}
    
    def get_anchor_suggestions(self, data: pd.DataFrame) -> Dict:
        """Get automatic anchor point suggestions"""
        try:
            swing_points = self.detect_swing_points(data)
            breakout_candles = self.detect_breakout_candles(data)
            
            suggestions = []
            
            # Add swing point suggestions
            for swing in swing_points['swing_highs'][-3:]:
                suggestions.append({
                    'type': 'swing_high',
                    'timestamp': str(swing['timestamp']),
                    'price': swing['price'],
                    'description': f"Swing High at {swing['price']:.2f}",
                    'confidence': 0.8
                })
            
            for swing in swing_points['swing_lows'][-3:]:
                suggestions.append({
                    'type': 'swing_low',
                    'timestamp': str(swing['timestamp']),
                    'price': swing['price'],
                    'description': f"Swing Low at {swing['price']:.2f}",
                    'confidence': 0.8
                })
            
            # Add breakout suggestions
            for breakout in breakout_candles[-3:]:
                suggestions.append({
                    'type': 'breakout',
                    'timestamp': str(breakout['timestamp']),
                    'price': breakout['price'],
                    'description': f"Volume Breakout (strength: {breakout['strength']:.1f}x)",
                    'confidence': min(0.9, breakout['strength'] / 3.0)
                })
            
            # Sort by confidence
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'suggestions': suggestions[:10],
                'auto_anchor': suggestions[0] if suggestions else None
            }
            
        except Exception as e:
            logger.error(f"Error getting anchor suggestions: {e}")
            return {'suggestions': [], 'auto_anchor': None}
    
    def create_vwap_overlay_config(self, vwap_result: Dict) -> Dict:
        """Create TradingView widget overlay configuration for anchored VWAP"""
        try:
            if not vwap_result.get('vwap_data'):
                return {}
            
            # Prepare data for TradingView overlay
            vwap_lines = []
            band_areas = []
            
            timestamps = [point['timestamp'] for point in vwap_result['vwap_data']]
            vwap_values = [point['vwap'] for point in vwap_result['vwap_data']]
            upper_1_values = [point['upper_1'] for point in vwap_result['vwap_data']]
            lower_1_values = [point['lower_1'] for point in vwap_result['vwap_data']]
            upper_2_values = [point['upper_2'] for point in vwap_result['vwap_data']]
            lower_2_values = [point['lower_2'] for point in vwap_result['vwap_data']]
            
            overlay_config = {
                'studies': [
                    {
                        'name': 'Anchored VWAP',
                        'id': 'anchored_vwap_main',
                        'type': 'line',
                        'data': list(zip(timestamps, vwap_values)),
                        'style': {
                            'color': '#4F8BFF',
                            'linewidth': 2,
                            'linestyle': 'solid'
                        }
                    },
                    {
                        'name': 'VWAP Upper Band 1',
                        'id': 'vwap_upper_1',
                        'type': 'line',
                        'data': list(zip(timestamps, upper_1_values)),
                        'style': {
                            'color': '#4F8BFF',
                            'linewidth': 1,
                            'linestyle': 'dashed',
                            'transparency': 60
                        }
                    },
                    {
                        'name': 'VWAP Lower Band 1',
                        'id': 'vwap_lower_1',
                        'type': 'line',
                        'data': list(zip(timestamps, lower_1_values)),
                        'style': {
                            'color': '#4F8BFF',
                            'linewidth': 1,
                            'linestyle': 'dashed',
                            'transparency': 60
                        }
                    }
                ],
                'markers': [
                    {
                        'name': 'VWAP Anchor',
                        'timestamp': vwap_result['anchor_point']['timestamp'],
                        'price': vwap_result['anchor_point']['price'],
                        'type': 'anchor',
                        'style': {
                            'color': '#FF6B6B',
                            'shape': 'circle',
                            'size': 8
                        },
                        'label': 'VWAP Anchor'
                    }
                ]
            }
            
            return overlay_config
            
        except Exception as e:
            logger.error(f"Error creating VWAP overlay config: {e}")
            return {}
    
    def analyze_price_vs_vwap(self, vwap_result: Dict) -> Dict:
        """Analyze current price position relative to anchored VWAP"""
        try:
            if not vwap_result.get('vwap_data'):
                return {}
            
            latest_data = vwap_result['vwap_data'][-1]
            current_price = latest_data['price']
            vwap_value = latest_data['vwap']
            upper_1 = latest_data['upper_1']
            lower_1 = latest_data['lower_1']
            upper_2 = latest_data['upper_2']
            lower_2 = latest_data['lower_2']
            
            # Determine position
            if current_price > upper_2:
                position = "Above +2σ Band"
                sentiment = "Strong Bullish"
                color = "#00D395"
            elif current_price > upper_1:
                position = "Above +1σ Band"
                sentiment = "Bullish"
                color = "#4F8BFF"
            elif current_price > vwap_value:
                position = "Above VWAP"
                sentiment = "Neutral Bullish"
                color = "#8892B0"
            elif current_price > lower_1:
                position = "Below VWAP"
                sentiment = "Neutral Bearish"
                color = "#8892B0"
            elif current_price > lower_2:
                position = "Below -1σ Band"
                sentiment = "Bearish"
                color = "#FFB800"
            else:
                position = "Below -2σ Band"
                sentiment = "Strong Bearish"
                color = "#F6465D"
            
            # Calculate percentage from VWAP
            vwap_deviation = ((current_price - vwap_value) / vwap_value) * 100
            
            analysis = {
                'position': position,
                'sentiment': sentiment,
                'color': color,
                'current_price': current_price,
                'vwap_value': vwap_value,
                'deviation_percent': vwap_deviation,
                'distance_to_bands': {
                    'upper_1': upper_1 - current_price,
                    'lower_1': current_price - lower_1,
                    'upper_2': upper_2 - current_price,
                    'lower_2': current_price - lower_2
                },
                'trading_signals': self._generate_vwap_signals(current_price, vwap_value, upper_1, lower_1, upper_2, lower_2)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing price vs VWAP: {e}")
            return {}
    
    def _generate_vwap_signals(self, price: float, vwap: float, upper_1: float, lower_1: float, upper_2: float, lower_2: float) -> List[Dict]:
        """Generate trading signals based on VWAP analysis"""
        signals = []
        
        # Mean reversion signals
        if price > upper_2:
            signals.append({
                'type': 'SELL',
                'reason': 'Price extended above +2σ band - mean reversion opportunity',
                'confidence': 0.75,
                'target': vwap,
                'stop_loss': price * 1.02
            })
        elif price < lower_2:
            signals.append({
                'type': 'BUY',
                'reason': 'Price extended below -2σ band - mean reversion opportunity',
                'confidence': 0.75,
                'target': vwap,
                'stop_loss': price * 0.98
            })
        
        # Trend continuation signals
        if price > vwap and price < upper_1:
            signals.append({
                'type': 'BUY',
                'reason': 'Price above VWAP with room to upper band',
                'confidence': 0.6,
                'target': upper_1,
                'stop_loss': vwap * 0.995
            })
        elif price < vwap and price > lower_1:
            signals.append({
                'type': 'SELL',
                'reason': 'Price below VWAP with room to lower band',
                'confidence': 0.6,
                'target': lower_1,
                'stop_loss': vwap * 1.005
            })
        
        return signals