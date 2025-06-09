"""
Auto-Fibonacci Detector Plugin
Automatically draws Fibonacci retracement levels from major pivot points
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AutoFibonacciPlugin:
    """Automatic Fibonacci level detection and drawing plugin"""
    
    plugin_name = "auto_fibonacci"
    
    def __init__(self):
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618]
        self.pivot_cache = {}
        
    def detect_major_pivots(self, data: pd.DataFrame, lookback: int = 20, min_strength: float = 0.02) -> Dict:
        """Detect major pivot highs and lows for Fibonacci drawing"""
        try:
            pivots = {
                'pivot_highs': [],
                'pivot_lows': [],
                'major_high': None,
                'major_low': None
            }
            
            # Calculate pivot points with varying lookback periods
            for window in [10, 15, 20, 25]:
                highs = data['high'].rolling(window=window*2+1, center=True).max()
                lows = data['low'].rolling(window=window*2+1, center=True).min()
                
                for i in range(window, len(data) - window):
                    # Pivot high detection
                    if data['high'].iloc[i] == highs.iloc[i]:
                        # Check if this is a significant pivot
                        price_change = abs(data['high'].iloc[i] - data['low'].iloc[max(0, i-50):i+50].min()) / data['high'].iloc[i]
                        
                        if price_change >= min_strength:
                            pivot_high = {
                                'timestamp': data.index[i],
                                'price': data['high'].iloc[i],
                                'strength': price_change,
                                'window': window,
                                'volume': data['volume'].iloc[i] if 'volume' in data.columns else 0
                            }
                            pivots['pivot_highs'].append(pivot_high)
                    
                    # Pivot low detection
                    if data['low'].iloc[i] == lows.iloc[i]:
                        # Check if this is a significant pivot
                        price_change = abs(data['high'].iloc[max(0, i-50):i+50].max() - data['low'].iloc[i]) / data['low'].iloc[i]
                        
                        if price_change >= min_strength:
                            pivot_low = {
                                'timestamp': data.index[i],
                                'price': data['low'].iloc[i],
                                'strength': price_change,
                                'window': window,
                                'volume': data['volume'].iloc[i] if 'volume' in data.columns else 0
                            }
                            pivots['pivot_lows'].append(pivot_low)
            
            # Remove duplicates and sort by strength
            pivots['pivot_highs'] = self._remove_duplicate_pivots(pivots['pivot_highs'])
            pivots['pivot_lows'] = self._remove_duplicate_pivots(pivots['pivot_lows'])
            
            # Find major pivots (highest strength in recent period)
            recent_highs = [p for p in pivots['pivot_highs'] if p['timestamp'] >= data.index[-100]]
            recent_lows = [p for p in pivots['pivot_lows'] if p['timestamp'] >= data.index[-100]]
            
            if recent_highs:
                pivots['major_high'] = max(recent_highs, key=lambda x: x['strength'])
            if recent_lows:
                pivots['major_low'] = max(recent_lows, key=lambda x: x['strength'])
            
            return pivots
            
        except Exception as e:
            logger.error(f"Error detecting major pivots: {e}")
            return {'pivot_highs': [], 'pivot_lows': [], 'major_high': None, 'major_low': None}
    
    def _remove_duplicate_pivots(self, pivots: List[Dict], min_distance: int = 5) -> List[Dict]:
        """Remove duplicate pivots that are too close to each other"""
        if not pivots:
            return []
        
        # Sort by timestamp
        pivots.sort(key=lambda x: x['timestamp'])
        
        filtered_pivots = []
        for pivot in pivots:
            # Check if this pivot is too close to any existing pivot
            is_duplicate = False
            for existing in filtered_pivots:
                if abs((pivot['timestamp'] - existing['timestamp']).total_seconds()) < min_distance * 3600:  # 5 hours
                    if pivot['strength'] <= existing['strength']:
                        is_duplicate = True
                        break
                    else:
                        # Remove the weaker existing pivot
                        filtered_pivots.remove(existing)
                        break
            
            if not is_duplicate:
                filtered_pivots.append(pivot)
        
        # Sort by strength and return top pivots
        filtered_pivots.sort(key=lambda x: x['strength'], reverse=True)
        return filtered_pivots[:10]  # Keep top 10 strongest pivots
    
    def calculate_fibonacci_levels(self, high_point: Dict, low_point: Dict, direction: str = "auto") -> Dict:
        """Calculate Fibonacci retracement levels between two points"""
        try:
            high_price = high_point['price']
            low_price = low_point['price']
            high_time = high_point['timestamp']
            low_time = low_point['timestamp']
            
            # Determine direction
            if direction == "auto":
                if high_time > low_time:
                    direction = "retracement"  # High came after low
                else:
                    direction = "extension"    # Low came after high
            
            # Calculate price range
            price_range = high_price - low_price
            
            fib_levels = {}
            
            if direction == "retracement":
                # Standard retracement (from high to low)
                for level in self.fib_levels:
                    fib_price = high_price - (price_range * level)
                    fib_levels[f"{level:.3f}"] = {
                        'level': level,
                        'price': fib_price,
                        'description': f"Fib {level:.1%}" if level <= 1.0 else f"Fib Ext {level:.1%}"
                    }
            else:
                # Extension (from low to high)
                for level in self.fib_levels:
                    fib_price = low_price + (price_range * level)
                    fib_levels[f"{level:.3f}"] = {
                        'level': level,
                        'price': fib_price,
                        'description': f"Fib {level:.1%}" if level <= 1.0 else f"Fib Ext {level:.1%}"
                    }
            
            return {
                'direction': direction,
                'high_point': high_point,
                'low_point': low_point,
                'price_range': price_range,
                'levels': fib_levels,
                'key_levels': self._identify_key_levels(fib_levels),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
    
    def _identify_key_levels(self, fib_levels: Dict) -> List[Dict]:
        """Identify the most important Fibonacci levels"""
        key_ratios = [0.382, 0.5, 0.618, 1.0, 1.272, 1.618]
        key_levels = []
        
        for ratio in key_ratios:
            ratio_str = f"{ratio:.3f}"
            if ratio_str in fib_levels:
                level_data = fib_levels[ratio_str].copy()
                level_data['importance'] = self._get_level_importance(ratio)
                key_levels.append(level_data)
        
        return key_levels
    
    def _get_level_importance(self, ratio: float) -> str:
        """Get the importance level of a Fibonacci ratio"""
        if ratio in [0.618, 0.382]:
            return "high"
        elif ratio in [0.5, 1.0]:
            return "medium"
        elif ratio in [1.272, 1.618]:
            return "extension"
        else:
            return "low"
    
    def auto_draw_fibonacci(self, data: pd.DataFrame, symbol: str = "BTC/USDT") -> List[Dict]:
        """Automatically draw Fibonacci levels for the most significant swing"""
        try:
            pivots = self.detect_major_pivots(data)
            fibonacci_drawings = []
            
            # Draw Fibonacci from major high to major low
            if pivots['major_high'] and pivots['major_low']:
                fib_result = self.calculate_fibonacci_levels(
                    pivots['major_high'], 
                    pivots['major_low']
                )
                
                if fib_result:
                    fib_result['symbol'] = symbol
                    fib_result['auto_generated'] = True
                    fib_result['confidence'] = min(
                        pivots['major_high']['strength'],
                        pivots['major_low']['strength']
                    )
                    fibonacci_drawings.append(fib_result)
            
            # Draw additional Fibonacci levels for other significant swings
            recent_highs = [p for p in pivots['pivot_highs'][:3] if p != pivots['major_high']]
            recent_lows = [p for p in pivots['pivot_lows'][:3] if p != pivots['major_low']]
            
            for high in recent_highs:
                for low in recent_lows:
                    if abs((high['timestamp'] - low['timestamp']).total_seconds()) > 24 * 3600:  # At least 1 day apart
                        fib_result = self.calculate_fibonacci_levels(high, low)
                        if fib_result:
                            fib_result['symbol'] = symbol
                            fib_result['auto_generated'] = True
                            fib_result['confidence'] = min(high['strength'], low['strength'])
                            fibonacci_drawings.append(fib_result)
            
            # Sort by confidence and return top drawings
            fibonacci_drawings.sort(key=lambda x: x['confidence'], reverse=True)
            return fibonacci_drawings[:3]  # Return top 3 most confident drawings
            
        except Exception as e:
            logger.error(f"Error auto-drawing Fibonacci: {e}")
            return []
    
    def create_fibonacci_overlay_config(self, fib_drawing: Dict) -> Dict:
        """Create TradingView widget overlay configuration for Fibonacci levels"""
        try:
            if not fib_drawing or not fib_drawing.get('levels'):
                return {}
            
            overlay_config = {
                'studies': [],
                'shapes': [],
                'markers': []
            }
            
            # Add Fibonacci level lines
            for level_key, level_data in fib_drawing['levels'].items():
                if float(level_key) in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
                    line_style = 'solid' if float(level_key) in [0.382, 0.5, 0.618] else 'dashed'
                    line_width = 2 if float(level_key) in [0.382, 0.618] else 1
                    
                    overlay_config['studies'].append({
                        'name': level_data['description'],
                        'id': f"fib_level_{level_key}",
                        'type': 'horizontal_line',
                        'price': level_data['price'],
                        'style': {
                            'color': self._get_fib_level_color(float(level_key)),
                            'linewidth': line_width,
                            'linestyle': line_style,
                            'transparency': 20
                        },
                        'label': {
                            'text': f"{level_data['description']} ({level_data['price']:.2f})",
                            'position': 'right'
                        }
                    })
            
            # Add pivot point markers
            overlay_config['markers'].extend([
                {
                    'name': 'Fibonacci High',
                    'timestamp': str(fib_drawing['high_point']['timestamp']),
                    'price': fib_drawing['high_point']['price'],
                    'type': 'fib_high',
                    'style': {
                        'color': '#F6465D',
                        'shape': 'triangle_down',
                        'size': 10
                    },
                    'label': f"H: {fib_drawing['high_point']['price']:.2f}"
                },
                {
                    'name': 'Fibonacci Low',
                    'timestamp': str(fib_drawing['low_point']['timestamp']),
                    'price': fib_drawing['low_point']['price'],
                    'type': 'fib_low',
                    'style': {
                        'color': '#00D395',
                        'shape': 'triangle_up',
                        'size': 10
                    },
                    'label': f"L: {fib_drawing['low_point']['price']:.2f}"
                }
            ])
            
            return overlay_config
            
        except Exception as e:
            logger.error(f"Error creating Fibonacci overlay config: {e}")
            return {}
    
    def _get_fib_level_color(self, level: float) -> str:
        """Get color for Fibonacci level based on importance"""
        color_map = {
            0.0: '#8892B0',      # Gray
            0.236: '#FFB800',    # Orange
            0.382: '#4F8BFF',    # Blue
            0.5: '#9C27B0',      # Purple
            0.618: '#FF6B6B',    # Red
            0.786: '#FFB800',    # Orange
            1.0: '#8892B0'       # Gray
        }
        return color_map.get(level, '#8892B0')
    
    def analyze_price_vs_fibonacci(self, current_price: float, fib_drawing: Dict) -> Dict:
        """Analyze current price position relative to Fibonacci levels"""
        try:
            if not fib_drawing or not fib_drawing.get('levels'):
                return {}
            
            # Find nearest Fibonacci levels
            levels = [(float(k), v['price']) for k, v in fib_drawing['levels'].items()]
            levels.sort(key=lambda x: x[1])  # Sort by price
            
            support_level = None
            resistance_level = None
            current_zone = None
            
            for i, (ratio, price) in enumerate(levels):
                if current_price >= price:
                    support_level = {'ratio': ratio, 'price': price}
                else:
                    resistance_level = {'ratio': ratio, 'price': price}
                    break
            
            # Determine current zone
            if support_level and resistance_level:
                current_zone = f"{support_level['ratio']:.3f} - {resistance_level['ratio']:.3f}"
            elif support_level:
                current_zone = f"Above {support_level['ratio']:.3f}"
            elif resistance_level:
                current_zone = f"Below {resistance_level['ratio']:.3f}"
            
            # Calculate distances
            support_distance = abs(current_price - support_level['price']) / current_price * 100 if support_level else None
            resistance_distance = abs(resistance_level['price'] - current_price) / current_price * 100 if resistance_level else None
            
            # Generate signals
            signals = self._generate_fibonacci_signals(current_price, support_level, resistance_level, fib_drawing)
            
            analysis = {
                'current_price': current_price,
                'current_zone': current_zone,
                'nearest_support': support_level,
                'nearest_resistance': resistance_level,
                'support_distance_percent': support_distance,
                'resistance_distance_percent': resistance_distance,
                'fibonacci_direction': fib_drawing['direction'],
                'signals': signals,
                'key_levels_nearby': self._find_nearby_key_levels(current_price, fib_drawing)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing price vs Fibonacci: {e}")
            return {}
    
    def _generate_fibonacci_signals(self, price: float, support: Dict, resistance: Dict, fib_drawing: Dict) -> List[Dict]:
        """Generate trading signals based on Fibonacci analysis"""
        signals = []
        
        try:
            # Bounce signals at key levels
            if support and support['ratio'] in [0.382, 0.5, 0.618]:
                distance_to_support = abs(price - support['price']) / price * 100
                if distance_to_support < 1.0:  # Within 1% of key support
                    signals.append({
                        'type': 'BUY',
                        'reason': f"Price near key Fibonacci support {support['ratio']:.1%}",
                        'confidence': 0.7,
                        'entry': support['price'],
                        'target': resistance['price'] if resistance else price * 1.05,
                        'stop_loss': support['price'] * 0.98
                    })
            
            if resistance and resistance['ratio'] in [0.382, 0.5, 0.618]:
                distance_to_resistance = abs(resistance['price'] - price) / price * 100
                if distance_to_resistance < 1.0:  # Within 1% of key resistance
                    signals.append({
                        'type': 'SELL',
                        'reason': f"Price near key Fibonacci resistance {resistance['ratio']:.1%}",
                        'confidence': 0.7,
                        'entry': resistance['price'],
                        'target': support['price'] if support else price * 0.95,
                        'stop_loss': resistance['price'] * 1.02
                    })
            
            # Extension signals
            if support and support['ratio'] > 1.0:
                signals.append({
                    'type': 'SELL',
                    'reason': f"Price extended beyond Fibonacci {support['ratio']:.1%} level",
                    'confidence': 0.6,
                    'entry': price,
                    'target': fib_drawing['levels']['1.000']['price'],
                    'stop_loss': price * 1.03
                })
        
        except Exception as e:
            logger.error(f"Error generating Fibonacci signals: {e}")
        
        return signals
    
    def _find_nearby_key_levels(self, price: float, fib_drawing: Dict, threshold_percent: float = 5.0) -> List[Dict]:
        """Find key Fibonacci levels near current price"""
        key_levels = []
        key_ratios = [0.382, 0.5, 0.618, 1.0]
        
        try:
            for ratio in key_ratios:
                ratio_str = f"{ratio:.3f}"
                if ratio_str in fib_drawing['levels']:
                    level_price = fib_drawing['levels'][ratio_str]['price']
                    distance_percent = abs(price - level_price) / price * 100
                    
                    if distance_percent <= threshold_percent:
                        key_levels.append({
                            'ratio': ratio,
                            'price': level_price,
                            'distance_percent': distance_percent,
                            'description': fib_drawing['levels'][ratio_str]['description']
                        })
            
            # Sort by distance
            key_levels.sort(key=lambda x: x['distance_percent'])
            
        except Exception as e:
            logger.error(f"Error finding nearby key levels: {e}")
        
        return key_levels