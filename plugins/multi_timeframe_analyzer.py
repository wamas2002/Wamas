"""
Multi-Timeframe Analysis Plugin
Advanced analysis across multiple timeframes with TradingView integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
import ccxt

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        self.exchanges = {
            'okx': ccxt.okx({'sandbox': False}),
            'binance': ccxt.binance({'sandbox': False})
        }
        
    def get_multi_timeframe_data(self, symbol: str, exchange: str = 'okx', limit: int = 100) -> Dict:
        """Get OHLCV data across multiple timeframes"""
        try:
            exchange_obj = self.exchanges.get(exchange)
            if not exchange_obj:
                raise Exception(f"Exchange {exchange} not supported for authentic data access")
            
            multi_tf_data = {}
            
            for tf in self.timeframes:
                try:
                    ohlcv = exchange_obj.fetch_ohlcv(symbol, tf, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Calculate basic indicators
                    df = self._calculate_indicators(df)
                    
                    multi_tf_data[tf] = {
                        'data': df.to_dict('records'),
                        'current_price': float(df['close'].iloc[-1]),
                        'change_24h': self._calculate_24h_change(df),
                        'volume_24h': float(df['volume'].sum()),
                        'trend': self._determine_trend(df),
                        'signals': self._generate_signals(df, tf)
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to fetch authentic {tf} data for {symbol}: {e}")
                    # Skip this timeframe if authentic data unavailable
                    continue
            
            return {
                'success': True,
                'symbol': symbol,
                'exchange': exchange,
                'timeframes': multi_tf_data,
                'analysis': self._analyze_confluence(multi_tf_data),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Multi-timeframe analysis error: {e}")
            raise Exception(f"Unable to fetch authentic multi-timeframe data for {symbol}: {e}")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for timeframe analysis"""
        try:
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return df
    
    def _calculate_24h_change(self, df: pd.DataFrame) -> float:
        """Calculate 24h price change percentage"""
        try:
            if len(df) < 2:
                return 0.0
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            return ((current_price - prev_price) / prev_price) * 100
        except:
            return 0.0
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine trend direction for timeframe"""
        try:
            if len(df) < 50:
                return 'neutral'
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Trend scoring
            score = 0
            
            # Price vs moving averages
            if current_price > sma_20:
                score += 1
            if current_price > sma_50:
                score += 1
            if sma_20 > sma_50:
                score += 1
            
            # RSI momentum
            if rsi > 50:
                score += 1
            elif rsi > 70:
                score -= 1  # Overbought
            elif rsi < 30:
                score -= 1  # Oversold
            
            # MACD
            if not pd.isna(df['macd'].iloc[-1]) and not pd.isna(df['macd_signal'].iloc[-1]):
                if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                    score += 1
            
            # Determine trend
            if score >= 3:
                return 'bullish'
            elif score <= 1:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Trend determination error: {e}")
            return 'neutral'
    
    def _generate_signals(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Generate trading signals for specific timeframe"""
        signals = []
        
        try:
            if len(df) < 20:
                return signals
            
            current = df.iloc[-1]
            
            # RSI signals
            if current['rsi'] < 30:
                signals.append({
                    'type': 'buy',
                    'indicator': 'RSI',
                    'strength': 'strong',
                    'message': f'RSI oversold ({current["rsi"]:.1f}) on {timeframe}',
                    'confidence': 0.75
                })
            elif current['rsi'] > 70:
                signals.append({
                    'type': 'sell',
                    'indicator': 'RSI',
                    'strength': 'moderate',
                    'message': f'RSI overbought ({current["rsi"]:.1f}) on {timeframe}',
                    'confidence': 0.65
                })
            
            # MACD signals
            if not pd.isna(current['macd']) and not pd.isna(current['macd_signal']):
                prev = df.iloc[-2]
                if (current['macd'] > current['macd_signal'] and 
                    prev['macd'] <= prev['macd_signal']):
                    signals.append({
                        'type': 'buy',
                        'indicator': 'MACD',
                        'strength': 'moderate',
                        'message': f'MACD bullish crossover on {timeframe}',
                        'confidence': 0.70
                    })
            
            # Bollinger Bands signals
            if current['close'] <= current['bb_lower']:
                signals.append({
                    'type': 'buy',
                    'indicator': 'Bollinger Bands',
                    'strength': 'moderate',
                    'message': f'Price touching lower Bollinger Band on {timeframe}',
                    'confidence': 0.60
                })
            elif current['close'] >= current['bb_upper']:
                signals.append({
                    'type': 'sell',
                    'indicator': 'Bollinger Bands',
                    'strength': 'moderate',
                    'message': f'Price touching upper Bollinger Band on {timeframe}',
                    'confidence': 0.60
                })
            
            # Volume confirmation
            if current['volume_ratio'] > 1.5:
                for signal in signals:
                    signal['confidence'] = min(signal['confidence'] + 0.1, 0.95)
                    signal['message'] += ' (High volume confirmation)'
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return signals
    
    def _analyze_confluence(self, multi_tf_data: Dict) -> Dict:
        """Analyze confluence across multiple timeframes"""
        try:
            confluence = {
                'overall_trend': 'neutral',
                'strength': 'weak',
                'buy_signals': 0,
                'sell_signals': 0,
                'key_levels': [],
                'recommendations': []
            }
            
            trends = []
            all_signals = []
            
            # Collect trends and signals from all timeframes
            for tf, data in multi_tf_data.items():
                trends.append(data['trend'])
                all_signals.extend(data['signals'])
            
            # Analyze trend confluence
            bullish_count = trends.count('bullish')
            bearish_count = trends.count('bearish')
            neutral_count = trends.count('neutral')
            
            if bullish_count > bearish_count:
                confluence['overall_trend'] = 'bullish'
                confluence['strength'] = 'strong' if bullish_count >= 5 else 'moderate'
            elif bearish_count > bullish_count:
                confluence['overall_trend'] = 'bearish'
                confluence['strength'] = 'strong' if bearish_count >= 5 else 'moderate'
            
            # Count signal types
            confluence['buy_signals'] = len([s for s in all_signals if s['type'] == 'buy'])
            confluence['sell_signals'] = len([s for s in all_signals if s['type'] == 'sell'])
            
            # Generate recommendations
            if confluence['buy_signals'] > confluence['sell_signals'] + 2:
                confluence['recommendations'].append({
                    'action': 'BUY',
                    'reason': f'Strong buy confluence ({confluence["buy_signals"]} buy vs {confluence["sell_signals"]} sell signals)',
                    'confidence': min(confluence['buy_signals'] * 0.15, 0.9)
                })
            elif confluence['sell_signals'] > confluence['buy_signals'] + 2:
                confluence['recommendations'].append({
                    'action': 'SELL',
                    'reason': f'Strong sell confluence ({confluence["sell_signals"]} sell vs {confluence["buy_signals"]} buy signals)',
                    'confidence': min(confluence['sell_signals'] * 0.15, 0.9)
                })
            else:
                confluence['recommendations'].append({
                    'action': 'HOLD',
                    'reason': 'Mixed signals across timeframes, wait for clearer direction',
                    'confidence': 0.6
                })
            
            return confluence
            
        except Exception as e:
            logger.error(f"Confluence analysis error: {e}")
            return {
                'overall_trend': 'neutral',
                'strength': 'weak',
                'buy_signals': 0,
                'sell_signals': 0,
                'recommendations': [{'action': 'HOLD', 'reason': 'Analysis unavailable', 'confidence': 0.5}]
            }
    
    def _get_authentic_data_only(self, symbol: str, exchange: str) -> Dict:
        """Get only authentic market data, no fallback allowed"""
        try:
            exchange_obj = self.exchanges.get(exchange)
            if not exchange_obj:
                raise Exception(f"Exchange {exchange} not available")
            
            # Fetch real OHLCV data for primary timeframe
            ohlcv = exchange_obj.fetch_ohlcv(symbol, '1h', limit=100)
            if not ohlcv:
                raise Exception(f"No market data available for {symbol}")
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return {
                'success': True,
                'symbol': symbol,
                'exchange': exchange,
                'authentic_data': True,
                'last_price': float(df['close'].iloc[-1]),
                'volume_24h': float(df['volume'].sum()),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Authentic data fetch failed for {symbol}: {e}")
            raise Exception(f"Unable to fetch authentic market data for {symbol} from {exchange}: {e}")

# Plugin instance
multi_timeframe_analyzer = MultiTimeframeAnalyzer()

def analyze_multi_timeframe(symbol: str, exchange: str = 'okx') -> Dict:
    """Main function to analyze multiple timeframes"""
    return multi_timeframe_analyzer.get_multi_timeframe_data(symbol, exchange)

def get_supported_timeframes() -> List[str]:
    """Get list of supported timeframes"""
    return multi_timeframe_analyzer.timeframes

def get_supported_exchanges() -> List[str]:
    """Get list of supported exchanges"""
    return list(multi_timeframe_analyzer.exchanges.keys())