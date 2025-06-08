"""
Asset Explorer Module
Display and analyze all OKX trading pairs with comprehensive metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
from trading.okx_data_service import OKXDataService
from ai.comprehensive_ml_pipeline import TradingMLPipeline
from ai.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')

class AssetExplorer:
    """Comprehensive asset exploration and analysis module"""
    
    def __init__(self):
        self.okx_data_service = OKXDataService()
        self.ml_pipeline = TradingMLPipeline()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        
        # Major trading pairs to focus on
        self.major_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'SOLUSDT', 'AVAXUSDT',
            'UNIUSDT', 'MATICUSDT', 'FTMUSDT', 'ATOMUSDT', 'NEARUSDT'
        ]
        
        # Extended pairs for comprehensive analysis
        self.extended_pairs = self.major_pairs + [
            'SANDUSDT', 'MANAUSDT', 'ALICEUSDT', 'CHZUSDT', 'ENJUSDT',
            'AAVEUSDT', 'MKRUSDT', 'COMPUSDT', 'SUSHIUSDT', 'CRVUSDT',
            'YFIUSDT', '1INCHUSDT', 'SNXUSDT', 'UMAUSDT', 'BALRUSDT',
            'ICPUSDT', 'FLOWUSDT', 'FILUSDT', 'EGLDUSDT', 'THETAUSDT'
        ]
    
    def get_all_assets_overview(self, sort_by: str = 'volume') -> List[Dict[str, Any]]:
        """Get comprehensive overview of all tracked assets"""
        assets_data = []
        
        for symbol in self.extended_pairs:
            try:
                asset_data = self._analyze_single_asset(symbol)
                if asset_data:
                    assets_data.append(asset_data)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort assets based on specified criteria
        return self._sort_assets(assets_data, sort_by)
    
    def _analyze_single_asset(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single asset"""
        try:
            # Get market data
            market_data = self.okx_data_service.get_historical_data(symbol, '1h', limit=168)  # 1 week
            if market_data.empty:
                return None
            
            # Basic market metrics
            current_price = market_data['close'].iloc[-1]
            price_24h_ago = market_data['close'].iloc[-24] if len(market_data) >= 24 else market_data['close'].iloc[0]
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
            
            # Volume analysis
            volume_24h = market_data['volume'].tail(24).sum()
            avg_volume_7d = market_data['volume'].mean()
            volume_ratio = volume_24h / (avg_volume_7d * 24) if avg_volume_7d > 0 else 1
            
            # Volatility analysis
            returns = market_data['close'].pct_change().dropna()
            volatility_24h = returns.tail(24).std() * np.sqrt(24) * 100
            volatility_7d = returns.std() * np.sqrt(168) * 100
            
            # Technical indicators
            technical_analysis = self._calculate_technical_indicators(market_data)
            
            # ML confidence
            ml_confidence = self._get_ml_confidence(symbol, market_data)
            
            # Sentiment score
            sentiment_score = self._get_sentiment_score(symbol)
            
            # Market cap estimation (using current price and volume as proxy)
            market_cap_proxy = current_price * volume_24h
            
            # Support and resistance levels
            support_resistance = self._calculate_support_resistance(market_data)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'volume_24h': volume_24h,
                'volume_ratio': volume_ratio,
                'volatility_24h': volatility_24h,
                'volatility_7d': volatility_7d,
                'market_cap_proxy': market_cap_proxy,
                'ml_confidence': ml_confidence,
                'sentiment_score': sentiment_score,
                'technical_score': technical_analysis['overall_score'],
                'rsi': technical_analysis['rsi'],
                'macd_signal': technical_analysis['macd_signal'],
                'trend_direction': technical_analysis['trend_direction'],
                'support_level': support_resistance['support'],
                'resistance_level': support_resistance['resistance'],
                'risk_level': self._calculate_risk_level(volatility_7d, ml_confidence),
                'opportunity_score': self._calculate_opportunity_score(
                    price_change_24h, volume_ratio, ml_confidence, sentiment_score
                ),
                'last_updated': datetime.now()
            }
        except Exception as e:
            print(f"Error in single asset analysis for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        try:
            if len(data) < 20:
                return {
                    'overall_score': 0.5,
                    'rsi': 50,
                    'macd_signal': 'neutral',
                    'trend_direction': 'sideways'
                }
            
            # RSI
            rsi = self._calculate_rsi(data['close'])
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # MACD
            macd_line, signal_line = self._calculate_macd(data['close'])
            macd_signal = 'neutral'
            if len(macd_line) > 1 and len(signal_line) > 1:
                if macd_line.iloc[-1] > signal_line.iloc[-1]:
                    macd_signal = 'bullish'
                elif macd_line.iloc[-1] < signal_line.iloc[-1]:
                    macd_signal = 'bearish'
            
            # Moving averages for trend
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean() if len(data) >= 50 else sma_20
            
            current_price = data['close'].iloc[-1]
            trend_direction = 'sideways'
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                    trend_direction = 'uptrend'
                elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                    trend_direction = 'downtrend'
            
            # Overall technical score
            score = 0.5
            if current_rsi < 30:
                score += 0.2  # Oversold
            elif current_rsi > 70:
                score -= 0.2  # Overbought
            
            if macd_signal == 'bullish':
                score += 0.15
            elif macd_signal == 'bearish':
                score -= 0.15
            
            if trend_direction == 'uptrend':
                score += 0.1
            elif trend_direction == 'downtrend':
                score -= 0.1
            
            return {
                'overall_score': max(0, min(1, score)),
                'rsi': current_rsi,
                'macd_signal': macd_signal,
                'trend_direction': trend_direction
            }
        except Exception:
            return {
                'overall_score': 0.5,
                'rsi': 50,
                'macd_signal': 'neutral',
                'trend_direction': 'sideways'
            }
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate key support and resistance levels"""
        try:
            if len(data) < 20:
                current_price = data['close'].iloc[-1] if not data.empty else 0
                return {
                    'support': current_price * 0.95,
                    'resistance': current_price * 1.05
                }
            
            # Calculate support and resistance using pivot points
            highs = data['high'].rolling(window=5, center=True).max()
            lows = data['low'].rolling(window=5, center=True).min()
            
            # Find recent support and resistance levels
            recent_data = data.tail(50)  # Last 50 periods
            
            resistance_levels = []
            support_levels = []
            
            for i in range(2, len(recent_data) - 2):
                if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                    resistance_levels.append(recent_data['high'].iloc[i])
                
                if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                    support_levels.append(recent_data['low'].iloc[i])
            
            current_price = data['close'].iloc[-1]
            
            # Find nearest levels
            resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
            support = max(support_levels) if support_levels else current_price * 0.95
            
            return {
                'support': support,
                'resistance': resistance
            }
        except Exception:
            current_price = data['close'].iloc[-1] if not data.empty else 0
            return {
                'support': current_price * 0.95,
                'resistance': current_price * 1.05
            }
    
    def _get_ml_confidence(self, symbol: str, data: pd.DataFrame) -> float:
        """Get ML model confidence for the asset"""
        try:
            prediction = self.ml_pipeline.predict(symbol, data.tail(1))
            return prediction.get('confidence', 0.5)
        except Exception:
            return 0.5
    
    def _get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score for the asset"""
        try:
            sentiment_data = self.sentiment_analyzer.get_symbol_sentiment(symbol)
            return sentiment_data.get('overall_sentiment', 0.5)
        except Exception:
            return 0.5
    
    def _calculate_risk_level(self, volatility: float, ml_confidence: float) -> str:
        """Calculate risk level based on volatility and confidence"""
        # Normalize risk score
        risk_score = volatility / 100 + (1 - ml_confidence)
        
        if risk_score < 0.3:
            return 'Low'
        elif risk_score < 0.6:
            return 'Medium'
        elif risk_score < 0.8:
            return 'High'
        else:
            return 'Very High'
    
    def _calculate_opportunity_score(self, price_change: float, volume_ratio: float, 
                                   ml_confidence: float, sentiment: float) -> float:
        """Calculate opportunity score based on multiple factors"""
        # Weighted scoring
        score = (
            abs(price_change) * 0.2 +  # Price momentum
            min(volume_ratio, 3) * 0.3 +  # Volume (capped at 3x)
            ml_confidence * 0.3 +  # ML confidence
            sentiment * 0.2  # Sentiment
        )
        
        # Normalize to 0-1 scale
        return min(1.0, score / 3.0)
    
    def _sort_assets(self, assets: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        """Sort assets by specified criteria"""
        if not assets:
            return []
        
        if sort_by == 'volume':
            return sorted(assets, key=lambda x: x['volume_24h'], reverse=True)
        elif sort_by == 'volatility':
            return sorted(assets, key=lambda x: x['volatility_24h'], reverse=True)
        elif sort_by == 'ml_confidence':
            return sorted(assets, key=lambda x: x['ml_confidence'], reverse=True)
        elif sort_by == 'price_change':
            return sorted(assets, key=lambda x: abs(x['price_change_24h']), reverse=True)
        elif sort_by == 'opportunity':
            return sorted(assets, key=lambda x: x['opportunity_score'], reverse=True)
        elif sort_by == 'risk':
            risk_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
            return sorted(assets, key=lambda x: risk_order.get(x['risk_level'], 1))
        else:
            return assets
    
    def get_assets_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get assets categorized by different criteria"""
        all_assets = self.get_all_assets_overview()
        
        return {
            'high_volume': [asset for asset in all_assets if asset['volume_ratio'] > 1.5],
            'low_volatility': [asset for asset in all_assets if asset['volatility_24h'] < 5],
            'high_confidence': [asset for asset in all_assets if asset['ml_confidence'] > 0.7],
            'trending_up': [asset for asset in all_assets if asset['price_change_24h'] > 2],
            'trending_down': [asset for asset in all_assets if asset['price_change_24h'] < -2],
            'oversold': [asset for asset in all_assets if asset['rsi'] < 30],
            'overbought': [asset for asset in all_assets if asset['rsi'] > 70],
            'best_opportunities': sorted(all_assets, key=lambda x: x['opportunity_score'], reverse=True)[:10]
        }
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview and statistics"""
        all_assets = self.get_all_assets_overview()
        
        if not all_assets:
            return {
                'total_assets': 0,
                'market_sentiment': 'neutral',
                'avg_volatility': 0,
                'trending_assets': {'up': 0, 'down': 0, 'sideways': 0}
            }
        
        # Calculate market statistics
        total_volume = sum(asset['volume_24h'] for asset in all_assets)
        avg_price_change = np.mean([asset['price_change_24h'] for asset in all_assets])
        avg_volatility = np.mean([asset['volatility_24h'] for asset in all_assets])
        avg_ml_confidence = np.mean([asset['ml_confidence'] for asset in all_assets])
        avg_sentiment = np.mean([asset['sentiment_score'] for asset in all_assets])
        
        # Count trending directions
        trend_counts = {
            'up': len([a for a in all_assets if a['trend_direction'] == 'uptrend']),
            'down': len([a for a in all_assets if a['trend_direction'] == 'downtrend']),
            'sideways': len([a for a in all_assets if a['trend_direction'] == 'sideways'])
        }
        
        # Determine overall market sentiment
        if avg_sentiment > 0.6:
            market_sentiment = 'bullish'
        elif avg_sentiment < 0.4:
            market_sentiment = 'bearish'
        else:
            market_sentiment = 'neutral'
        
        return {
            'total_assets': len(all_assets),
            'total_volume_24h': total_volume,
            'avg_price_change_24h': avg_price_change,
            'avg_volatility_24h': avg_volatility,
            'avg_ml_confidence': avg_ml_confidence,
            'avg_sentiment': avg_sentiment,
            'market_sentiment': market_sentiment,
            'trending_assets': trend_counts,
            'high_volume_assets': len([a for a in all_assets if a['volume_ratio'] > 1.5]),
            'high_confidence_assets': len([a for a in all_assets if a['ml_confidence'] > 0.7]),
            'volatile_assets': len([a for a in all_assets if a['volatility_24h'] > 10]),
            'last_updated': datetime.now()
        }
    
    def search_assets(self, query: str) -> List[Dict[str, Any]]:
        """Search assets by symbol or criteria"""
        all_assets = self.get_all_assets_overview()
        
        query_lower = query.lower()
        results = []
        
        for asset in all_assets:
            symbol_lower = asset['symbol'].lower()
            
            # Search by symbol
            if query_lower in symbol_lower:
                results.append(asset)
                continue
            
            # Search by criteria
            if query_lower in ['high_volume', 'volume'] and asset['volume_ratio'] > 1.5:
                results.append(asset)
            elif query_lower in ['low_volatility', 'stable'] and asset['volatility_24h'] < 5:
                results.append(asset)
            elif query_lower in ['high_confidence', 'confident'] and asset['ml_confidence'] > 0.7:
                results.append(asset)
            elif query_lower in ['bullish', 'up'] and asset['price_change_24h'] > 2:
                results.append(asset)
            elif query_lower in ['bearish', 'down'] and asset['price_change_24h'] < -2:
                results.append(asset)
        
        return results
    
    def get_asset_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific asset"""
        # Get extended market data
        market_data_1h = self.okx_data_service.get_historical_data(symbol, '1h', limit=168)
        market_data_4h = self.okx_data_service.get_historical_data(symbol, '4h', limit=168)
        market_data_1d = self.okx_data_service.get_historical_data(symbol, '1d', limit=30)
        
        if market_data_1h.empty:
            return {'error': f'No data available for {symbol}'}
        
        # Basic analysis
        basic_analysis = self._analyze_single_asset(symbol)
        
        # Multi-timeframe analysis
        timeframe_analysis = {
            '1h': self._calculate_technical_indicators(market_data_1h),
            '4h': self._calculate_technical_indicators(market_data_4h) if not market_data_4h.empty else None,
            '1d': self._calculate_technical_indicators(market_data_1d) if not market_data_1d.empty else None
        }
        
        # Price levels analysis
        price_levels = self._get_detailed_price_levels(market_data_1h)
        
        # Volume profile
        volume_analysis = self._analyze_volume_profile(market_data_1h)
        
        # ML prediction with explanation
        ml_analysis = self._get_detailed_ml_analysis(symbol, market_data_1h)
        
        return {
            'symbol': symbol,
            'basic_analysis': basic_analysis,
            'timeframe_analysis': timeframe_analysis,
            'price_levels': price_levels,
            'volume_analysis': volume_analysis,
            'ml_analysis': ml_analysis,
            'last_updated': datetime.now()
        }
    
    def _get_detailed_price_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed price level analysis"""
        try:
            current_price = data['close'].iloc[-1]
            high_24h = data['high'].tail(24).max()
            low_24h = data['low'].tail(24).min()
            high_7d = data['high'].max()
            low_7d = data['low'].min()
            
            return {
                'current_price': current_price,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'high_7d': high_7d,
                'low_7d': low_7d,
                'distance_to_high_24h': (high_24h - current_price) / current_price * 100,
                'distance_to_low_24h': (current_price - low_24h) / current_price * 100,
                'position_in_24h_range': (current_price - low_24h) / (high_24h - low_24h) if high_24h != low_24h else 0.5
            }
        except Exception:
            return {}
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile and patterns"""
        try:
            current_volume = data['volume'].iloc[-1]
            avg_volume_24h = data['volume'].tail(24).mean()
            avg_volume_7d = data['volume'].mean()
            
            # Volume trend
            volume_trend = 'stable'
            if current_volume > avg_volume_24h * 1.5:
                volume_trend = 'increasing'
            elif current_volume < avg_volume_24h * 0.5:
                volume_trend = 'decreasing'
            
            return {
                'current_volume': current_volume,
                'avg_volume_24h': avg_volume_24h,
                'avg_volume_7d': avg_volume_7d,
                'volume_ratio_24h': current_volume / avg_volume_24h if avg_volume_24h > 0 else 1,
                'volume_trend': volume_trend
            }
        except Exception:
            return {}
    
    def _get_detailed_ml_analysis(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed ML analysis with explanations"""
        try:
            prediction = self.ml_pipeline.predict(symbol, data.tail(1))
            
            signal = prediction.get('signal', 0.5)
            confidence = prediction.get('confidence', 0.5)
            
            # Determine recommendation
            if signal >= 0.7:
                recommendation = 'Strong Buy'
            elif signal >= 0.6:
                recommendation = 'Buy'
            elif signal >= 0.4:
                recommendation = 'Hold'
            elif signal >= 0.3:
                recommendation = 'Sell'
            else:
                recommendation = 'Strong Sell'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'recommendation': recommendation,
                'explanation': f"ML models show {recommendation.lower()} signal with {confidence:.1%} confidence based on technical and market factors."
            }
        except Exception:
            return {
                'signal': 0.5,
                'confidence': 0.5,
                'recommendation': 'Hold',
                'explanation': 'Unable to generate ML prediction at this time.'
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line