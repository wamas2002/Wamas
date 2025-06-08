"""
AI Financial Advisor Module
Provides Buy/Hold/Sell recommendations with AI-generated explanations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import sqlite3
from trading.okx_data_service import OKXDataService
from ai.comprehensive_ml_pipeline import TradingMLPipeline
from ai.market_sentiment_analyzer import MarketSentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')

class AIFinancialAdvisor:
    """AI Financial Advisor providing trading recommendations with explanations"""
    
    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self.okx_data_service = OKXDataService()
        self.ml_pipeline = TradingMLPipeline()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
        # Signal strength thresholds
        self.buy_threshold = 0.7
        self.sell_threshold = 0.3
        
    def get_recommendations(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get AI recommendations for multiple symbols"""
        recommendations = {}
        
        for symbol in symbols:
            try:
                recommendation = self._analyze_symbol(symbol)
                if recommendation['confidence'] >= self.confidence_threshold:
                    recommendations[symbol] = recommendation
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
                
        return recommendations
    
    def _analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a single symbol and generate recommendation"""
        # Get current market data
        current_data = self.okx_data_service.get_historical_data(symbol, '1h', limit=100)
        
        if current_data.empty:
            return self._create_no_data_response(symbol)
        
        # Get ML prediction
        ml_prediction = self.ml_pipeline.predict(symbol, current_data.tail(1))
        
        # Get sentiment analysis
        sentiment_score = self._get_sentiment_score(symbol)
        
        # Generate technical analysis
        technical_analysis = self._generate_technical_analysis(current_data)
        
        # Combine signals
        combined_signal = self._combine_signals(ml_prediction, sentiment_score, technical_analysis)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(combined_signal)
        
        # Generate explanation
        explanation = self._generate_explanation(symbol, ml_prediction, sentiment_score, technical_analysis, recommendation)
        
        return {
            'symbol': symbol,
            'recommendation': recommendation['action'],
            'confidence': recommendation['confidence'],
            'explanation': explanation,
            'ml_signal': ml_prediction.get('signal', 0),
            'sentiment_score': sentiment_score,
            'technical_score': technical_analysis['score'],
            'price': current_data['close'].iloc[-1] if not current_data.empty else 0,
            'timestamp': datetime.now()
        }
    
    def _get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score for symbol"""
        try:
            sentiment_data = self.sentiment_analyzer.get_symbol_sentiment(symbol)
            return sentiment_data.get('overall_sentiment', 0.5)
        except:
            return 0.5  # Neutral sentiment as fallback
    
    def _generate_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate technical analysis indicators"""
        if len(data) < 20:
            return {'score': 0.5, 'indicators': {}}
        
        try:
            # Calculate technical indicators
            indicators = {}
            
            # RSI
            rsi = self._calculate_rsi(data['close'])
            indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
            
            # Moving averages
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean() if len(data) >= 50 else sma_20
            indicators['sma_20'] = sma_20.iloc[-1] if not sma_20.empty else data['close'].iloc[-1]
            indicators['sma_50'] = sma_50.iloc[-1] if not sma_50.empty else data['close'].iloc[-1]
            
            # Price position relative to moving averages
            current_price = data['close'].iloc[-1]
            indicators['price_vs_sma20'] = (current_price - indicators['sma_20']) / indicators['sma_20']
            indicators['price_vs_sma50'] = (current_price - indicators['sma_50']) / indicators['sma_50']
            
            # Volume trend
            volume_sma = data['volume'].rolling(20).mean()
            indicators['volume_trend'] = (data['volume'].iloc[-1] - volume_sma.iloc[-1]) / volume_sma.iloc[-1] if not volume_sma.empty else 0
            
            # Calculate overall technical score
            score = self._calculate_technical_score(indicators)
            
            return {
                'score': score,
                'indicators': indicators
            }
        except Exception as e:
            return {'score': 0.5, 'indicators': {}}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_technical_score(self, indicators: Dict[str, float]) -> float:
        """Calculate overall technical score from indicators"""
        score = 0.5  # Start neutral
        
        try:
            # RSI scoring
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                score += 0.2  # Oversold - bullish
            elif rsi > 70:
                score -= 0.2  # Overbought - bearish
            
            # Price vs moving averages
            price_vs_sma20 = indicators.get('price_vs_sma20', 0)
            price_vs_sma50 = indicators.get('price_vs_sma50', 0)
            
            if price_vs_sma20 > 0.02:  # 2% above SMA20
                score += 0.15
            elif price_vs_sma20 < -0.02:  # 2% below SMA20
                score -= 0.15
            
            if price_vs_sma50 > 0.05:  # 5% above SMA50
                score += 0.1
            elif price_vs_sma50 < -0.05:  # 5% below SMA50
                score -= 0.1
            
            # Volume trend
            volume_trend = indicators.get('volume_trend', 0)
            if volume_trend > 0.5:  # High volume
                score += 0.05
            
            # Ensure score stays within bounds
            score = max(0, min(1, score))
            
        except Exception:
            score = 0.5
        
        return score
    
    def _combine_signals(self, ml_prediction: Dict[str, Any], sentiment_score: float, technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all signals into a unified signal"""
        try:
            # Weight the different signals
            ml_weight = 0.5
            sentiment_weight = 0.2
            technical_weight = 0.3
            
            # Get individual signals
            ml_signal = ml_prediction.get('signal', 0.5)
            ml_confidence = ml_prediction.get('confidence', 0.5)
            technical_signal = technical_analysis.get('score', 0.5)
            
            # Calculate weighted signal
            combined_signal = (
                ml_signal * ml_weight +
                sentiment_score * sentiment_weight +
                technical_signal * technical_weight
            )
            
            # Calculate combined confidence
            combined_confidence = (
                ml_confidence * ml_weight +
                0.7 * sentiment_weight +  # Medium confidence for sentiment
                0.6 * technical_weight    # Medium confidence for technical
            )
            
            return {
                'signal': combined_signal,
                'confidence': combined_confidence,
                'components': {
                    'ml': ml_signal,
                    'sentiment': sentiment_score,
                    'technical': technical_signal
                }
            }
        except Exception:
            return {
                'signal': 0.5,
                'confidence': 0.4,
                'components': {'ml': 0.5, 'sentiment': 0.5, 'technical': 0.5}
            }
    
    def _generate_recommendation(self, combined_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate buy/hold/sell recommendation"""
        signal = combined_signal['signal']
        confidence = combined_signal['confidence']
        
        if signal >= self.buy_threshold:
            action = 'BUY'
        elif signal <= self.sell_threshold:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'confidence': confidence,
            'signal_strength': signal
        }
    
    def _generate_explanation(self, symbol: str, ml_prediction: Dict[str, Any], 
                            sentiment_score: float, technical_analysis: Dict[str, Any], 
                            recommendation: Dict[str, Any]) -> str:
        """Generate AI explanation for the recommendation"""
        action = recommendation['action']
        confidence = recommendation['confidence']
        
        # Start with action and confidence
        explanation = f"{action} recommendation with {confidence:.1%} confidence. "
        
        # Add ML analysis
        ml_signal = ml_prediction.get('signal', 0.5)
        if ml_signal > 0.6:
            explanation += "ML models indicate strong bullish momentum. "
        elif ml_signal < 0.4:
            explanation += "ML models suggest bearish pressure. "
        else:
            explanation += "ML models show neutral signals. "
        
        # Add sentiment analysis
        if sentiment_score > 0.6:
            explanation += "Market sentiment is positive. "
        elif sentiment_score < 0.4:
            explanation += "Market sentiment is negative. "
        else:
            explanation += "Market sentiment is neutral. "
        
        # Add technical analysis
        indicators = technical_analysis.get('indicators', {})
        rsi = indicators.get('rsi', 50)
        
        if rsi > 70:
            explanation += "RSI indicates overbought conditions. "
        elif rsi < 30:
            explanation += "RSI shows oversold levels. "
        
        price_vs_sma20 = indicators.get('price_vs_sma20', 0)
        if price_vs_sma20 > 0.02:
            explanation += "Price is trending above key moving averages. "
        elif price_vs_sma20 < -0.02:
            explanation += "Price is below key support levels. "
        
        # Add risk warning for high-confidence signals
        if confidence > 0.8:
            explanation += "High confidence signal - consider position sizing. "
        elif confidence < 0.5:
            explanation += "Lower confidence - use caution and smaller position size. "
        
        return explanation.strip()
    
    def _create_no_data_response(self, symbol: str) -> Dict[str, Any]:
        """Create response when no data is available"""
        return {
            'symbol': symbol,
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'explanation': f"Insufficient data available for {symbol} analysis. Recommend HOLD until more data is available.",
            'ml_signal': 0.5,
            'sentiment_score': 0.5,
            'technical_score': 0.5,
            'price': 0,
            'timestamp': datetime.now()
        }
    
    def get_top_recommendations(self, symbols: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top recommendations sorted by confidence"""
        recommendations = self.get_recommendations(symbols)
        
        # Convert to list and sort by confidence
        rec_list = list(recommendations.values())
        rec_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        return rec_list[:limit]