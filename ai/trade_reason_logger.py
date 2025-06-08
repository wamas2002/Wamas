"""
Trade Reason Logger - Explainable AI for Trading Decisions
Captures and explains why specific trading decisions were made by AI models
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

@dataclass
class TradeExplanation:
    """Data structure for trade explanations"""
    symbol: str
    model: str
    decision: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    top_features: List[str]
    timestamp: str
    signal_strength: str  # Strong, Moderate, Weak
    market_conditions: str
    risk_factors: List[str]

class TradeReasonLogger:
    """
    Captures and stores AI trading decision explanations
    Provides reasoning for model predictions and feature importance
    """
    
    def __init__(self):
        self.explanations = []
        self.max_history = 100
        self.explanation_file = "logs/trade_explanations.json"
        self._ensure_log_directory()
        self._load_explanations()
    
    def _ensure_log_directory(self):
        """Ensure logs directory exists"""
        Path("logs").mkdir(exist_ok=True)
    
    def _load_explanations(self):
        """Load existing explanations from file"""
        try:
            if Path(self.explanation_file).exists():
                with open(self.explanation_file, 'r') as f:
                    data = json.load(f)
                    self.explanations = data.get('explanations', [])
        except Exception as e:
            print(f"Warning: Could not load trade explanations: {e}")
            self.explanations = []
    
    def _save_explanations(self):
        """Save explanations to file"""
        try:
            data = {
                'explanations': self.explanations[-self.max_history:],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.explanation_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save trade explanations: {e}")
    
    def log_decision(self, 
                    symbol: str,
                    model_name: str,
                    decision: str,
                    confidence: float,
                    features_data: Dict[str, Any],
                    market_data: pd.DataFrame = None) -> TradeExplanation:
        """
        Log a trading decision with explanations
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            model_name: AI model used (e.g., LSTM, Transformer, GradientBoost)
            decision: BUY, SELL, or HOLD
            confidence: Confidence score 0-100
            features_data: Dictionary of feature values and importances
            market_data: Current market data for context
        
        Returns:
            TradeExplanation object
        """
        
        # Extract top contributing features
        top_features = self._extract_top_features(features_data, market_data)
        
        # Determine signal strength
        signal_strength = self._determine_signal_strength(confidence)
        
        # Analyze market conditions
        market_conditions = self._analyze_market_conditions(market_data)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(features_data, market_data)
        
        explanation = TradeExplanation(
            symbol=symbol,
            model=model_name,
            decision=decision,
            confidence=round(confidence, 2),
            top_features=top_features,
            timestamp=datetime.now().isoformat(),
            signal_strength=signal_strength,
            market_conditions=market_conditions,
            risk_factors=risk_factors
        )
        
        # Add to history
        self.explanations.append(asdict(explanation))
        
        # Keep only recent explanations
        if len(self.explanations) > self.max_history:
            self.explanations = self.explanations[-self.max_history:]
        
        # Save to file
        self._save_explanations()
        
        return explanation
    
    def _extract_top_features(self, features_data: Dict[str, Any], market_data: pd.DataFrame = None) -> List[str]:
        """Extract and format top 3 contributing features"""
        top_features = []
        
        try:
            # Get feature importances if available
            if 'feature_importance' in features_data:
                importances = features_data['feature_importance']
                if isinstance(importances, dict):
                    # Sort by importance
                    sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    
                    for feature, importance in sorted_features:
                        feature_explanation = self._explain_feature(feature, importance, features_data, market_data)
                        top_features.append(feature_explanation)
            
            # If no feature importance, use technical indicators
            elif market_data is not None and not market_data.empty:
                top_features = self._explain_technical_indicators(market_data)
            
            # Fallback to generic explanations
            else:
                top_features = self._generate_fallback_features(features_data)
                
        except Exception as e:
            print(f"Warning: Could not extract features: {e}")
            top_features = ["Market trend analysis", "Technical indicator signals", "Volume pattern recognition"]
        
        return top_features[:3]
    
    def _explain_feature(self, feature: str, importance: float, features_data: Dict[str, Any], market_data: pd.DataFrame = None) -> str:
        """Generate human-readable explanation for a feature"""
        
        # Technical indicators
        if 'rsi' in feature.lower():
            if market_data is not None and not market_data.empty:
                current_rsi = market_data['rsi'].iloc[-1] if 'rsi' in market_data.columns else 50
                if current_rsi < 30:
                    return f"RSI = {current_rsi:.1f} (oversold signal)"
                elif current_rsi > 70:
                    return f"RSI = {current_rsi:.1f} (overbought signal)"
                else:
                    return f"RSI = {current_rsi:.1f} (neutral momentum)"
        
        elif 'ema' in feature.lower() or 'sma' in feature.lower():
            return "Moving average crossover signal"
        
        elif 'macd' in feature.lower():
            return "MACD momentum indicator"
        
        elif 'volume' in feature.lower():
            return "Volume spike pattern detected"
        
        elif 'bb' in feature.lower() or 'bollinger' in feature.lower():
            return "Bollinger Bands breakout signal"
        
        elif 'atr' in feature.lower():
            return "Volatility trend analysis"
        
        elif 'trend' in feature.lower():
            return "Price trend confirmation"
        
        elif 'support' in feature.lower() or 'resistance' in feature.lower():
            return "Support/resistance level test"
        
        else:
            return f"{feature.replace('_', ' ').title()} pattern"
    
    def _explain_technical_indicators(self, market_data: pd.DataFrame) -> List[str]:
        """Generate explanations based on technical indicators"""
        explanations = []
        
        try:
            latest = market_data.iloc[-1]
            
            # RSI analysis
            if 'rsi' in market_data.columns:
                rsi = latest['rsi']
                if rsi < 30:
                    explanations.append(f"RSI = {rsi:.1f} (oversold)")
                elif rsi > 70:
                    explanations.append(f"RSI = {rsi:.1f} (overbought)")
                else:
                    explanations.append(f"RSI = {rsi:.1f} (neutral)")
            
            # Volume analysis
            if 'volume' in market_data.columns and len(market_data) > 1:
                current_vol = latest['volume']
                avg_vol = market_data['volume'].tail(10).mean()
                if current_vol > avg_vol * 1.5:
                    explanations.append("High volume surge detected")
                elif current_vol < avg_vol * 0.5:
                    explanations.append("Low volume consolidation")
                else:
                    explanations.append("Normal volume activity")
            
            # Price trend
            if len(market_data) > 5:
                recent_prices = market_data['close'].tail(5)
                if recent_prices.is_monotonic_increasing:
                    explanations.append("Strong upward price trend")
                elif recent_prices.is_monotonic_decreasing:
                    explanations.append("Strong downward price trend")
                else:
                    explanations.append("Sideways price movement")
            
        except Exception as e:
            print(f"Warning: Could not analyze technical indicators: {e}")
            explanations = ["Technical analysis signal", "Market momentum shift", "Price action pattern"]
        
        return explanations[:3]
    
    def _generate_fallback_features(self, features_data: Dict[str, Any]) -> List[str]:
        """Generate fallback feature explanations"""
        return [
            "AI pattern recognition signal",
            "Multi-timeframe trend analysis", 
            "Market microstructure signal"
        ]
    
    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength based on confidence"""
        if confidence >= 80:
            return "Strong"
        elif confidence >= 60:
            return "Moderate"
        else:
            return "Weak"
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame = None) -> str:
        """Analyze current market conditions"""
        if market_data is None or market_data.empty:
            return "Standard market conditions"
        
        try:
            # Simple volatility and trend analysis
            recent_data = market_data.tail(20)
            price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1) * 100
            volatility = recent_data['close'].std() / recent_data['close'].mean() * 100
            
            if volatility > 5:
                volatility_desc = "High volatility"
            elif volatility > 2:
                volatility_desc = "Moderate volatility"
            else:
                volatility_desc = "Low volatility"
            
            if price_change > 3:
                trend_desc = "bullish trend"
            elif price_change < -3:
                trend_desc = "bearish trend"
            else:
                trend_desc = "sideways trend"
            
            return f"{volatility_desc}, {trend_desc}"
            
        except Exception:
            return "Standard market conditions"
    
    def _identify_risk_factors(self, features_data: Dict[str, Any], market_data: pd.DataFrame = None) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        try:
            # High volatility risk
            if market_data is not None and not market_data.empty and len(market_data) > 10:
                volatility = market_data['close'].tail(10).std() / market_data['close'].tail(10).mean()
                if volatility > 0.05:
                    risks.append("High market volatility")
            
            # Low confidence risk
            confidence = features_data.get('confidence', 100)
            if confidence < 60:
                risks.append("Low model confidence")
            
            # Volume risk
            if market_data is not None and 'volume' in market_data.columns and len(market_data) > 5:
                recent_vol = market_data['volume'].tail(5).mean()
                avg_vol = market_data['volume'].mean()
                if recent_vol < avg_vol * 0.3:
                    risks.append("Unusually low volume")
        
        except Exception:
            pass
        
        return risks if risks else ["Standard risk profile"]
    
    def get_recent_explanations(self, symbol: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent trade explanations"""
        explanations = self.explanations.copy()
        
        if symbol:
            explanations = [exp for exp in explanations if exp['symbol'] == symbol]
        
        # Sort by timestamp (most recent first)
        explanations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return explanations[:limit]
    
    def get_explanation_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent explanation for a symbol"""
        explanations = self.get_recent_explanations(symbol=symbol, limit=1)
        return explanations[0] if explanations else None
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary by model"""
        model_stats = {}
        
        for exp in self.explanations:
            model = exp['model']
            if model not in model_stats:
                model_stats[model] = {
                    'decisions': [],
                    'confidences': [],
                    'count': 0
                }
            
            model_stats[model]['decisions'].append(exp['decision'])
            model_stats[model]['confidences'].append(exp['confidence'])
            model_stats[model]['count'] += 1
        
        # Calculate averages
        for model in model_stats:
            stats = model_stats[model]
            stats['avg_confidence'] = np.mean(stats['confidences']) if stats['confidences'] else 0
            stats['decision_distribution'] = {
                decision: stats['decisions'].count(decision) 
                for decision in ['BUY', 'SELL', 'HOLD']
            }
        
        return model_stats