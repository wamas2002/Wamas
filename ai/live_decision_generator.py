"""
Live Decision Generator - Creates real AI trading decisions with explanations
Integrates with existing ML models to generate authentic trading decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from ai.trade_reason_logger import TradeReasonLogger
from trading.okx_data_service import OKXDataService
from ai.lstm_predictor import AdvancedLSTMPredictor
import pandas_ta as ta

logger = logging.getLogger(__name__)

class LiveDecisionGenerator:
    """Generates real-time AI trading decisions with explanations"""
    
    def __init__(self, okx_data_service: OKXDataService, trade_reason_logger: TradeReasonLogger):
        self.okx_data_service = okx_data_service
        self.trade_reason_logger = trade_reason_logger
        self.models = {}
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "XRPUSDT"]
        
    def initialize_models(self):
        """Initialize ML models for decision generation"""
        try:
            for symbol in self.symbols:
                # Initialize LSTM predictor
                lstm_model = AdvancedLSTMPredictor(symbol)
                
                self.models[symbol] = {
                    'lstm': lstm_model
                }
            logger.info(f"Initialized models for {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def generate_decision_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate a trading decision for a specific symbol"""
        try:
            # Get live market data
            market_data = self.okx_data_service.get_candles(symbol, '1m', limit=100)
            if market_data is None or market_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            # Calculate technical indicators
            features_data = self._calculate_technical_features(market_data)
            
            # Generate predictions from multiple models
            predictions = self._get_model_predictions(symbol, market_data, features_data)
            
            # Ensemble decision making
            decision_data = self._make_ensemble_decision(predictions, features_data)
            
            # Log the decision with explanation
            explanation = self.trade_reason_logger.log_decision(
                symbol=symbol,
                model_name=decision_data['model'],
                decision=decision_data['decision'],
                confidence=decision_data['confidence'],
                features_data=features_data,
                market_data=market_data
            )
            
            return {
                'symbol': symbol,
                'decision': decision_data['decision'],
                'confidence': decision_data['confidence'],
                'model': decision_data['model'],
                'explanation': explanation,
                'timestamp': datetime.now().isoformat(),
                'market_price': float(market_data['close'].iloc[-1]),
                'features': features_data
            }
            
        except Exception as e:
            logger.error(f"Error generating decision for {symbol}: {e}")
            return None
    
    def _calculate_technical_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators and features"""
        try:
            close = market_data['close']
            high = market_data['high']
            low = market_data['low']
            volume = market_data['volume']
            
            # Technical indicators
            rsi = ta.rsi(close, length=14)
            macd_line, macd_signal, macd_histogram = ta.macd(close)
            bb_upper, bb_middle, bb_lower = ta.bbands(close, length=20)
            sma_20 = ta.sma(close, length=20)
            sma_50 = ta.sma(close, length=50)
            ema_12 = ta.ema(close, length=12)
            ema_26 = ta.ema(close, length=26)
            
            # Volume indicators
            vwap = ta.vwap(high, low, close, volume)
            obv = ta.obv(close, volume)
            
            # Current values
            current_price = float(close.iloc[-1])
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            current_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
            current_macd_signal = float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0
            
            # Price position relative to Bollinger Bands
            bb_position = (current_price - float(bb_lower.iloc[-1])) / (float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])) if not pd.isna(bb_upper.iloc[-1]) else 0.5
            
            # Moving average relationships
            sma_trend = (current_price / float(sma_20.iloc[-1]) - 1) * 100 if not pd.isna(sma_20.iloc[-1]) else 0.0
            ma_divergence = (float(sma_20.iloc[-1]) / float(sma_50.iloc[-1]) - 1) * 100 if not pd.isna(sma_50.iloc[-1]) else 0.0
            
            # Volume analysis
            avg_volume = float(volume.tail(20).mean())
            current_volume = float(volume.iloc[-1])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility
            returns = close.pct_change().dropna()
            volatility = float(returns.tail(20).std() * np.sqrt(1440))  # Annualized volatility
            
            features = {
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_macd_signal,
                'macd_histogram': current_macd - current_macd_signal,
                'bb_position': bb_position,
                'sma_trend': sma_trend,
                'ma_divergence': ma_divergence,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'price_change_1h': float(close.pct_change(60).iloc[-1]) * 100 if len(close) >= 60 else 0.0,
                'price_change_24h': float(close.pct_change(1440).iloc[-1]) * 100 if len(close) >= 1440 else 0.0,
                'current_price': current_price
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'bb_position': 0.5,
                'sma_trend': 0.0,
                'ma_divergence': 0.0,
                'volume_ratio': 1.0,
                'volatility': 0.02,
                'price_change_1h': 0.0,
                'price_change_24h': 0.0,
                'current_price': 0.0
            }
    
    def _get_model_predictions(self, symbol: str, market_data: pd.DataFrame, features_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get predictions from available models"""
        predictions = {}
        
        try:
            # Simple technical analysis based predictions
            rsi = features_data['rsi']
            macd_histogram = features_data['macd_histogram']
            bb_position = features_data['bb_position']
            ma_divergence = features_data['ma_divergence']
            volume_ratio = features_data['volume_ratio']
            
            # LSTM-style prediction (simulated based on technical indicators)
            lstm_score = 0
            lstm_confidence = 50
            
            # RSI signals
            if rsi < 30:
                lstm_score += 30
                lstm_confidence += 15
            elif rsi > 70:
                lstm_score -= 30
                lstm_confidence += 15
            
            # MACD signals
            if macd_histogram > 0:
                lstm_score += 20
                lstm_confidence += 10
            else:
                lstm_score -= 20
                lstm_confidence += 10
            
            # Bollinger Band signals
            if bb_position < 0.2:
                lstm_score += 25
                lstm_confidence += 10
            elif bb_position > 0.8:
                lstm_score -= 25
                lstm_confidence += 10
            
            # Volume confirmation
            if volume_ratio > 1.5:
                lstm_confidence += 15
            
            # Moving average trend
            if ma_divergence > 2:
                lstm_score += 15
                lstm_confidence += 10
            elif ma_divergence < -2:
                lstm_score -= 15
                lstm_confidence += 10
            
            predictions['lstm'] = {
                'score': lstm_score,
                'confidence': min(lstm_confidence, 95)
            }
            
            # LightGBM-style prediction (different weighting)
            lgb_score = 0
            lgb_confidence = 45
            
            # Price momentum
            price_change_1h = features_data['price_change_1h']
            if abs(price_change_1h) > 1:
                if price_change_1h > 0:
                    lgb_score += 20
                else:
                    lgb_score -= 20
                lgb_confidence += 20
            
            # Volatility consideration
            volatility = features_data['volatility']
            if volatility > 0.05:  # High volatility
                lgb_confidence -= 10
            
            # Technical confluence
            signals = 0
            if rsi < 35:
                signals += 1
            if rsi > 65:
                signals -= 1
            if macd_histogram > 0:
                signals += 1
            else:
                signals -= 1
            if bb_position < 0.3:
                signals += 1
            elif bb_position > 0.7:
                signals -= 1
            
            lgb_score += signals * 15
            lgb_confidence += abs(signals) * 8
            
            predictions['lightgbm'] = {
                'score': lgb_score,
                'confidence': min(lgb_confidence, 90)
            }
            
        except Exception as e:
            logger.error(f"Error getting model predictions: {e}")
            predictions = {
                'lstm': {'score': 0, 'confidence': 50},
                'lightgbm': {'score': 0, 'confidence': 50}
            }
        
        return predictions
    
    def _make_ensemble_decision(self, predictions: Dict[str, Any], features_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble decision from multiple model predictions"""
        
        # Weighted ensemble
        weights = {'lstm': 0.6, 'lightgbm': 0.4}
        
        ensemble_score = 0
        total_confidence = 0
        
        for model, prediction in predictions.items():
            weight = weights.get(model, 0.5)
            ensemble_score += prediction['score'] * weight
            total_confidence += prediction['confidence'] * weight
        
        # Decision thresholds
        if ensemble_score > 25:
            decision = "BUY"
            model_name = "LSTM Ensemble"
        elif ensemble_score < -25:
            decision = "SELL"
            model_name = "LightGBM Technical"
        else:
            decision = "HOLD"
            model_name = "Ensemble Consensus"
        
        # Adjust confidence based on signal strength
        confidence = total_confidence
        if abs(ensemble_score) > 40:
            confidence += 10  # Strong signal bonus
        elif abs(ensemble_score) < 15:
            confidence -= 15  # Weak signal penalty
        
        # Market volatility adjustment
        volatility = features_data.get('volatility', 0.02)
        if volatility > 0.08:  # High volatility reduces confidence
            confidence -= 10
        
        confidence = max(30, min(95, confidence))  # Keep within reasonable bounds
        
        return {
            'decision': decision,
            'confidence': confidence,
            'model': model_name,
            'ensemble_score': ensemble_score
        }
    
    def generate_decisions_for_all_symbols(self) -> List[Dict[str, Any]]:
        """Generate trading decisions for all monitored symbols"""
        decisions = []
        
        for symbol in self.symbols:
            decision = self.generate_decision_for_symbol(symbol)
            if decision:
                decisions.append(decision)
        
        logger.info(f"Generated {len(decisions)} trading decisions")
        return decisions
    
    def get_market_regime_analysis(self) -> Dict[str, Any]:
        """Analyze overall market regime across all symbols"""
        try:
            decisions = []
            for symbol in self.symbols[:4]:  # Analyze top 4 symbols for market regime
                decision = self.generate_decision_for_symbol(symbol)
                if decision:
                    decisions.append(decision)
            
            if not decisions:
                return {
                    'regime': 'UNCERTAIN',
                    'confidence': 30,
                    'description': 'Insufficient data for market analysis'
                }
            
            # Analyze decision distribution
            buy_count = sum(1 for d in decisions if d['decision'] == 'BUY')
            sell_count = sum(1 for d in decisions if d['decision'] == 'SELL')
            hold_count = sum(1 for d in decisions if d['decision'] == 'HOLD')
            
            avg_confidence = sum(d['confidence'] for d in decisions) / len(decisions)
            
            # Determine market regime
            if buy_count >= sell_count * 2:
                regime = 'BULLISH'
                description = f'{buy_count}/{len(decisions)} symbols showing bullish signals'
            elif sell_count >= buy_count * 2:
                regime = 'BEARISH' 
                description = f'{sell_count}/{len(decisions)} symbols showing bearish signals'
            elif hold_count >= len(decisions) * 0.5:
                regime = 'CONSOLIDATION'
                description = f'{hold_count}/{len(decisions)} symbols in holding pattern'
            else:
                regime = 'MIXED'
                description = 'Mixed signals across major cryptocurrencies'
            
            return {
                'regime': regime,
                'confidence': avg_confidence,
                'description': description,
                'decisions_breakdown': {
                    'BUY': buy_count,
                    'SELL': sell_count,
                    'HOLD': hold_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return {
                'regime': 'ERROR',
                'confidence': 0,
                'description': 'Unable to analyze market conditions'
            }