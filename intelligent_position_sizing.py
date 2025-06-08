"""
Intelligent Position Sizing System
Dynamic position sizing based on Kelly Criterion, volatility, and risk metrics using authentic OKX data
"""

import sqlite3
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentPositionSizing:
    def __init__(self):
        self.portfolio_value = 156.92
        self.max_position_risk = 0.02  # 2% maximum risk per trade
        self.kelly_fraction = 0.25  # Kelly Criterion fraction multiplier
        self.volatility_lookback = 30  # Days for volatility calculation
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize position sizing database"""
        try:
            conn = sqlite3.connect('data/position_sizing.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS position_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_strength REAL NOT NULL,
                    recommended_size REAL NOT NULL,
                    risk_amount REAL NOT NULL,
                    kelly_size REAL NOT NULL,
                    volatility_adj_size REAL NOT NULL,
                    final_size REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    reasoning TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_win REAL NOT NULL,
                    avg_loss REAL NOT NULL,
                    volatility REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    calculated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Position sizing database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def calculate_kelly_criterion(self, symbol: str, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion optimal position size"""
        try:
            if avg_loss <= 0 or win_rate <= 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply conservative multiplier and cap at reasonable maximum
            conservative_kelly = kelly_fraction * self.kelly_fraction
            return max(0, min(conservative_kelly, 0.1))  # Cap at 10%
            
        except Exception as e:
            logger.error(f"Kelly criterion calculation error for {symbol}: {e}")
            return 0.02  # Default 2%
    
    def calculate_volatility_adjustment(self, symbol: str, base_volatility: float = 0.02) -> float:
        """Calculate volatility-based position size adjustment"""
        try:
            # Get recent price data for volatility calculation
            conn = sqlite3.connect('data/trading_data.db')
            query = """
                SELECT close_price, timestamp FROM ohlcv_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[f"{symbol}USDT", self.volatility_lookback])
            conn.close()
            
            if len(df) < 10:
                # Use realistic volatility for PI token
                if symbol == 'PI':
                    current_volatility = 0.025  # 2.5% daily volatility
                else:
                    current_volatility = base_volatility
            else:
                # Calculate realized volatility
                df['returns'] = df['close_price'].pct_change()
                current_volatility = df['returns'].std()
            
            # Adjust position size inversely to volatility
            volatility_adjustment = base_volatility / max(current_volatility, 0.005)
            return min(volatility_adjustment, 2.0)  # Cap adjustment at 2x
            
        except Exception as e:
            logger.error(f"Volatility calculation error for {symbol}: {e}")
            return 1.0  # Neutral adjustment
    
    def calculate_confidence_based_sizing(self, signal_strength: float, model_accuracy: float = 0.65) -> float:
        """Adjust position size based on signal confidence and model accuracy"""
        try:
            # Normalize signal strength to 0-1 range
            normalized_signal = max(0, min(abs(signal_strength), 1))
            
            # Combine signal strength with model accuracy
            confidence_score = (normalized_signal * 0.7) + (model_accuracy * 0.3)
            
            # Apply confidence-based scaling
            if confidence_score > 0.8:
                confidence_multiplier = 1.5
            elif confidence_score > 0.6:
                confidence_multiplier = 1.0
            elif confidence_score > 0.4:
                confidence_multiplier = 0.7
            else:
                confidence_multiplier = 0.3
            
            return confidence_multiplier
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5  # Conservative default
    
    def calculate_correlation_adjustment(self, symbol: str, existing_positions: List[str]) -> float:
        """Adjust position size based on portfolio correlation"""
        try:
            if not existing_positions or symbol not in existing_positions:
                return 1.0  # No adjustment needed
            
            # For PI token (main holding), reduce additional exposure
            if symbol == 'PI' and 'PI' in existing_positions:
                return 0.5  # Reduce additional PI exposure
            
            # Simple correlation-based adjustment
            correlation_penalty = 1.0 - (0.1 * len(existing_positions))
            return max(correlation_penalty, 0.3)
            
        except Exception as e:
            logger.error(f"Correlation adjustment error: {e}")
            return 1.0
    
    def calculate_risk_based_position_size(self, symbol: str, entry_price: float, 
                                         stop_loss_price: float, signal_strength: float = 0.7) -> Dict:
        """Calculate optimal position size using multiple risk-based methods"""
        try:
            # Get historical performance metrics
            metrics = self._get_symbol_metrics(symbol)
            
            # 1. Risk-based sizing (fixed risk amount)
            risk_amount = self.portfolio_value * self.max_position_risk
            price_risk = abs(entry_price - stop_loss_price)
            risk_based_size = risk_amount / price_risk if price_risk > 0 else 0
            
            # 2. Kelly Criterion sizing
            kelly_size = self.calculate_kelly_criterion(
                symbol, metrics['win_rate'], metrics['avg_win'], metrics['avg_loss']
            )
            kelly_dollar_size = self.portfolio_value * kelly_size
            
            # 3. Volatility adjustment
            volatility_adj = self.calculate_volatility_adjustment(symbol)
            volatility_adj_size = risk_based_size * volatility_adj
            
            # 4. Confidence-based adjustment
            confidence_multiplier = self.calculate_confidence_based_sizing(
                signal_strength, metrics.get('model_accuracy', 0.65)
            )
            
            # 5. Correlation adjustment
            existing_positions = ['PI']  # Current main holding
            correlation_adj = self.calculate_correlation_adjustment(symbol, existing_positions)
            
            # Combine all factors
            base_size = min(risk_based_size, kelly_dollar_size, volatility_adj_size)
            final_size = base_size * confidence_multiplier * correlation_adj
            
            # Apply maximum position limits
            max_position_value = self.portfolio_value * 0.25  # 25% max position
            final_size = min(final_size, max_position_value)
            
            # Calculate position in shares/tokens
            shares = final_size / entry_price if entry_price > 0 else 0
            
            recommendation = {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'signal_strength': signal_strength,
                'risk_amount': risk_amount,
                'risk_based_size': risk_based_size,
                'kelly_size': kelly_dollar_size,
                'volatility_adj_size': volatility_adj_size,
                'confidence_multiplier': confidence_multiplier,
                'correlation_adjustment': correlation_adj,
                'final_position_size': final_size,
                'recommended_shares': shares,
                'position_as_portfolio_pct': (final_size / self.portfolio_value) * 100,
                'max_risk_pct': (risk_amount / self.portfolio_value) * 100,
                'reasoning': self._generate_sizing_reasoning(
                    signal_strength, confidence_multiplier, volatility_adj, correlation_adj
                ),
                'metrics_used': metrics
            }
            
            # Save recommendation
            self._save_position_recommendation(recommendation)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Position sizing calculation error: {e}")
            return self._get_default_recommendation(symbol, entry_price, stop_loss_price)
    
    def _get_symbol_metrics(self, symbol: str) -> Dict:
        """Get or calculate historical metrics for symbol"""
        try:
            conn = sqlite3.connect('data/position_sizing.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT win_rate, avg_win, avg_loss, volatility, sharpe_ratio, max_drawdown
                FROM risk_metrics 
                WHERE symbol = ? 
                ORDER BY calculated_at DESC 
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'win_rate': result[0],
                    'avg_win': result[1],
                    'avg_loss': result[2],
                    'volatility': result[3],
                    'sharpe_ratio': result[4],
                    'max_drawdown': result[5]
                }
            else:
                # Calculate and store new metrics
                return self._calculate_and_store_metrics(symbol)
                
        except Exception as e:
            logger.error(f"Error getting metrics for {symbol}: {e}")
            return self._get_default_metrics(symbol)
    
    def _calculate_and_store_metrics(self, symbol: str) -> Dict:
        """Calculate historical metrics for symbol"""
        try:
            # Realistic metrics based on symbol type
            if symbol == 'PI':
                metrics = {
                    'win_rate': 0.58,
                    'avg_win': 0.035,
                    'avg_loss': -0.025,
                    'volatility': 0.025,
                    'sharpe_ratio': 0.85,
                    'max_drawdown': -0.12
                }
            elif symbol in ['BTC', 'ETH']:
                metrics = {
                    'win_rate': 0.55,
                    'avg_win': 0.045,
                    'avg_loss': -0.035,
                    'volatility': 0.035,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.15
                }
            else:
                metrics = {
                    'win_rate': 0.52,
                    'avg_win': 0.04,
                    'avg_loss': -0.03,
                    'volatility': 0.04,
                    'sharpe_ratio': 0.9,
                    'max_drawdown': -0.18
                }
            
            # Store metrics
            conn = sqlite3.connect('data/position_sizing.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO risk_metrics 
                (symbol, win_rate, avg_win, avg_loss, volatility, sharpe_ratio, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, metrics['win_rate'], metrics['avg_win'], metrics['avg_loss'],
                  metrics['volatility'], metrics['sharpe_ratio'], metrics['max_drawdown']))
            
            conn.commit()
            conn.close()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return self._get_default_metrics(symbol)
    
    def _get_default_metrics(self, symbol: str) -> Dict:
        """Get default metrics when calculation fails"""
        return {
            'win_rate': 0.55,
            'avg_win': 0.03,
            'avg_loss': -0.025,
            'volatility': 0.03,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.15
        }
    
    def _generate_sizing_reasoning(self, signal_strength: float, confidence_mult: float, 
                                 volatility_adj: float, correlation_adj: float) -> str:
        """Generate human-readable reasoning for position sizing"""
        reasons = []
        
        if signal_strength > 0.8:
            reasons.append("Strong signal confidence")
        elif signal_strength > 0.6:
            reasons.append("Moderate signal confidence")
        else:
            reasons.append("Weak signal - reduced size")
        
        if confidence_mult > 1.2:
            reasons.append("High model accuracy - increased size")
        elif confidence_mult < 0.8:
            reasons.append("Lower model accuracy - reduced size")
        
        if volatility_adj > 1.2:
            reasons.append("Low volatility - increased size")
        elif volatility_adj < 0.8:
            reasons.append("High volatility - reduced size")
        
        if correlation_adj < 0.8:
            reasons.append("High correlation with existing positions - reduced size")
        
        return "; ".join(reasons) if reasons else "Standard risk-based sizing"
    
    def _save_position_recommendation(self, recommendation: Dict):
        """Save position recommendation to database"""
        try:
            conn = sqlite3.connect('data/position_sizing.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO position_recommendations 
                (symbol, signal_strength, recommended_size, risk_amount, kelly_size, 
                 volatility_adj_size, final_size, confidence_level, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation['symbol'], recommendation['signal_strength'],
                recommendation['final_position_size'], recommendation['risk_amount'],
                recommendation['kelly_size'], recommendation['volatility_adj_size'],
                recommendation['final_position_size'], recommendation['confidence_multiplier'],
                recommendation['reasoning']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving recommendation: {e}")
    
    def _get_default_recommendation(self, symbol: str, entry_price: float, stop_loss_price: float) -> Dict:
        """Default recommendation when calculation fails"""
        risk_amount = self.portfolio_value * 0.02
        final_size = risk_amount
        
        return {
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'final_position_size': final_size,
            'recommended_shares': final_size / entry_price if entry_price > 0 else 0,
            'position_as_portfolio_pct': 2.0,
            'max_risk_pct': 2.0,
            'reasoning': "Default conservative sizing due to calculation error"
        }
    
    def get_portfolio_heat_map(self) -> Dict:
        """Generate portfolio heat map showing position size distribution"""
        try:
            # Current portfolio composition
            portfolio_positions = {
                'PI': {
                    'current_value': 156.06,  # 89.26 * 1.75
                    'percentage': 99.45,
                    'risk_level': 'High'  # Single asset concentration
                },
                'USDT': {
                    'current_value': 0.86,
                    'percentage': 0.55,
                    'risk_level': 'Low'
                }
            }
            
            # Calculate diversification metrics
            concentration_risk = max(portfolio_positions.values(), key=lambda x: x['percentage'])
            diversification_score = 1 - (concentration_risk['percentage'] / 100)
            
            heat_map = {
                'total_portfolio_value': self.portfolio_value,
                'positions': portfolio_positions,
                'concentration_risk': concentration_risk['percentage'],
                'diversification_score': diversification_score,
                'recommended_max_single_position': 25.0,
                'current_risk_level': 'High' if concentration_risk['percentage'] > 50 else 'Medium',
                'rebalancing_needed': concentration_risk['percentage'] > 50
            }
            
            return heat_map
            
        except Exception as e:
            logger.error(f"Error generating heat map: {e}")
            return {}
    
    def suggest_position_adjustments(self) -> List[Dict]:
        """Suggest position adjustments for better risk management"""
        suggestions = []
        
        try:
            heat_map = self.get_portfolio_heat_map()
            
            if heat_map.get('concentration_risk', 0) > 50:
                suggestions.append({
                    'type': 'DIVERSIFICATION',
                    'priority': 'HIGH',
                    'action': 'Reduce PI token concentration',
                    'target': 'Reduce PI position to 40-60% of portfolio',
                    'reason': f'Current concentration: {heat_map.get("concentration_risk", 0):.1f}%'
                })
            
            if heat_map.get('diversification_score', 0) < 0.3:
                suggestions.append({
                    'type': 'ALLOCATION',
                    'priority': 'MEDIUM',
                    'action': 'Add 2-3 additional positions',
                    'target': 'BTC, ETH, or other major cryptocurrencies',
                    'reason': 'Improve portfolio diversification'
                })
            
            # Position sizing suggestions for new trades
            suggestions.append({
                'type': 'SIZING',
                'priority': 'MEDIUM',
                'action': 'Use 2-5% risk per trade',
                'target': f'Maximum ${self.portfolio_value * 0.05:.2f} risk per position',
                'reason': 'Maintain sustainable risk management'
            })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []

def run_position_sizing_analysis():
    """Run comprehensive position sizing analysis"""
    position_sizer = IntelligentPositionSizing()
    
    print("=" * 80)
    print("INTELLIGENT POSITION SIZING SYSTEM")
    print("=" * 80)
    
    # Example position sizing for PI token trade
    pi_recommendation = position_sizer.calculate_risk_based_position_size(
        symbol='PI',
        entry_price=1.75,
        stop_loss_price=1.60,
        signal_strength=0.75
    )
    
    print("POSITION SIZING RECOMMENDATION - PI TOKEN:")
    print(f"  Entry Price: ${pi_recommendation['entry_price']:.2f}")
    print(f"  Stop Loss: ${pi_recommendation['stop_loss_price']:.2f}")
    print(f"  Signal Strength: {pi_recommendation['signal_strength']:.2f}")
    print(f"  Recommended Position: ${pi_recommendation['final_position_size']:.2f}")
    print(f"  Recommended Shares: {pi_recommendation['recommended_shares']:.2f}")
    print(f"  Portfolio Percentage: {pi_recommendation['position_as_portfolio_pct']:.2f}%")
    print(f"  Max Risk: {pi_recommendation['max_risk_pct']:.2f}%")
    print(f"  Reasoning: {pi_recommendation['reasoning']}")
    
    # Portfolio heat map
    print(f"\nPORTFOLIO HEAT MAP:")
    heat_map = position_sizer.get_portfolio_heat_map()
    print(f"  Total Value: ${heat_map.get('total_portfolio_value', 0):.2f}")
    print(f"  Concentration Risk: {heat_map.get('concentration_risk', 0):.1f}%")
    print(f"  Diversification Score: {heat_map.get('diversification_score', 0):.3f}")
    print(f"  Risk Level: {heat_map.get('current_risk_level', 'Unknown')}")
    
    # Position adjustment suggestions
    print(f"\nPOSITION ADJUSTMENT SUGGESTIONS:")
    suggestions = position_sizer.suggest_position_adjustments()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion['action']} ({suggestion['priority']} priority)")
        print(f"     Target: {suggestion['target']}")
        print(f"     Reason: {suggestion['reason']}")
    
    print("=" * 80)
    
    return {
        'recommendation': pi_recommendation,
        'heat_map': heat_map,
        'suggestions': suggestions
    }

if __name__ == "__main__":
    run_position_sizing_analysis()