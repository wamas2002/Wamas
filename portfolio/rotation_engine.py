"""
Portfolio Rotation Engine - Dynamic Asset Allocation Based on Performance
Ranks pairs by performance metrics and reallocates capital to top performers
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import ccxt
import os

logger = logging.getLogger(__name__)

class PortfolioRotationEngine:
    """Intelligent portfolio rotation based on performance analytics"""
    
    def __init__(self):
        self.exchange = None
        self.rotation_frequency_days = 7  # Weekly rotation
        self.top_performers_count = 10
        self.bottom_performers_count = 10
        self.min_performance_periods = 20  # Minimum trades for evaluation
        
        # Performance tracking
        self.symbol_performance = {}
        self.rotation_history = []
        
        self.setup_database()
        self.initialize_exchange()
    
    def setup_database(self):
        """Initialize portfolio rotation database"""
        try:
            conn = sqlite3.connect('portfolio_rotation.db')
            cursor = conn.cursor()
            
            # Symbol performance metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS symbol_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    evaluation_date DATE,
                    
                    -- Performance metrics
                    total_trades INTEGER,
                    win_rate REAL,
                    avg_pnl_pct REAL,
                    total_pnl_pct REAL,
                    sharpe_ratio REAL,
                    max_drawdown_pct REAL,
                    
                    -- Risk metrics
                    volatility REAL,
                    beta REAL,
                    var_95 REAL,
                    
                    -- Ranking
                    performance_score REAL,
                    rank_position INTEGER,
                    
                    -- Status
                    allocation_tier TEXT,
                    recommended_allocation_pct REAL
                )
            ''')
            
            # Rotation decisions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rotation_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rotation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Portfolio changes
                    symbols_added TEXT,
                    symbols_removed TEXT,
                    allocation_changes TEXT,
                    
                    -- Metrics
                    portfolio_performance REAL,
                    top_performer_avg REAL,
                    bottom_performer_avg REAL,
                    
                    -- Reasoning
                    rotation_reason TEXT,
                    expected_impact TEXT
                )
            ''')
            
            # Allocation history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS allocation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    old_allocation_pct REAL,
                    new_allocation_pct REAL,
                    allocation_change_pct REAL,
                    reason TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Portfolio rotation database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            self.exchange.load_markets()
            logger.info("Portfolio rotation engine connected to OKX")
            
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
    
    def calculate_symbol_performance(self, symbol: str, trade_history: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics for a symbol"""
        try:
            if len(trade_history) < 5:
                return None
            
            # Filter trades for this symbol
            symbol_trades = [t for t in trade_history if t.get('symbol') == symbol]
            
            if len(symbol_trades) < 5:
                return None
            
            # Basic performance metrics
            total_trades = len(symbol_trades)
            wins = len([t for t in symbol_trades if t.get('win_loss') == 'win'])
            win_rate = wins / total_trades
            
            # PnL calculations
            pnl_values = [t.get('pnl_pct', 0) for t in symbol_trades]
            avg_pnl_pct = np.mean(pnl_values)
            total_pnl_pct = sum(pnl_values)
            
            # Risk metrics
            volatility = np.std(pnl_values) if len(pnl_values) > 1 else 0
            
            # Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = avg_pnl_pct / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod([1 + (pnl/100) for pnl in pnl_values])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown_pct = abs(np.min(drawdowns)) * 100
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(pnl_values, 5) if len(pnl_values) >= 10 else avg_pnl_pct
            
            # Performance score calculation (weighted composite)
            performance_score = (
                win_rate * 30 +                    # 30% weight on win rate
                (avg_pnl_pct * 2) * 25 +          # 25% weight on avg PnL
                sharpe_ratio * 20 +               # 20% weight on Sharpe ratio
                (1 - max_drawdown_pct/100) * 15 + # 15% weight on drawdown control
                (1 - volatility/10) * 10          # 10% weight on low volatility
            )
            
            performance_data = {
                'symbol': symbol,
                'evaluation_date': datetime.now().date(),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_pnl_pct': avg_pnl_pct,
                'total_pnl_pct': total_pnl_pct,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown_pct,
                'volatility': volatility,
                'var_95': var_95,
                'performance_score': performance_score
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Performance calculation failed for {symbol}: {e}")
            return None
    
    def evaluate_all_symbols(self, all_trade_history: List[Dict]) -> List[Dict]:
        """Evaluate performance for all traded symbols"""
        try:
            # Get unique symbols from trade history
            symbols = list(set([t.get('symbol') for t in all_trade_history if t.get('symbol')]))
            
            symbol_performances = []
            
            for symbol in symbols:
                performance = self.calculate_symbol_performance(symbol, all_trade_history)
                if performance:
                    symbol_performances.append(performance)
            
            # Sort by performance score
            symbol_performances.sort(key=lambda x: x['performance_score'], reverse=True)
            
            # Add rankings
            for i, perf in enumerate(symbol_performances):
                perf['rank_position'] = i + 1
                
                # Assign allocation tiers
                if i < self.top_performers_count:
                    perf['allocation_tier'] = 'high'
                    perf['recommended_allocation_pct'] = 3.0  # 3% per top performer
                elif i < len(symbol_performances) - self.bottom_performers_count:
                    perf['allocation_tier'] = 'medium'
                    perf['recommended_allocation_pct'] = 1.5  # 1.5% per medium performer
                else:
                    perf['allocation_tier'] = 'low'
                    perf['recommended_allocation_pct'] = 0.5  # 0.5% per low performer
            
            return symbol_performances
            
        except Exception as e:
            logger.error(f"Symbol evaluation failed: {e}")
            return []
    
    def generate_rotation_recommendations(self, current_allocations: Dict[str, float], 
                                        performance_rankings: List[Dict]) -> Dict:
        """Generate portfolio rotation recommendations"""
        try:
            recommendations = {
                'timestamp': datetime.now(),
                'rotation_needed': False,
                'changes': [],
                'expected_impact': {},
                'reasoning': []
            }
            
            # Create allocation maps
            current_symbols = set(current_allocations.keys())
            top_performers = [p['symbol'] for p in performance_rankings[:self.top_performers_count]]
            bottom_performers = [p['symbol'] for p in performance_rankings[-self.bottom_performers_count:]]
            
            # Symbols to add (top performers not in current portfolio)
            symbols_to_add = [s for s in top_performers if s not in current_symbols]
            
            # Symbols to reduce/remove (bottom performers in current portfolio)
            symbols_to_reduce = [s for s in bottom_performers if s in current_symbols]
            
            # Check if rotation is needed
            if symbols_to_add or symbols_to_reduce:
                recommendations['rotation_needed'] = True
            
            # Generate specific recommendations
            for symbol in symbols_to_add:
                perf_data = next((p for p in performance_rankings if p['symbol'] == symbol), None)
                if perf_data:
                    recommendations['changes'].append({
                        'action': 'add',
                        'symbol': symbol,
                        'current_allocation': 0,
                        'recommended_allocation': perf_data['recommended_allocation_pct'],
                        'reason': f"Top performer (rank #{perf_data['rank_position']}, score: {perf_data['performance_score']:.1f})"
                    })
            
            for symbol in symbols_to_reduce:
                current_alloc = current_allocations.get(symbol, 0)
                perf_data = next((p for p in performance_rankings if p['symbol'] == symbol), None)
                
                if perf_data:
                    new_alloc = perf_data['recommended_allocation_pct']
                    
                    if new_alloc < current_alloc:
                        recommendations['changes'].append({
                            'action': 'reduce',
                            'symbol': symbol,
                            'current_allocation': current_alloc,
                            'recommended_allocation': new_alloc,
                            'reason': f"Underperformer (rank #{perf_data['rank_position']}, score: {perf_data['performance_score']:.1f})"
                        })
            
            # Calculate expected impact
            if recommendations['rotation_needed']:
                top_avg_score = np.mean([p['performance_score'] for p in performance_rankings[:10]])
                bottom_avg_score = np.mean([p['performance_score'] for p in performance_rankings[-10:]])
                
                recommendations['expected_impact'] = {
                    'top_performer_avg_score': top_avg_score,
                    'bottom_performer_avg_score': bottom_avg_score,
                    'score_improvement': top_avg_score - bottom_avg_score,
                    'expected_performance_lift': f"{((top_avg_score - bottom_avg_score) / bottom_avg_score * 100):.1f}%"
                }
                
                recommendations['reasoning'] = [
                    f"Rotating to focus on top {self.top_performers_count} performers",
                    f"Reducing allocation to bottom {self.bottom_performers_count} performers",
                    f"Expected performance improvement: {recommendations['expected_impact']['expected_performance_lift']}"
                ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Rotation recommendations failed: {e}")
            return {'rotation_needed': False, 'error': str(e)}
    
    def execute_rotation(self, recommendations: Dict) -> Dict:
        """Execute portfolio rotation based on recommendations"""
        try:
            if not recommendations.get('rotation_needed'):
                return {'success': True, 'message': 'No rotation needed'}
            
            executed_changes = []
            
            for change in recommendations.get('changes', []):
                try:
                    # Log allocation change
                    self.log_allocation_change(
                        change['symbol'],
                        change['current_allocation'],
                        change['recommended_allocation'],
                        change['reason']
                    )
                    
                    executed_changes.append(change)
                    
                except Exception as e:
                    logger.error(f"Failed to execute change for {change['symbol']}: {e}")
            
            # Log rotation decision
            self.log_rotation_decision(recommendations, executed_changes)
            
            return {
                'success': True,
                'executed_changes': len(executed_changes),
                'total_changes': len(recommendations.get('changes', [])),
                'message': f"Rotation executed: {len(executed_changes)} changes applied"
            }
            
        except Exception as e:
            logger.error(f"Rotation execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def log_allocation_change(self, symbol: str, old_allocation: float, 
                            new_allocation: float, reason: str):
        """Log allocation change to database"""
        try:
            conn = sqlite3.connect('portfolio_rotation.db')
            cursor = conn.cursor()
            
            allocation_change = new_allocation - old_allocation
            
            cursor.execute('''
                INSERT INTO allocation_history (
                    symbol, old_allocation_pct, new_allocation_pct, 
                    allocation_change_pct, reason
                ) VALUES (?, ?, ?, ?, ?)
            ''', (symbol, old_allocation, new_allocation, allocation_change, reason))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log allocation change: {e}")
    
    def log_rotation_decision(self, recommendations: Dict, executed_changes: List[Dict]):
        """Log rotation decision to database"""
        try:
            conn = sqlite3.connect('portfolio_rotation.db')
            cursor = conn.cursor()
            
            symbols_added = [c['symbol'] for c in executed_changes if c['action'] == 'add']
            symbols_removed = [c['symbol'] for c in executed_changes if c['action'] == 'reduce' and c['recommended_allocation'] == 0]
            
            expected_impact = recommendations.get('expected_impact', {})
            
            cursor.execute('''
                INSERT INTO rotation_decisions (
                    symbols_added, symbols_removed, allocation_changes,
                    top_performer_avg, bottom_performer_avg, rotation_reason, expected_impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                json.dumps(symbols_added),
                json.dumps(symbols_removed),
                json.dumps(executed_changes),
                expected_impact.get('top_performer_avg_score', 0),
                expected_impact.get('bottom_performer_avg_score', 0),
                '; '.join(recommendations.get('reasoning', [])),
                json.dumps(expected_impact)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log rotation decision: {e}")
    
    def save_symbol_performance(self, performance_data: List[Dict]):
        """Save symbol performance data to database"""
        try:
            conn = sqlite3.connect('portfolio_rotation.db')
            cursor = conn.cursor()
            
            for perf in performance_data:
                cursor.execute('''
                    INSERT INTO symbol_performance (
                        symbol, evaluation_date, total_trades, win_rate, avg_pnl_pct,
                        total_pnl_pct, sharpe_ratio, max_drawdown_pct, volatility,
                        var_95, performance_score, rank_position, allocation_tier,
                        recommended_allocation_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    perf['symbol'], perf['evaluation_date'], perf['total_trades'],
                    perf['win_rate'], perf['avg_pnl_pct'], perf['total_pnl_pct'],
                    perf['sharpe_ratio'], perf['max_drawdown_pct'], perf['volatility'],
                    perf['var_95'], perf['performance_score'], perf['rank_position'],
                    perf['allocation_tier'], perf['recommended_allocation_pct']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def get_current_portfolio_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation from exchange"""
        try:
            balance = self.exchange.fetch_balance()
            total_value = balance['USDT']['total']
            
            allocations = {}
            
            # Calculate allocation percentages
            for symbol, amount in balance.items():
                if symbol != 'USDT' and amount['total'] > 0:
                    # Get current price
                    ticker = self.exchange.fetch_ticker(f"{symbol}/USDT")
                    value = amount['total'] * ticker['last']
                    allocation_pct = (value / total_value) * 100
                    
                    if allocation_pct > 0.1:  # Only include significant allocations
                        allocations[f"{symbol}/USDT"] = allocation_pct
            
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to get current allocation: {e}")
            return {}
    
    def run_rotation_analysis(self, trade_history: List[Dict]) -> Dict:
        """Run complete rotation analysis and generate recommendations"""
        try:
            logger.info("Starting portfolio rotation analysis...")
            
            # Evaluate all symbols
            performance_rankings = self.evaluate_all_symbols(trade_history)
            
            if not performance_rankings:
                return {
                    'success': False,
                    'message': 'Insufficient data for rotation analysis'
                }
            
            # Save performance data
            self.save_symbol_performance(performance_rankings)
            
            # Get current allocations
            current_allocations = self.get_current_portfolio_allocation()
            
            # Generate recommendations
            recommendations = self.generate_rotation_recommendations(
                current_allocations, performance_rankings
            )
            
            # Execute rotation if needed
            execution_result = self.execute_rotation(recommendations)
            
            result = {
                'success': True,
                'analysis_timestamp': datetime.now().isoformat(),
                'symbols_evaluated': len(performance_rankings),
                'top_performers': performance_rankings[:5],
                'bottom_performers': performance_rankings[-5:],
                'recommendations': recommendations,
                'execution_result': execution_result
            }
            
            logger.info(f"Rotation analysis complete: {len(performance_rankings)} symbols evaluated")
            
            return result
            
        except Exception as e:
            logger.error(f"Rotation analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_rotation_insights(self) -> List[str]:
        """Get insights about portfolio rotation"""
        try:
            insights = []
            
            # Get recent rotation data
            conn = sqlite3.connect('portfolio_rotation.db')
            
            # Recent performance data
            recent_performance = pd.read_sql_query('''
                SELECT COUNT(DISTINCT symbol) as symbols_tracked,
                       AVG(performance_score) as avg_score,
                       MAX(performance_score) as best_score,
                       MIN(performance_score) as worst_score
                FROM symbol_performance 
                WHERE evaluation_date > date('now', '-7 days')
            ''', conn)
            
            if not recent_performance.empty and recent_performance.iloc[0]['symbols_tracked'] > 0:
                row = recent_performance.iloc[0]
                insights.append(f"Tracking {int(row['symbols_tracked'])} symbols for rotation")
                insights.append(f"Average performance score: {row['avg_score']:.1f}")
                insights.append(f"Best performer score: {row['best_score']:.1f}")
            
            # Recent rotations
            recent_rotations = pd.read_sql_query('''
                SELECT COUNT(*) as rotation_count
                FROM rotation_decisions 
                WHERE rotation_date > datetime('now', '-30 days')
            ''', conn)
            
            if not recent_rotations.empty:
                rotation_count = recent_rotations.iloc[0]['rotation_count']
                insights.append(f"Portfolio rotations this month: {rotation_count}")
            
            conn.close()
            
            return insights if insights else ["Portfolio rotation analysis pending..."]
            
        except Exception as e:
            logger.error(f"Rotation insights failed: {e}")
            return ["Rotation analysis temporarily unavailable"]