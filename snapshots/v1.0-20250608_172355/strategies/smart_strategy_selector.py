"""
Smart Strategy Selector
Automatically evaluates and switches strategies every 6 hours based on performance and market conditions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import schedule
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from config import Config

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    symbol: str
    start_time: datetime
    end_time: Optional[datetime]
    total_return: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    current_streak: int  # Winning/losing streak

@dataclass
class StrategyEvaluation:
    """Strategy evaluation result"""
    symbol: str
    current_strategy: str
    recommended_strategy: str
    performance_score: float
    market_suitability_score: float
    overall_score: float
    switch_confidence: float
    reasoning: str
    evaluation_time: datetime

class SmartStrategySelector:
    """Smart strategy selector with 6-hour re-evaluation cycle"""
    
    def __init__(self, autoconfig_engine, strategy_engine, okx_data_service, 
                 advanced_risk_manager, db_path: str = "data/smart_selector.db"):
        self.autoconfig_engine = autoconfig_engine
        self.strategy_engine = strategy_engine
        self.okx_data_service = okx_data_service
        self.advanced_risk_manager = advanced_risk_manager
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Evaluation parameters
        self.evaluation_config = {
            'lookback_hours': 24,  # 24 hours for performance evaluation
            'min_trades_threshold': 3,  # Minimum trades for meaningful evaluation
            'performance_weight': 0.4,
            'market_suitability_weight': 0.4,
            'risk_weight': 0.2,
            'switch_threshold': 0.75,  # Only switch if confidence > 75%
            'underperformance_threshold': -0.05  # Switch if strategy underperforms by 5%
        }
        
        # Strategy performance benchmarks
        self.strategy_benchmarks = {
            'dca': {
                'expected_win_rate': 0.65,
                'expected_return': 0.08,
                'expected_drawdown': 0.12,
                'volatility_tolerance': 0.8
            },
            'grid': {
                'expected_win_rate': 0.72,
                'expected_return': 0.06,
                'expected_drawdown': 0.08,
                'volatility_tolerance': 0.6
            },
            'breakout': {
                'expected_win_rate': 0.58,
                'expected_return': 0.15,
                'expected_drawdown': 0.18,
                'volatility_tolerance': 1.2
            },
            'ping_pong': {
                'expected_win_rate': 0.68,
                'expected_return': 0.09,
                'expected_drawdown': 0.10,
                'volatility_tolerance': 0.7
            },
            'trailing_atr': {
                'expected_win_rate': 0.62,
                'expected_return': 0.12,
                'expected_drawdown': 0.15,
                'volatility_tolerance': 1.0
            }
        }
        
        self.running = False
        self.is_running = False
        self.evaluation_thread = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database for strategy performance tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_time DATETIME,
                    end_time DATETIME,
                    total_return REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    avg_trade_duration REAL,
                    volatility REAL,
                    current_streak INTEGER,
                    performance_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_evaluations (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    current_strategy TEXT,
                    recommended_strategy TEXT,
                    performance_score REAL,
                    market_suitability_score REAL,
                    overall_score REAL,
                    switch_confidence REAL,
                    reasoning TEXT,
                    evaluation_time DATETIME,
                    action_taken TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_switches (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    old_strategy TEXT,
                    new_strategy TEXT,
                    switch_time DATETIME,
                    switch_reason TEXT,
                    confidence REAL,
                    pre_switch_performance TEXT,
                    market_conditions TEXT
                )
            """)
    
    def start_evaluation_cycle(self):
        """Start the 6-hour evaluation cycle"""
        if self.running:
            self.logger.warning("Strategy selector is already running")
            return
        
        self.running = True
        self.is_running = True
        
        # Schedule evaluations every 6 hours
        schedule.every(6).hours.do(self._run_full_evaluation)
        
        # Also run at startup
        self._run_full_evaluation()
        
        # Start background thread
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        
        self.logger.info("Smart Strategy Selector started with 6-hour evaluation cycle")
    
    def stop_evaluation_cycle(self):
        """Stop the evaluation cycle"""
        self.running = False
        schedule.clear()
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5)
        self.logger.info("Smart Strategy Selector stopped")
    
    def _evaluation_loop(self):
        """Background evaluation loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _run_full_evaluation(self):
        """Run full strategy evaluation for all symbols"""
        self.logger.info("Starting 6-hour strategy evaluation cycle")
        
        try:
            evaluations = []
            switches_made = 0
            
            for symbol in Config.SUPPORTED_SYMBOLS:
                try:
                    evaluation = self.evaluate_strategy_for_symbol(symbol)
                    evaluations.append(evaluation)
                    
                    # Store evaluation
                    self._store_evaluation(evaluation)
                    
                    # Execute switch if recommended
                    if evaluation.switch_confidence > self.evaluation_config['switch_threshold']:
                        success = self._execute_strategy_switch(evaluation)
                        if success:
                            switches_made += 1
                
                except Exception as e:
                    self.logger.error(f"Error evaluating strategy for {symbol}: {str(e)}")
                    continue
            
            # Generate evaluation summary
            self._generate_evaluation_summary(evaluations, switches_made)
            
        except Exception as e:
            self.logger.error(f"Error in full evaluation cycle: {str(e)}")
    
    def evaluate_strategy_for_symbol(self, symbol: str) -> StrategyEvaluation:
        """Evaluate current strategy performance and recommend optimal strategy"""
        
        # Get current strategy
        current_strategy = self.autoconfig_engine.get_strategy_for_symbol(symbol)
        if not current_strategy:
            current_strategy = 'grid'  # Default fallback
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(symbol, current_strategy)
        
        # Calculate market suitability score
        market_suitability_score = self._calculate_market_suitability_score(symbol, current_strategy)
        
        # Get alternative strategy recommendations
        alternative_strategies = self._get_alternative_strategies(symbol)
        
        # Find best alternative
        best_alternative = self._find_best_alternative(symbol, alternative_strategies)
        
        # Calculate overall scores
        current_overall_score = (
            performance_score * self.evaluation_config['performance_weight'] +
            market_suitability_score * self.evaluation_config['market_suitability_weight']
        )
        
        # Calculate alternative score
        alt_performance_score = self._calculate_hypothetical_performance_score(symbol, best_alternative)
        alt_market_score = self._calculate_market_suitability_score(symbol, best_alternative)
        
        alternative_overall_score = (
            alt_performance_score * self.evaluation_config['performance_weight'] +
            alt_market_score * self.evaluation_config['market_suitability_weight']
        )
        
        # Determine recommendation
        if alternative_overall_score > current_overall_score + 0.1:  # 10% improvement threshold
            recommended_strategy = best_alternative
            switch_confidence = min(alternative_overall_score - current_overall_score, 1.0)
            reasoning = f"Alternative strategy {best_alternative} shows {(alternative_overall_score - current_overall_score)*100:.1f}% better score"
        else:
            recommended_strategy = current_strategy
            switch_confidence = 0.0
            reasoning = f"Current strategy {current_strategy} is performing adequately"
        
        return StrategyEvaluation(
            symbol=symbol,
            current_strategy=current_strategy,
            recommended_strategy=recommended_strategy,
            performance_score=performance_score,
            market_suitability_score=market_suitability_score,
            overall_score=current_overall_score,
            switch_confidence=switch_confidence,
            reasoning=reasoning,
            evaluation_time=datetime.now()
        )
    
    def _calculate_performance_score(self, symbol: str, strategy: str) -> float:
        """Calculate performance score for current strategy"""
        try:
            # Get recent performance data
            cutoff_time = datetime.now() - timedelta(hours=self.evaluation_config['lookback_hours'])
            
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM strategy_performance 
                    WHERE symbol = ? AND strategy_name = ? AND start_time >= ?
                    ORDER BY start_time DESC
                """
                df = pd.read_sql_query(query, conn, params=(symbol, strategy, cutoff_time))
            
            if df.empty:
                return 0.5  # Neutral score if no data
            
            # Calculate weighted performance metrics
            recent_performance = df.iloc[0] if not df.empty else None
            
            if recent_performance is None:
                return 0.5
            
            # Get benchmark for this strategy
            benchmark = self.strategy_benchmarks.get(strategy, {})
            
            # Calculate normalized scores (0-1)
            return_score = min(max(recent_performance['total_return'] / benchmark.get('expected_return', 0.1), 0), 1)
            winrate_score = recent_performance['win_rate'] / benchmark.get('expected_win_rate', 0.6)
            drawdown_score = 1 - (abs(recent_performance['max_drawdown']) / benchmark.get('expected_drawdown', 0.2))
            
            # Weighted average
            performance_score = (return_score * 0.4 + winrate_score * 0.4 + drawdown_score * 0.2)
            
            return min(max(performance_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score for {symbol}: {str(e)}")
            return 0.5
    
    def _calculate_market_suitability_score(self, symbol: str, strategy: str) -> float:
        """Calculate how well strategy suits current market conditions"""
        try:
            # Get current market data
            data = self.okx_data_service.get_historical_data(symbol, '1h', limit=48)  # 48 hours
            if data.empty:
                return 0.5
            
            # Calculate market metrics
            volatility = data['close'].pct_change().std() * np.sqrt(24)  # Daily volatility
            volume_ratio = data['volume'].iloc[-1] / data['volume'].mean()
            
            # Calculate trend strength
            ema_short = data['close'].ewm(span=12).mean()
            ema_long = data['close'].ewm(span=26).mean()
            trend_strength = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
            
            # Strategy-specific suitability scoring
            suitability_score = 0.5  # Default
            
            if strategy == 'grid':
                # Grid works best in ranging markets with low volatility
                suitability_score = max(0, 1 - volatility * 20) * 0.6 + (1 - min(trend_strength * 10, 1)) * 0.4
            
            elif strategy == 'breakout':
                # Breakout works best in trending markets with high volume
                suitability_score = min(volatility * 15, 1) * 0.5 + min(volume_ratio / 2, 1) * 0.3 + min(trend_strength * 5, 1) * 0.2
            
            elif strategy == 'dca':
                # DCA works consistently across market conditions
                suitability_score = 0.7 - min(volatility * 10, 0.2)  # Slightly lower in high volatility
            
            elif strategy == 'ping_pong':
                # Ping-pong works in ranging markets
                suitability_score = max(0, 1 - volatility * 25) * 0.7 + (1 - min(trend_strength * 8, 1)) * 0.3
            
            elif strategy == 'trailing_atr':
                # Trailing ATR works in trending markets
                suitability_score = min(trend_strength * 8, 1) * 0.6 + min(volatility * 12, 1) * 0.4
            
            return min(max(suitability_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating market suitability for {symbol}: {str(e)}")
            return 0.5
    
    def _calculate_hypothetical_performance_score(self, symbol: str, strategy: str) -> float:
        """Calculate hypothetical performance score for alternative strategy"""
        # Use benchmark data adjusted for current market conditions
        try:
            data = self.okx_data_service.get_historical_data(symbol, '1h', limit=24)
            if data.empty:
                return 0.5
            
            volatility = data['close'].pct_change().std() * np.sqrt(24)
            benchmark = self.strategy_benchmarks.get(strategy, {})
            
            # Adjust benchmark performance based on current volatility
            volatility_adjustment = min(volatility / benchmark.get('volatility_tolerance', 1.0), 1.5)
            adjusted_return = benchmark.get('expected_return', 0.08) / volatility_adjustment
            adjusted_winrate = benchmark.get('expected_win_rate', 0.65) * (2 - volatility_adjustment)
            
            # Normalize to 0-1 score
            return min(max((adjusted_return + adjusted_winrate) / 2, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating hypothetical performance for {symbol}: {str(e)}")
            return 0.5
    
    def _get_alternative_strategies(self, symbol: str) -> List[str]:
        """Get list of alternative strategies to consider"""
        current_strategy = self.autoconfig_engine.get_strategy_for_symbol(symbol)
        all_strategies = list(self.strategy_benchmarks.keys())
        
        # Remove current strategy from alternatives
        alternatives = [s for s in all_strategies if s != current_strategy]
        
        return alternatives
    
    def _find_best_alternative(self, symbol: str, alternatives: List[str]) -> str:
        """Find the best alternative strategy"""
        best_strategy = alternatives[0] if alternatives else 'grid'
        best_score = 0
        
        for strategy in alternatives:
            score = self._calculate_market_suitability_score(symbol, strategy)
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    def _execute_strategy_switch(self, evaluation: StrategyEvaluation) -> bool:
        """Execute strategy switch if recommended"""
        try:
            if evaluation.current_strategy == evaluation.recommended_strategy:
                return False  # No switch needed
            
            # Get current market conditions for logging
            data = self.okx_data_service.get_historical_data(evaluation.symbol, '1h', limit=1)
            current_price = data['close'].iloc[-1] if not data.empty else 0
            
            # Store pre-switch performance
            pre_switch_perf = self._get_current_performance_snapshot(evaluation.symbol, evaluation.current_strategy)
            
            # Execute the switch through AutoConfig Engine
            self.autoconfig_engine.force_strategy_switch(
                evaluation.symbol,
                evaluation.recommended_strategy,
                f"Smart Selector: {evaluation.reasoning}"
            )
            
            # Log the switch
            self._log_strategy_switch(
                evaluation.symbol,
                evaluation.current_strategy,
                evaluation.recommended_strategy,
                evaluation.reasoning,
                evaluation.switch_confidence,
                pre_switch_perf,
                current_price
            )
            
            self.logger.info(f"Strategy switched for {evaluation.symbol}: {evaluation.current_strategy} -> {evaluation.recommended_strategy}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing strategy switch for {evaluation.symbol}: {str(e)}")
            return False
    
    def _get_current_performance_snapshot(self, symbol: str, strategy: str) -> Dict[str, Any]:
        """Get current performance snapshot before switching"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM strategy_performance 
                    WHERE symbol = ? AND strategy_name = ? AND start_time >= ?
                    ORDER BY start_time DESC LIMIT 1
                """
                df = pd.read_sql_query(query, conn, params=(symbol, strategy, cutoff_time))
            
            if not df.empty:
                return df.iloc[0].to_dict()
            else:
                return {'no_data': True}
                
        except Exception as e:
            self.logger.error(f"Error getting performance snapshot: {str(e)}")
            return {'error': str(e)}
    
    def _log_strategy_switch(self, symbol: str, old_strategy: str, new_strategy: str,
                           reason: str, confidence: float, pre_switch_perf: Dict[str, Any],
                           current_price: float):
        """Log strategy switch for tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO strategy_switches (
                    symbol, old_strategy, new_strategy, switch_time,
                    switch_reason, confidence, pre_switch_performance, market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, old_strategy, new_strategy, datetime.now(),
                reason, confidence, json.dumps(pre_switch_perf),
                json.dumps({'current_price': current_price})
            ))
    
    def _store_evaluation(self, evaluation: StrategyEvaluation):
        """Store evaluation results in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO strategy_evaluations (
                    symbol, current_strategy, recommended_strategy, performance_score,
                    market_suitability_score, overall_score, switch_confidence,
                    reasoning, evaluation_time, action_taken
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation.symbol,
                evaluation.current_strategy,
                evaluation.recommended_strategy,
                evaluation.performance_score,
                evaluation.market_suitability_score,
                evaluation.overall_score,
                evaluation.switch_confidence,
                evaluation.reasoning,
                evaluation.evaluation_time,
                'switch' if evaluation.switch_confidence > self.evaluation_config['switch_threshold'] else 'keep'
            ))
    
    def _generate_evaluation_summary(self, evaluations: List[StrategyEvaluation], switches_made: int):
        """Generate and log evaluation cycle summary"""
        total_symbols = len(evaluations)
        avg_performance_score = np.mean([e.performance_score for e in evaluations])
        avg_market_score = np.mean([e.market_suitability_score for e in evaluations])
        
        # Strategy distribution
        strategy_counts = {}
        for eval in evaluations:
            strategy = eval.recommended_strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        summary = {
            'evaluation_time': datetime.now().isoformat(),
            'total_symbols_evaluated': total_symbols,
            'switches_made': switches_made,
            'switch_rate': switches_made / total_symbols if total_symbols > 0 else 0,
            'avg_performance_score': avg_performance_score,
            'avg_market_suitability_score': avg_market_score,
            'strategy_distribution': strategy_counts
        }
        
        self.logger.info(f"Evaluation cycle completed: {switches_made}/{total_symbols} switches made")
        self.logger.info(f"Average scores - Performance: {avg_performance_score:.2f}, Market: {avg_market_score:.2f}")
        
        return summary
    
    def get_evaluation_history(self, symbol: str = None, days: int = 7) -> pd.DataFrame:
        """Get evaluation history"""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                query = """
                    SELECT * FROM strategy_evaluations 
                    WHERE symbol = ? AND evaluation_time >= ?
                    ORDER BY evaluation_time DESC
                """
                params = (symbol, datetime.now() - timedelta(days=days))
            else:
                query = """
                    SELECT * FROM strategy_evaluations 
                    WHERE evaluation_time >= ?
                    ORDER BY evaluation_time DESC
                """
                params = (datetime.now() - timedelta(days=days),)
            
            return pd.read_sql_query(query, conn, params=params)
    
    def get_switch_history(self, symbol: str = None, days: int = 7) -> pd.DataFrame:
        """Get strategy switch history"""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                query = """
                    SELECT * FROM strategy_switches 
                    WHERE symbol = ? AND switch_time >= ?
                    ORDER BY switch_time DESC
                """
                params = (symbol, datetime.now() - timedelta(days=days))
            else:
                query = """
                    SELECT * FROM strategy_switches 
                    WHERE switch_time >= ?
                    ORDER BY switch_time DESC
                """
                params = (datetime.now() - timedelta(days=days),)
            
            return pd.read_sql_query(query, conn, params=params)
    
    def get_strategy_effectiveness_report(self) -> Dict[str, Any]:
        """Get comprehensive strategy effectiveness report"""
        with sqlite3.connect(self.db_path) as conn:
            # Switch success rate
            switch_query = """
                SELECT old_strategy, new_strategy, COUNT(*) as switch_count,
                       AVG(confidence) as avg_confidence
                FROM strategy_switches 
                WHERE switch_time >= ?
                GROUP BY old_strategy, new_strategy
            """
            cutoff_date = datetime.now() - timedelta(days=30)
            switch_df = pd.read_sql_query(switch_query, conn, params=(cutoff_date,))
            
            # Evaluation trends
            eval_query = """
                SELECT symbol, AVG(performance_score) as avg_perf,
                       AVG(market_suitability_score) as avg_market,
                       COUNT(*) as eval_count
                FROM strategy_evaluations 
                WHERE evaluation_time >= ?
                GROUP BY symbol
            """
            eval_df = pd.read_sql_query(eval_query, conn, params=(cutoff_date,))
            
            return {
                'switch_patterns': switch_df.to_dict('records'),
                'evaluation_trends': eval_df.to_dict('records'),
                'total_switches': len(switch_df),
                'evaluation_frequency': '6 hours',
                'last_evaluation': datetime.now().isoformat()
            }
    
    def update_strategy_performance(self, symbol: str, strategy: str, 
                                  performance_data: Dict[str, Any]):
        """Update strategy performance data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO strategy_performance (
                    strategy_name, symbol, start_time, end_time, total_return,
                    win_rate, sharpe_ratio, max_drawdown, total_trades,
                    avg_trade_duration, volatility, current_streak, performance_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy, symbol, 
                performance_data.get('start_time', datetime.now()),
                performance_data.get('end_time'),
                performance_data.get('total_return', 0),
                performance_data.get('win_rate', 0),
                performance_data.get('sharpe_ratio', 0),
                performance_data.get('max_drawdown', 0),
                performance_data.get('total_trades', 0),
                performance_data.get('avg_trade_duration', 0),
                performance_data.get('volatility', 0),
                performance_data.get('current_streak', 0),
                json.dumps(performance_data)
            ))