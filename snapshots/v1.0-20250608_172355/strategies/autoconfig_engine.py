"""
AutoConfig Engine
Monitors market conditions and automatically selects optimal strategies per symbol
Rebalances every 6 hours based on volume, volatility, and market regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass
from .strategy_engine import StrategyEngine

@dataclass
class MarketCondition:
    """Market condition analysis result"""
    symbol: str
    volatility: float
    volume_ratio: float
    trend_strength: float
    regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    recommended_strategy: str
    confidence: float
    timestamp: datetime

class MarketRegimeDetector:
    """Detects current market regime for strategy selection"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
    
    def analyze_market_condition(self, data: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions and recommend strategy"""
        if len(data) < self.lookback_period:
            return MarketCondition(
                symbol="", volatility=0, volume_ratio=1, trend_strength=0,
                regime="unknown", recommended_strategy="dca", confidence=0.5,
                timestamp=datetime.now()
            )
        
        # Calculate volatility (normalized ATR)
        volatility = self._calculate_volatility(data)
        
        # Calculate volume ratio
        volume_ratio = self._calculate_volume_ratio(data)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Determine market regime
        regime = self._determine_regime(volatility, volume_ratio, trend_strength)
        
        # Recommend strategy based on regime
        strategy, confidence = self._recommend_strategy(regime, volatility, volume_ratio, trend_strength)
        
        return MarketCondition(
            symbol="",
            volatility=volatility,
            volume_ratio=volume_ratio,
            trend_strength=trend_strength,
            regime=regime,
            recommended_strategy=strategy,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate normalized volatility using ATR"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        
        # Normalize by price
        current_atr = atr.tail(1).iloc[0]
        current_price = data['close'].tail(1).iloc[0]
        
        return (current_atr / current_price) if current_price > 0 else 0
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """Calculate current volume vs average volume"""
        current_volume = data['volume'].tail(5).mean()
        avg_volume = data['volume'].tail(self.lookback_period).mean()
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like calculation"""
        # Calculate directional movement
        high_diff = data['high'].diff()
        low_diff = data['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate true range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Smooth the values
        period = 14
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / true_range.rolling(period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / true_range.rolling(period).mean()
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.tail(1).iloc[0] / 100 if not pd.isna(adx.tail(1).iloc[0]) else 0
    
    def _determine_regime(self, volatility: float, volume_ratio: float, trend_strength: float) -> str:
        """Determine market regime based on metrics"""
        # High volatility threshold
        if volatility > 0.05:  # 5% daily volatility
            return "volatile"
        
        # Strong trend
        if trend_strength > 0.25:
            # Check trend direction using recent price action
            return "trending_up" if volume_ratio > 1.2 else "trending_down"
        
        # Low volatility, weak trend = ranging market
        if volatility < 0.02 and trend_strength < 0.15:
            return "ranging"
        
        # Default to ranging for moderate conditions
        return "ranging"
    
    def _recommend_strategy(self, regime: str, volatility: float, volume_ratio: float, trend_strength: float) -> Tuple[str, float]:
        """Recommend strategy based on market regime"""
        
        if regime == "trending_up":
            if volume_ratio > 2.0:  # High volume breakout
                return "breakout", 0.85
            else:
                return "trailing_atr", 0.75
        
        elif regime == "trending_down":
            if volatility > 0.03:  # High volatility
                return "dca", 0.80
            else:
                return "trailing_atr", 0.70
        
        elif regime == "ranging":
            if volatility < 0.015:  # Very low volatility
                return "grid", 0.85
            else:
                return "ping_pong", 0.75
        
        elif regime == "volatile":
            if volume_ratio > 1.5:  # High volume volatility
                return "breakout", 0.70
            else:
                return "dca", 0.65
        
        # Default fallback
        return "dca", 0.50

class AutoConfigEngine:
    """Main AutoConfig Engine for automatic strategy selection and management"""
    
    def __init__(self, db_path: str = "data/autoconfig.db"):
        self.db_path = db_path
        self.strategy_engine = StrategyEngine()
        self.regime_detector = MarketRegimeDetector()
        self.active_strategies = {}  # symbol -> strategy_type
        self.strategy_switches = []  # History of strategy changes
        self.last_rebalance = {}  # symbol -> timestamp
        self.rebalance_interval = timedelta(hours=6)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for tracking strategy changes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_switches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                old_strategy TEXT,
                new_strategy TEXT,
                regime TEXT,
                volatility REAL,
                volume_ratio REAL,
                trend_strength REAL,
                confidence REAL,
                reason TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                volatility REAL,
                volume_ratio REAL,
                trend_strength REAL,
                regime TEXT,
                recommended_strategy TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_and_select_strategy(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions and select optimal strategy for symbol"""
        # Analyze current market condition
        condition = self.regime_detector.analyze_market_condition(data)
        condition.symbol = symbol
        
        # Store market condition in database
        self._store_market_condition(condition)
        
        # Get current active strategy
        current_strategy = self.active_strategies.get(symbol, None)
        
        # Check if strategy should be switched
        should_switch, reason = self._should_switch_strategy(
            symbol, current_strategy, condition
        )
        
        result = {
            'symbol': symbol,
            'current_strategy': current_strategy,
            'recommended_strategy': condition.recommended_strategy,
            'should_switch': should_switch,
            'reason': reason,
            'market_condition': {
                'regime': condition.regime,
                'volatility': condition.volatility,
                'volume_ratio': condition.volume_ratio,
                'trend_strength': condition.trend_strength,
                'confidence': condition.confidence
            }
        }
        
        # Execute strategy switch if needed
        if should_switch:
            self._switch_strategy(symbol, current_strategy, condition, reason)
            result['switched'] = True
            result['new_strategy'] = condition.recommended_strategy
        
        return result
    
    def _should_switch_strategy(self, symbol: str, current_strategy: Optional[str], condition: MarketCondition) -> Tuple[bool, str]:
        """Determine if strategy should be switched"""
        # Always switch if no current strategy
        if current_strategy is None:
            return True, "Initial strategy selection"
        
        # Check if enough time has passed since last rebalance
        last_rebalance = self.last_rebalance.get(symbol)
        if last_rebalance and datetime.now() - last_rebalance < self.rebalance_interval:
            return False, f"Rebalance interval not reached (last: {last_rebalance})"
        
        # Switch if recommended strategy is different and confidence is high
        if condition.recommended_strategy != current_strategy:
            if condition.confidence >= 0.7:
                return True, f"High confidence switch to {condition.recommended_strategy} (confidence: {condition.confidence:.2f})"
            elif condition.confidence >= 0.6:
                # Additional check for regime change
                recent_conditions = self._get_recent_conditions(symbol, hours=2)
                if len(recent_conditions) >= 3:
                    consistent_regime = all(c['regime'] == condition.regime for c in recent_conditions)
                    if consistent_regime:
                        return True, f"Consistent regime change to {condition.regime}"
        
        return False, "No switch needed"
    
    def _switch_strategy(self, symbol: str, old_strategy: Optional[str], condition: MarketCondition, reason: str):
        """Execute strategy switch and log the change"""
        # Update active strategy
        self.active_strategies[symbol] = condition.recommended_strategy
        self.last_rebalance[symbol] = datetime.now()
        
        # Log strategy switch
        switch_record = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'old_strategy': old_strategy,
            'new_strategy': condition.recommended_strategy,
            'regime': condition.regime,
            'volatility': condition.volatility,
            'volume_ratio': condition.volume_ratio,
            'trend_strength': condition.trend_strength,
            'confidence': condition.confidence,
            'reason': reason
        }
        
        self.strategy_switches.append(switch_record)
        self._store_strategy_switch(switch_record)
        
        print(f"Strategy switch for {symbol}: {old_strategy} -> {condition.recommended_strategy} ({reason})")
    
    def _store_market_condition(self, condition: MarketCondition):
        """Store market condition in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO market_conditions 
            (symbol, timestamp, volatility, volume_ratio, trend_strength, regime, recommended_strategy, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            condition.symbol,
            condition.timestamp,
            condition.volatility,
            condition.volume_ratio,
            condition.trend_strength,
            condition.regime,
            condition.recommended_strategy,
            condition.confidence
        ))
        
        conn.commit()
        conn.close()
    
    def _store_strategy_switch(self, switch_record: Dict[str, Any]):
        """Store strategy switch in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO strategy_switches 
            (symbol, timestamp, old_strategy, new_strategy, regime, volatility, volume_ratio, trend_strength, confidence, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            switch_record['symbol'],
            switch_record['timestamp'],
            switch_record['old_strategy'],
            switch_record['new_strategy'],
            switch_record['regime'],
            switch_record['volatility'],
            switch_record['volume_ratio'],
            switch_record['trend_strength'],
            switch_record['confidence'],
            switch_record['reason']
        ))
        
        conn.commit()
        conn.close()
    
    def _get_recent_conditions(self, symbol: str, hours: int = 6) -> List[Dict[str, Any]]:
        """Get recent market conditions for symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT regime, volatility, volume_ratio, trend_strength, confidence, timestamp
            FROM market_conditions 
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (symbol, cutoff_time))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'regime': row[0],
                'volatility': row[1],
                'volume_ratio': row[2],
                'trend_strength': row[3],
                'confidence': row[4],
                'timestamp': row[5]
            })
        
        conn.close()
        return results
    
    def get_strategy_for_symbol(self, symbol: str) -> str:
        """Get currently active strategy for symbol"""
        if symbol not in self.active_strategies:
            # Initialize strategy for new symbol using live OKX data
            try:
                from trading.okx_data_service import OKXDataService
                okx_service = OKXDataService()
                data = okx_service.get_historical_data(symbol, '1h', limit=100)
                
                if not data.empty:
                    # Analyze market conditions and select initial strategy
                    result = self.analyze_and_select_strategy(symbol, data)
                    self.active_strategies[symbol] = result['recommended_strategy']
                    print(f"Strategy initialized for {symbol}: {result['recommended_strategy']} (market-based)")
                else:
                    # Use grid strategy as fallback for new symbols
                    self.active_strategies[symbol] = 'grid'
                    print(f"Strategy initialized for {symbol}: grid (fallback)")
                    
            except Exception as e:
                # Default to grid strategy for new symbols
                self.active_strategies[symbol] = 'grid'
                print(f"Strategy initialized for {symbol}: grid (default)")
        
        return self.active_strategies[symbol]
    
    def generate_strategy_signal(self, symbol: str, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Generate trading signal using the active strategy for symbol"""
        active_strategy = self.get_strategy_for_symbol(symbol)
        
        if active_strategy is None:
            # Auto-select strategy if none active
            analysis = self.analyze_and_select_strategy(symbol, data)
            active_strategy = analysis.get('new_strategy') or analysis.get('recommended_strategy')
        
        if active_strategy:
            return self.strategy_engine.generate_signals(symbol, active_strategy, data, current_price)
        else:
            return {'action': 'hold', 'reason': 'No strategy selected'}
    
    def get_strategy_status(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive strategy status for symbol"""
        active_strategy = self.get_strategy_for_symbol(symbol)
        last_rebalance = self.last_rebalance.get(symbol)
        
        # Get recent conditions
        recent_conditions = self._get_recent_conditions(symbol, hours=24)
        
        # Get strategy switches
        recent_switches = [s for s in self.strategy_switches 
                          if s['symbol'] == symbol and 
                          datetime.now() - s['timestamp'] < timedelta(days=7)]
        
        return {
            'symbol': symbol,
            'active_strategy': active_strategy,
            'last_rebalance': last_rebalance,
            'next_rebalance': last_rebalance + self.rebalance_interval if last_rebalance else None,
            'recent_conditions': recent_conditions[:5],  # Last 5 conditions
            'recent_switches': recent_switches[-3:],  # Last 3 switches
            'strategy_info': self.strategy_engine.get_all_strategies_info(symbol)
        }
    
    def get_all_strategies_status(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get strategy status for all symbols"""
        return {symbol: self.get_strategy_status(symbol) for symbol in symbols}
    
    def force_strategy_switch(self, symbol: str, strategy_type: str, reason: str = "Manual override"):
        """Manually force a strategy switch"""
        if strategy_type not in self.strategy_engine.get_available_strategies():
            raise ValueError(f"Invalid strategy type: {strategy_type}")
        
        old_strategy = self.active_strategies.get(symbol)
        
        # Create dummy condition for logging
        condition = MarketCondition(
            symbol=symbol,
            volatility=0,
            volume_ratio=1,
            trend_strength=0,
            regime="manual",
            recommended_strategy=strategy_type,
            confidence=1.0,
            timestamp=datetime.now()
        )
        
        self._switch_strategy(symbol, old_strategy, condition, reason)
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary of strategy switches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Get strategy switch statistics
        cursor.execute('''
            SELECT new_strategy, COUNT(*) as switch_count
            FROM strategy_switches 
            WHERE timestamp > ?
            GROUP BY new_strategy
            ORDER BY switch_count DESC
        ''', (cutoff_time,))
        
        strategy_usage = dict(cursor.fetchall())
        
        # Get regime distribution
        cursor.execute('''
            SELECT regime, COUNT(*) as count
            FROM market_conditions 
            WHERE timestamp > ?
            GROUP BY regime
            ORDER BY count DESC
        ''', (cutoff_time,))
        
        regime_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'period_days': days,
            'strategy_usage': strategy_usage,
            'regime_distribution': regime_distribution,
            'total_switches': sum(strategy_usage.values()),
            'active_symbols': len(self.active_strategies),
            'available_strategies': self.strategy_engine.get_available_strategies()
        }