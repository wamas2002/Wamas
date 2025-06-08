"""
Advanced Risk Manager with Multi-level TP/SL and Trailing Stops
Supports multiple take profit levels, stop losses, and ATR-based trailing stops
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from config import Config

@dataclass
class TakeProfitLevel:
    """Take profit level configuration"""
    level: int  # 1, 2, 3, etc.
    price: float
    percentage: float  # % of position to close
    triggered: bool = False
    trigger_time: Optional[datetime] = None

@dataclass
class StopLossConfig:
    """Stop loss configuration"""
    price: float
    percentage: float  # % from entry price
    trailing: bool = False
    atr_multiplier: Optional[float] = None
    updated_time: datetime = None

@dataclass
class PositionRisk:
    """Position risk management configuration"""
    symbol: str
    entry_price: float
    position_size: float
    entry_time: datetime
    take_profits: List[TakeProfitLevel]
    stop_loss: StopLossConfig
    trailing_stop: Optional[Dict[str, Any]] = None
    max_risk_pct: float = 0.02  # 2% max risk per trade
    current_pnl: float = 0.0
    unrealized_pnl: float = 0.0

@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    symbol: str
    current_price: float
    position_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    distance_to_sl: float
    distance_to_next_tp: float
    risk_reward_ratio: float
    atr_value: float
    volatility_risk: float

class AdvancedRiskManager:
    """Advanced risk management with multi-level TP/SL and trailing stops"""
    
    def __init__(self, db_path: str = "data/risk_management.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.active_positions = {}  # symbol -> PositionRisk
        
        # Default risk parameters
        self.default_config = {
            'max_risk_per_trade': 0.02,  # 2%
            'max_portfolio_risk': 0.10,  # 10%
            'default_tp_levels': [0.03, 0.06, 0.10],  # 3%, 6%, 10%
            'default_sl_percentage': 0.02,  # 2%
            'trailing_stop_atr_multiplier': 2.0,
            'position_sizing_method': 'fixed_risk'  # 'fixed_risk', 'kelly', 'volatility_adjusted'
        }
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database for risk management tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_risk (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price REAL,
                    position_size REAL,
                    entry_time DATETIME,
                    take_profits TEXT,
                    stop_loss TEXT,
                    trailing_stop TEXT,
                    max_risk_pct REAL,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    event_type TEXT,
                    event_time DATETIME,
                    price REAL,
                    description TEXT,
                    pnl REAL,
                    position_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics_history (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME,
                    current_price REAL,
                    unrealized_pnl REAL,
                    distance_to_sl REAL,
                    distance_to_tp REAL,
                    atr_value REAL,
                    risk_metrics TEXT
                )
            """)
    
    def create_position_risk(self, symbol: str, entry_price: float, position_size: float,
                           tp_levels: Optional[List[float]] = None,
                           sl_percentage: Optional[float] = None,
                           use_trailing_stop: bool = True) -> PositionRisk:
        """Create comprehensive risk management for a new position"""
        
        # Use defaults if not specified
        if tp_levels is None:
            tp_levels = self.default_config['default_tp_levels']
        if sl_percentage is None:
            sl_percentage = self.default_config['default_sl_percentage']
        
        # Create take profit levels
        take_profits = []
        for i, tp_pct in enumerate(tp_levels):
            tp_price = entry_price * (1 + tp_pct)
            tp_level = TakeProfitLevel(
                level=i + 1,
                price=tp_price,
                percentage=0.33 if len(tp_levels) == 3 else 1.0 / len(tp_levels),  # Equal distribution
                triggered=False
            )
            take_profits.append(tp_level)
        
        # Create stop loss
        sl_price = entry_price * (1 - sl_percentage)
        stop_loss = StopLossConfig(
            price=sl_price,
            percentage=sl_percentage,
            trailing=use_trailing_stop,
            atr_multiplier=self.default_config['trailing_stop_atr_multiplier'] if use_trailing_stop else None,
            updated_time=datetime.now()
        )
        
        # Create trailing stop configuration
        trailing_stop = None
        if use_trailing_stop:
            trailing_stop = {
                'enabled': True,
                'atr_multiplier': self.default_config['trailing_stop_atr_multiplier'],
                'highest_price': entry_price,
                'current_stop': sl_price,
                'last_update': datetime.now()
            }
        
        position_risk = PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            entry_time=datetime.now(),
            take_profits=take_profits,
            stop_loss=stop_loss,
            trailing_stop=trailing_stop,
            max_risk_pct=self.default_config['max_risk_per_trade']
        )
        
        # Store in active positions
        self.active_positions[symbol] = position_risk
        
        # Save to database
        self._store_position_risk(position_risk)
        
        self.logger.info(f"Created position risk management for {symbol} at {entry_price}")
        return position_risk
    
    def update_position_risk(self, symbol: str, current_price: float, 
                           atr_value: Optional[float] = None) -> RiskMetrics:
        """Update risk management for an existing position"""
        
        if symbol not in self.active_positions:
            raise ValueError(f"No active position found for {symbol}")
        
        position = self.active_positions[symbol]
        
        # Calculate current P&L
        unrealized_pnl = (current_price - position.entry_price) * position.position_size
        unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
        position.unrealized_pnl = unrealized_pnl
        
        # Check take profit levels
        self._check_take_profit_triggers(position, current_price)
        
        # Update trailing stop if enabled
        if position.trailing_stop and position.trailing_stop['enabled'] and atr_value:
            self._update_trailing_stop(position, current_price, atr_value)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(position, current_price, atr_value)
        
        # Store metrics
        self._store_risk_metrics(risk_metrics)
        
        # Check for risk alerts
        self._check_risk_alerts(position, risk_metrics)
        
        return risk_metrics
    
    def _check_take_profit_triggers(self, position: PositionRisk, current_price: float):
        """Check and trigger take profit levels"""
        for tp_level in position.take_profits:
            if not tp_level.triggered and current_price >= tp_level.price:
                tp_level.triggered = True
                tp_level.trigger_time = datetime.now()
                
                # Log take profit event
                self._log_risk_event(
                    position.symbol,
                    'take_profit_triggered',
                    current_price,
                    f"Take Profit Level {tp_level.level} triggered at {current_price}",
                    (current_price - position.entry_price) * position.position_size * tp_level.percentage
                )
                
                self.logger.info(f"TP{tp_level.level} triggered for {position.symbol} at {current_price}")
    
    def _update_trailing_stop(self, position: PositionRisk, current_price: float, atr_value: float):
        """Update trailing stop based on ATR and price movement"""
        trailing_stop = position.trailing_stop
        
        # Update highest price seen
        if current_price > trailing_stop['highest_price']:
            trailing_stop['highest_price'] = current_price
            
            # Calculate new trailing stop price
            atr_multiplier = trailing_stop['atr_multiplier']
            new_stop_price = current_price - (atr_value * atr_multiplier)
            
            # Only move stop loss up (for long positions)
            if new_stop_price > trailing_stop['current_stop']:
                old_stop = trailing_stop['current_stop']
                trailing_stop['current_stop'] = new_stop_price
                trailing_stop['last_update'] = datetime.now()
                
                # Update the main stop loss
                position.stop_loss.price = new_stop_price
                position.stop_loss.updated_time = datetime.now()
                
                self._log_risk_event(
                    position.symbol,
                    'trailing_stop_updated',
                    current_price,
                    f"Trailing stop moved from {old_stop:.4f} to {new_stop_price:.4f}",
                    0
                )
                
                self.logger.info(f"Trailing stop updated for {position.symbol}: {new_stop_price:.4f}")
    
    def _calculate_risk_metrics(self, position: PositionRisk, current_price: float, 
                              atr_value: Optional[float] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Basic metrics
        position_value = current_price * position.position_size
        unrealized_pnl = (current_price - position.entry_price) * position.position_size
        unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Distance calculations
        distance_to_sl = abs(current_price - position.stop_loss.price) / current_price
        
        # Find next untriggered take profit
        next_tp_price = None
        for tp_level in position.take_profits:
            if not tp_level.triggered:
                next_tp_price = tp_level.price
                break
        
        distance_to_next_tp = 0.0
        if next_tp_price:
            distance_to_next_tp = abs(next_tp_price - current_price) / current_price
        
        # Risk-reward ratio
        potential_loss = abs(position.entry_price - position.stop_loss.price) * position.position_size
        potential_gain = 0.0
        if next_tp_price:
            potential_gain = abs(next_tp_price - position.entry_price) * position.position_size
        
        risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0.0
        
        # Volatility risk
        volatility_risk = 0.5  # Default
        if atr_value:
            volatility_risk = min(atr_value / current_price * 100, 1.0)  # Normalize to 0-1
        
        return RiskMetrics(
            symbol=position.symbol,
            current_price=current_price,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            distance_to_sl=distance_to_sl,
            distance_to_next_tp=distance_to_next_tp,
            risk_reward_ratio=risk_reward_ratio,
            atr_value=atr_value or 0.0,
            volatility_risk=volatility_risk
        )
    
    def _check_risk_alerts(self, position: PositionRisk, metrics: RiskMetrics):
        """Check for risk alerts and warnings"""
        alerts = []
        
        # Stop loss proximity alert
        if metrics.distance_to_sl < 0.01:  # Within 1% of stop loss
            alerts.append("CRITICAL: Position near stop loss")
        
        # Large unrealized loss alert
        if metrics.unrealized_pnl_pct < -0.05:  # More than 5% loss
            alerts.append("WARNING: Position showing significant loss")
        
        # High volatility alert
        if metrics.volatility_risk > 0.8:
            alerts.append("WARNING: High volatility detected")
        
        # Log alerts
        for alert in alerts:
            self._log_risk_event(
                position.symbol,
                'risk_alert',
                metrics.current_price,
                alert,
                metrics.unrealized_pnl
            )
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float, risk_amount: float,
                              method: str = 'fixed_risk') -> float:
        """Calculate optimal position size based on risk management"""
        
        if method == 'fixed_risk':
            # Fixed risk amount per trade
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share > 0:
                position_size = risk_amount / risk_per_share
            else:
                position_size = 0.0
        
        elif method == 'percentage_risk':
            # Risk percentage of portfolio
            portfolio_value = 100000  # Would be real portfolio value
            risk_amount = portfolio_value * self.default_config['max_risk_per_trade']
            risk_per_share = abs(entry_price - stop_loss_price)
            position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0.0
        
        elif method == 'volatility_adjusted':
            # Adjust position size based on volatility
            base_position = risk_amount / abs(entry_price - stop_loss_price)
            # Reduce size for high volatility assets (would use real volatility data)
            volatility_factor = 0.8  # Placeholder
            position_size = base_position * volatility_factor
        
        else:
            position_size = 1.0  # Default
        
        return max(position_size, 0.01)  # Minimum position size
    
    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary"""
        
        total_positions = len(self.active_positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
        
        # Risk by symbol
        risk_by_symbol = {}
        for symbol, position in self.active_positions.items():
            risk_by_symbol[symbol] = {
                'unrealized_pnl': position.unrealized_pnl,
                'max_risk_pct': position.max_risk_pct,
                'position_size': position.position_size,
                'entry_price': position.entry_price
            }
        
        # Take profit status
        tp_status = {}
        for symbol, position in self.active_positions.items():
            triggered_tps = sum(1 for tp in position.take_profits if tp.triggered)
            total_tps = len(position.take_profits)
            tp_status[symbol] = f"{triggered_tps}/{total_tps}"
        
        return {
            'total_positions': total_positions,
            'total_unrealized_pnl': total_unrealized_pnl,
            'risk_by_symbol': risk_by_symbol,
            'take_profit_status': tp_status,
            'last_updated': datetime.now().isoformat()
        }
    
    def modify_take_profit_levels(self, symbol: str, new_tp_levels: List[Dict[str, Any]]):
        """Modify take profit levels for an existing position"""
        if symbol not in self.active_positions:
            raise ValueError(f"No active position found for {symbol}")
        
        position = self.active_positions[symbol]
        
        # Update take profit levels
        new_tps = []
        for i, tp_data in enumerate(new_tp_levels):
            tp_level = TakeProfitLevel(
                level=i + 1,
                price=tp_data['price'],
                percentage=tp_data.get('percentage', 0.33),
                triggered=tp_data.get('triggered', False)
            )
            new_tps.append(tp_level)
        
        position.take_profits = new_tps
        
        self._log_risk_event(
            symbol,
            'tp_levels_modified',
            position.entry_price,
            f"Take profit levels modified: {len(new_tps)} levels",
            0
        )
    
    def modify_stop_loss(self, symbol: str, new_sl_price: float, 
                        enable_trailing: bool = None):
        """Modify stop loss for an existing position"""
        if symbol not in self.active_positions:
            raise ValueError(f"No active position found for {symbol}")
        
        position = self.active_positions[symbol]
        old_sl_price = position.stop_loss.price
        
        # Update stop loss
        position.stop_loss.price = new_sl_price
        position.stop_loss.updated_time = datetime.now()
        
        if enable_trailing is not None:
            position.stop_loss.trailing = enable_trailing
            if enable_trailing and not position.trailing_stop:
                position.trailing_stop = {
                    'enabled': True,
                    'atr_multiplier': self.default_config['trailing_stop_atr_multiplier'],
                    'highest_price': position.entry_price,
                    'current_stop': new_sl_price,
                    'last_update': datetime.now()
                }
        
        self._log_risk_event(
            symbol,
            'stop_loss_modified',
            new_sl_price,
            f"Stop loss moved from {old_sl_price:.4f} to {new_sl_price:.4f}",
            0
        )
    
    def close_position(self, symbol: str, close_price: float, reason: str = "Manual close"):
        """Close a position and calculate final P&L"""
        if symbol not in self.active_positions:
            raise ValueError(f"No active position found for {symbol}")
        
        position = self.active_positions[symbol]
        final_pnl = (close_price - position.entry_price) * position.position_size
        
        # Log closure
        self._log_risk_event(
            symbol,
            'position_closed',
            close_price,
            f"Position closed: {reason}. Final P&L: {final_pnl:.2f}",
            final_pnl
        )
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE position_risk SET status = 'closed' 
                WHERE symbol = ? AND status = 'active'
            """, (symbol,))
        
        # Remove from active positions
        del self.active_positions[symbol]
        
        self.logger.info(f"Position closed for {symbol} at {close_price}. P&L: {final_pnl:.2f}")
        return final_pnl
    
    def _store_position_risk(self, position: PositionRisk):
        """Store position risk configuration in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO position_risk (
                    symbol, entry_price, position_size, entry_time,
                    take_profits, stop_loss, trailing_stop, max_risk_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol,
                position.entry_price,
                position.position_size,
                position.entry_time,
                json.dumps([asdict(tp) for tp in position.take_profits]),
                json.dumps(asdict(position.stop_loss)),
                json.dumps(position.trailing_stop),
                position.max_risk_pct
            ))
    
    def _store_risk_metrics(self, metrics: RiskMetrics):
        """Store risk metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO risk_metrics_history (
                    symbol, timestamp, current_price, unrealized_pnl,
                    distance_to_sl, distance_to_tp, atr_value, risk_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol,
                datetime.now(),
                metrics.current_price,
                metrics.unrealized_pnl,
                metrics.distance_to_sl,
                metrics.distance_to_next_tp,
                metrics.atr_value,
                json.dumps(asdict(metrics))
            ))
    
    def _log_risk_event(self, symbol: str, event_type: str, price: float, 
                       description: str, pnl: float):
        """Log risk management events"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO risk_events (
                    symbol, event_type, event_time, price, description, pnl,
                    position_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, event_type, datetime.now(), price, description, pnl,
                json.dumps(asdict(self.active_positions.get(symbol, {})))
            ))
    
    def get_risk_events(self, symbol: str = None, days: int = 7) -> pd.DataFrame:
        """Get risk management events history"""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                query = """
                    SELECT * FROM risk_events 
                    WHERE symbol = ? AND event_time >= ?
                    ORDER BY event_time DESC
                """
                params = (symbol, datetime.now() - timedelta(days=days))
            else:
                query = """
                    SELECT * FROM risk_events 
                    WHERE event_time >= ?
                    ORDER BY event_time DESC
                """
                params = (datetime.now() - timedelta(days=days),)
            
            return pd.read_sql_query(query, conn, params=params)
    
    def load_active_positions(self):
        """Load active positions from database on startup"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM position_risk WHERE status = 'active'"
            df = pd.read_sql_query(query, conn)
            
            for _, row in df.iterrows():
                # Reconstruct position object
                take_profits = [TakeProfitLevel(**tp) for tp in json.loads(row['take_profits'])]
                stop_loss_data = json.loads(row['stop_loss'])
                stop_loss = StopLossConfig(**stop_loss_data)
                trailing_stop = json.loads(row['trailing_stop']) if row['trailing_stop'] else None
                
                position = PositionRisk(
                    symbol=row['symbol'],
                    entry_price=row['entry_price'],
                    position_size=row['position_size'],
                    entry_time=datetime.fromisoformat(row['entry_time']),
                    take_profits=take_profits,
                    stop_loss=stop_loss,
                    trailing_stop=trailing_stop,
                    max_risk_pct=row['max_risk_pct']
                )
                
                self.active_positions[row['symbol']] = position