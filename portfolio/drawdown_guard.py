"""
Drawdown Guard - Equity Curve Protection System
Monitors portfolio drawdown and triggers protective measures when thresholds are exceeded
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import threading
import time

logger = logging.getLogger(__name__)

class DrawdownGuard:
    """Monitor equity curve drawdown and implement protective measures"""
    
    def __init__(self):
        self.max_drawdown_threshold = 8.0  # 8% maximum drawdown
        self.rolling_window_trades = 10    # Monitor last 10 trades
        self.soft_pause_threshold = 6.0    # Soft pause at 6% drawdown
        self.emergency_stop_threshold = 12.0  # Emergency stop at 12%
        
        # Protection states
        self.protection_active = False
        self.protection_level = 'none'  # none, soft, emergency
        self.pause_start_time = None
        self.recovery_threshold = 2.0  # Resume when drawdown < 2%
        
        # Equity tracking
        self.equity_history = []
        self.drawdown_history = []
        
        self.setup_database()
    
    def setup_database(self):
        """Initialize drawdown protection database"""
        try:
            conn = sqlite3.connect('drawdown_protection.db')
            cursor = conn.cursor()
            
            # Equity curve tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    portfolio_value REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    total_equity REAL,
                    
                    -- Drawdown metrics
                    peak_equity REAL,
                    current_drawdown_pct REAL,
                    max_drawdown_pct REAL,
                    drawdown_duration_hours REAL,
                    
                    -- Risk status
                    protection_level TEXT DEFAULT 'none',
                    risk_score REAL
                )
            ''')
            
            # Drawdown events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drawdown_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_start DATETIME,
                    event_end DATETIME,
                    max_drawdown_pct REAL,
                    duration_hours REAL,
                    
                    -- Triggers
                    trigger_level TEXT,
                    protection_activated BOOLEAN,
                    recovery_method TEXT,
                    
                    -- Impact
                    trades_paused INTEGER,
                    opportunity_cost REAL,
                    notes TEXT
                )
            ''')
            
            # Protection actions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS protection_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action_type TEXT,
                    trigger_drawdown_pct REAL,
                    protection_level TEXT,
                    
                    -- Action details
                    action_taken TEXT,
                    expected_duration REAL,
                    actual_duration REAL,
                    effectiveness_score REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Drawdown protection database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def calculate_portfolio_equity(self, balance_data: Dict, trade_history: List[Dict]) -> Dict:
        """Calculate current portfolio equity and drawdown metrics"""
        try:
            # Get current portfolio value
            portfolio_value = balance_data.get('USDT', {}).get('total', 0)
            
            # Add value of other holdings
            for symbol, balance in balance_data.items():
                if symbol != 'USDT' and balance.get('total', 0) > 0:
                    # This would normally fetch current market price
                    # For now, estimate based on recent trades
                    recent_trades = [t for t in trade_history[-10:] if t.get('symbol', '').startswith(symbol)]
                    if recent_trades:
                        avg_price = np.mean([t.get('price', 0) for t in recent_trades])
                        portfolio_value += balance['total'] * avg_price
            
            # Calculate realized PnL from trade history
            realized_pnl = sum([t.get('pnl_pct', 0) / 100 * t.get('position_size', 0) 
                              for t in trade_history if t.get('win_loss')])
            
            # Estimate unrealized PnL (simplified)
            unrealized_pnl = 0  # Would calculate from open positions
            
            total_equity = portfolio_value + realized_pnl + unrealized_pnl
            
            # Update equity history
            equity_record = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_equity': total_equity
            }
            
            self.equity_history.append(equity_record)
            
            # Keep only recent history
            if len(self.equity_history) > 500:
                self.equity_history = self.equity_history[-500:]
            
            return equity_record
            
        except Exception as e:
            logger.error(f"Equity calculation failed: {e}")
            return {'total_equity': 0, 'timestamp': datetime.now()}
    
    def calculate_drawdown_metrics(self, equity_history: List[Dict]) -> Dict:
        """Calculate comprehensive drawdown metrics"""
        try:
            if len(equity_history) < 2:
                return {
                    'current_drawdown_pct': 0,
                    'max_drawdown_pct': 0,
                    'peak_equity': 0,
                    'drawdown_duration_hours': 0
                }
            
            # Extract equity values
            equity_values = [e['total_equity'] for e in equity_history]
            timestamps = [e['timestamp'] for e in equity_history]
            
            # Calculate running maximum (peak)
            running_max = np.maximum.accumulate(equity_values)
            
            # Calculate drawdowns
            drawdowns = (np.array(equity_values) - running_max) / running_max
            drawdown_pcts = drawdowns * 100
            
            # Current drawdown
            current_drawdown_pct = abs(drawdown_pcts[-1])
            
            # Maximum drawdown
            max_drawdown_pct = abs(np.min(drawdown_pcts))
            
            # Peak equity
            peak_equity = running_max[-1]
            
            # Drawdown duration (time since last peak)
            last_peak_idx = np.where(running_max == peak_equity)[0][-1]
            if last_peak_idx < len(timestamps) - 1:
                drawdown_start = timestamps[last_peak_idx]
                drawdown_duration = (timestamps[-1] - drawdown_start).total_seconds() / 3600
            else:
                drawdown_duration = 0
            
            # Store drawdown history
            drawdown_record = {
                'timestamp': timestamps[-1],
                'current_drawdown_pct': current_drawdown_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'peak_equity': peak_equity,
                'drawdown_duration_hours': drawdown_duration
            }
            
            self.drawdown_history.append(drawdown_record)
            
            # Keep only recent history
            if len(self.drawdown_history) > 200:
                self.drawdown_history = self.drawdown_history[-200:]
            
            return drawdown_record
            
        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}")
            return {
                'current_drawdown_pct': 0,
                'max_drawdown_pct': 0,
                'peak_equity': 0,
                'drawdown_duration_hours': 0
            }
    
    def assess_protection_level(self, drawdown_metrics: Dict) -> str:
        """Assess required protection level based on drawdown"""
        try:
            current_drawdown = drawdown_metrics['current_drawdown_pct']
            
            if current_drawdown >= self.emergency_stop_threshold:
                return 'emergency'
            elif current_drawdown >= self.max_drawdown_threshold:
                return 'hard'
            elif current_drawdown >= self.soft_pause_threshold:
                return 'soft'
            else:
                return 'none'
                
        except Exception as e:
            logger.error(f"Protection level assessment failed: {e}")
            return 'none'
    
    def activate_protection(self, protection_level: str, drawdown_metrics: Dict) -> Dict:
        """Activate drawdown protection measures"""
        try:
            current_drawdown = drawdown_metrics['current_drawdown_pct']
            
            protection_actions = {
                'soft': {
                    'description': 'Soft pause - Reduce position sizes by 50%',
                    'position_size_multiplier': 0.5,
                    'signal_confidence_threshold': 85.0,
                    'max_concurrent_trades': 3,
                    'review_interval_hours': 2
                },
                'hard': {
                    'description': 'Hard pause - Stop new trades, monitor only',
                    'position_size_multiplier': 0.0,
                    'signal_confidence_threshold': 95.0,
                    'max_concurrent_trades': 0,
                    'review_interval_hours': 6
                },
                'emergency': {
                    'description': 'Emergency stop - Close all positions, full system pause',
                    'position_size_multiplier': 0.0,
                    'signal_confidence_threshold': 100.0,
                    'max_concurrent_trades': 0,
                    'review_interval_hours': 24,
                    'close_all_positions': True
                }
            }
            
            if protection_level not in protection_actions:
                return {'success': False, 'message': 'Invalid protection level'}
            
            action = protection_actions[protection_level]
            
            # Update protection state
            self.protection_active = True
            self.protection_level = protection_level
            self.pause_start_time = datetime.now()
            
            # Log protection activation
            self.log_protection_action(
                'activate',
                current_drawdown,
                protection_level,
                action['description']
            )
            
            logger.warning(f"Drawdown protection activated: {protection_level} "
                          f"(Drawdown: {current_drawdown:.2f}%)")
            
            return {
                'success': True,
                'protection_level': protection_level,
                'action': action,
                'trigger_drawdown': current_drawdown,
                'message': action['description']
            }
            
        except Exception as e:
            logger.error(f"Protection activation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_recovery_conditions(self, drawdown_metrics: Dict) -> bool:
        """Check if conditions are met for recovery from protection"""
        try:
            current_drawdown = drawdown_metrics['current_drawdown_pct']
            
            # Recovery conditions
            if current_drawdown <= self.recovery_threshold:
                return True
            
            # Time-based recovery for soft protection
            if self.protection_level == 'soft' and self.pause_start_time:
                hours_paused = (datetime.now() - self.pause_start_time).total_seconds() / 3600
                if hours_paused >= 4:  # Auto-recovery after 4 hours
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Recovery condition check failed: {e}")
            return False
    
    def deactivate_protection(self, reason: str = 'recovery') -> Dict:
        """Deactivate drawdown protection"""
        try:
            if not self.protection_active:
                return {'success': True, 'message': 'Protection not active'}
            
            # Calculate protection duration
            if self.pause_start_time:
                duration_hours = (datetime.now() - self.pause_start_time).total_seconds() / 3600
            else:
                duration_hours = 0
            
            # Log protection deactivation
            self.log_protection_action(
                'deactivate',
                0,  # Current drawdown would be calculated
                self.protection_level,
                f"Protection deactivated: {reason}",
                duration_hours
            )
            
            # Reset protection state
            previous_level = self.protection_level
            self.protection_active = False
            self.protection_level = 'none'
            self.pause_start_time = None
            
            logger.info(f"Drawdown protection deactivated: {previous_level} -> none ({reason})")
            
            return {
                'success': True,
                'previous_level': previous_level,
                'duration_hours': duration_hours,
                'reason': reason,
                'message': f"Protection deactivated after {duration_hours:.1f} hours"
            }
            
        except Exception as e:
            logger.error(f"Protection deactivation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def should_block_trade(self, signal_confidence: float, position_size: float) -> Tuple[bool, str]:
        """Determine if trade should be blocked due to drawdown protection"""
        try:
            if not self.protection_active:
                return False, "No protection active"
            
            protection_actions = {
                'soft': {
                    'min_confidence': 85.0,
                    'max_position_multiplier': 0.5
                },
                'hard': {
                    'min_confidence': 95.0,
                    'max_position_multiplier': 0.0
                },
                'emergency': {
                    'min_confidence': 100.0,
                    'max_position_multiplier': 0.0
                }
            }
            
            action = protection_actions.get(self.protection_level, {})
            
            # Check confidence threshold
            min_confidence = action.get('min_confidence', 0)
            if signal_confidence < min_confidence:
                return True, f"Signal confidence {signal_confidence:.1f}% below {min_confidence:.1f}% threshold"
            
            # Check position size
            max_multiplier = action.get('max_position_multiplier', 1.0)
            if max_multiplier == 0.0:
                return True, f"All trading paused ({self.protection_level} protection)"
            
            return False, "Trade allowed under protection"
            
        except Exception as e:
            logger.error(f"Trade blocking check failed: {e}")
            return True, "Error in protection check - blocking trade"
    
    def adjust_position_size(self, original_size: float) -> float:
        """Adjust position size based on protection level"""
        try:
            if not self.protection_active:
                return original_size
            
            multipliers = {
                'soft': 0.5,
                'hard': 0.0,
                'emergency': 0.0
            }
            
            multiplier = multipliers.get(self.protection_level, 1.0)
            adjusted_size = original_size * multiplier
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Position size adjustment failed: {e}")
            return original_size * 0.5  # Conservative fallback
    
    def log_protection_action(self, action_type: str, trigger_drawdown: float, 
                            protection_level: str, action_description: str, 
                            duration: float = None):
        """Log protection action to database"""
        try:
            conn = sqlite3.connect('drawdown_protection.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO protection_actions (
                    action_type, trigger_drawdown_pct, protection_level,
                    action_taken, actual_duration
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                action_type, trigger_drawdown, protection_level,
                action_description, duration
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log protection action: {e}")
    
    def save_equity_data(self, equity_record: Dict, drawdown_metrics: Dict):
        """Save equity and drawdown data to database"""
        try:
            conn = sqlite3.connect('drawdown_protection.db')
            cursor = conn.cursor()
            
            # Calculate risk score
            risk_score = min(100, drawdown_metrics['current_drawdown_pct'] * 10)
            
            cursor.execute('''
                INSERT INTO equity_curve (
                    portfolio_value, unrealized_pnl, realized_pnl, total_equity,
                    peak_equity, current_drawdown_pct, max_drawdown_pct,
                    drawdown_duration_hours, protection_level, risk_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                equity_record['portfolio_value'],
                equity_record['unrealized_pnl'],
                equity_record['realized_pnl'],
                equity_record['total_equity'],
                drawdown_metrics['peak_equity'],
                drawdown_metrics['current_drawdown_pct'],
                drawdown_metrics['max_drawdown_pct'],
                drawdown_metrics['drawdown_duration_hours'],
                self.protection_level,
                risk_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save equity data: {e}")
    
    def comprehensive_drawdown_check(self, balance_data: Dict, trade_history: List[Dict]) -> Dict:
        """Perform comprehensive drawdown analysis and protection check"""
        try:
            # Calculate current equity
            equity_record = self.calculate_portfolio_equity(balance_data, trade_history)
            
            # Calculate drawdown metrics
            drawdown_metrics = self.calculate_drawdown_metrics(self.equity_history)
            
            # Save data
            self.save_equity_data(equity_record, drawdown_metrics)
            
            # Assess protection needs
            required_protection = self.assess_protection_level(drawdown_metrics)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'equity_data': equity_record,
                'drawdown_metrics': drawdown_metrics,
                'protection_status': {
                    'active': self.protection_active,
                    'level': self.protection_level,
                    'required_level': required_protection
                },
                'actions_taken': []
            }
            
            # Handle protection state changes
            if not self.protection_active and required_protection != 'none':
                # Activate protection
                activation_result = self.activate_protection(required_protection, drawdown_metrics)
                result['actions_taken'].append(activation_result)
                
            elif self.protection_active:
                # Check for recovery
                if self.check_recovery_conditions(drawdown_metrics):
                    deactivation_result = self.deactivate_protection('recovery')
                    result['actions_taken'].append(deactivation_result)
                elif required_protection != self.protection_level:
                    # Adjust protection level
                    self.deactivate_protection('level_change')
                    activation_result = self.activate_protection(required_protection, drawdown_metrics)
                    result['actions_taken'].append(activation_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive drawdown check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'protection_status': {'active': self.protection_active, 'level': self.protection_level}
            }
    
    def get_protection_insights(self) -> List[str]:
        """Get insights about drawdown protection status"""
        try:
            insights = []
            
            if self.protection_active:
                insights.append(f"Drawdown protection ACTIVE: {self.protection_level}")
                if self.pause_start_time:
                    duration = (datetime.now() - self.pause_start_time).total_seconds() / 3600
                    insights.append(f"Protection duration: {duration:.1f} hours")
            else:
                insights.append("Drawdown protection: INACTIVE")
            
            # Recent drawdown info
            if self.drawdown_history:
                latest = self.drawdown_history[-1]
                insights.append(f"Current drawdown: {latest['current_drawdown_pct']:.2f}%")
                insights.append(f"Maximum drawdown: {latest['max_drawdown_pct']:.2f}%")
            
            # Protection thresholds
            insights.append(f"Thresholds: Soft {self.soft_pause_threshold}%, Hard {self.max_drawdown_threshold}%, Emergency {self.emergency_stop_threshold}%")
            
            return insights
            
        except Exception as e:
            logger.error(f"Protection insights failed: {e}")
            return ["Drawdown protection monitoring active"]
    
    def get_protection_status(self) -> Dict:
        """Get current protection status"""
        return {
            'protection_active': self.protection_active,
            'protection_level': self.protection_level,
            'pause_start_time': self.pause_start_time.isoformat() if self.pause_start_time else None,
            'thresholds': {
                'soft': self.soft_pause_threshold,
                'hard': self.max_drawdown_threshold,
                'emergency': self.emergency_stop_threshold,
                'recovery': self.recovery_threshold
            }
        }