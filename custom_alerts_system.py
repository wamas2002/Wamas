"""
Custom Alerts System
Real-time notifications for price targets, AI signals, and portfolio events using authentic OKX data
"""

import sqlite3
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    id: int
    alert_type: str
    symbol: str
    condition: str
    target_value: float
    current_value: float
    message: str
    is_active: bool
    created_at: str
    triggered_at: Optional[str] = None

class CustomAlertsSystem:
    def __init__(self):
        self.alerts_db = 'data/alerts.db'
        self.portfolio_db = 'data/portfolio_tracking.db'
        self.trading_db = 'data/trading_data.db'
        self.ai_db = 'data/ai_performance.db'
        self.monitoring_active = False
        self.monitor_thread = None
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize alerts database with proper schema"""
        try:
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    current_value REAL DEFAULT 0,
                    message TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TEXT,
                    user_email TEXT,
                    notification_method TEXT DEFAULT 'system'
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER,
                    symbol TEXT,
                    alert_type TEXT,
                    condition TEXT,
                    target_value REAL,
                    actual_value REAL,
                    message TEXT,
                    triggered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (alert_id) REFERENCES alerts (id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Alerts database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing alerts database: {e}")
    
    def create_price_alert(self, symbol: str, condition: str, target_price: float, message: str = None) -> int:
        """Create a price-based alert for a specific symbol"""
        try:
            if not message:
                message = f"{symbol} price {condition} ${target_price:.2f}"
            
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (alert_type, symbol, condition, target_value, message)
                VALUES (?, ?, ?, ?, ?)
            """, ('price', symbol, condition, target_price, message))
            
            alert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Created price alert for {symbol}: {condition} ${target_price}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating price alert: {e}")
            return -1
    
    def create_portfolio_alert(self, condition: str, target_value: float, message: str = None) -> int:
        """Create a portfolio value alert"""
        try:
            if not message:
                message = f"Portfolio value {condition} ${target_value:.2f}"
            
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (alert_type, symbol, condition, target_value, message)
                VALUES (?, ?, ?, ?, ?)
            """, ('portfolio', 'PORTFOLIO', condition, target_value, message))
            
            alert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Created portfolio alert: {condition} ${target_value}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating portfolio alert: {e}")
            return -1
    
    def create_ai_alert(self, model_type: str, condition: str, target_accuracy: float, message: str = None) -> int:
        """Create an AI model performance alert"""
        try:
            if not message:
                message = f"{model_type} accuracy {condition} {target_accuracy:.1f}%"
            
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (alert_type, symbol, condition, target_value, message)
                VALUES (?, ?, ?, ?, ?)
            """, ('ai_performance', model_type, condition, target_accuracy, message))
            
            alert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Created AI performance alert for {model_type}: {condition} {target_accuracy}%")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating AI alert: {e}")
            return -1
    
    def create_volatility_alert(self, symbol: str, volatility_threshold: float, message: str = None) -> int:
        """Create a volatility spike alert"""
        try:
            if not message:
                message = f"{symbol} volatility spike above {volatility_threshold:.1f}%"
            
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (alert_type, symbol, condition, target_value, message)
                VALUES (?, ?, ?, ?, ?)
            """, ('volatility', symbol, 'above', volatility_threshold, message))
            
            alert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Created volatility alert for {symbol}: above {volatility_threshold}%")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating volatility alert: {e}")
            return -1
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from OKX data"""
        try:
            from okx_account_integration import OKXAccountIntegration
            okx = OKXAccountIntegration()
            
            # Convert symbol format for OKX API
            okx_symbol = f"{symbol}-USDT" if not symbol.endswith('-USDT') else symbol
            price = okx.get_current_price(okx_symbol)
            
            return price if price else None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_portfolio_value(self) -> Optional[float]:
        """Get current portfolio value from database"""
        try:
            conn = sqlite3.connect(self.portfolio_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT total_value FROM portfolio_metrics 
                ORDER BY timestamp DESC LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return None
    
    def get_ai_model_accuracy(self, model_type: str) -> Optional[float]:
        """Get current AI model accuracy"""
        try:
            conn = sqlite3.connect(self.ai_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT prediction_accuracy FROM model_evaluation_results 
                WHERE model_type = ? 
                ORDER BY evaluation_date DESC LIMIT 1
            """, (model_type,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] * 100 if result else None  # Convert to percentage
            
        except Exception as e:
            logger.error(f"Error getting AI model accuracy for {model_type}: {e}")
            return None
    
    def calculate_volatility(self, symbol: str, periods: int = 20) -> Optional[float]:
        """Calculate volatility for a symbol"""
        try:
            conn = sqlite3.connect(self.trading_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT close_price FROM ohlcv_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (f"{symbol}USDT", periods))
            
            prices = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if len(prices) < 2:
                return None
            
            # Calculate returns
            returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
            
            # Calculate volatility (standard deviation)
            import numpy as np
            volatility = np.std(returns) * 100  # Convert to percentage
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return None
    
    def check_alerts(self) -> List[Alert]:
        """Check all active alerts and trigger those that meet conditions"""
        triggered_alerts = []
        
        try:
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM alerts WHERE is_active = TRUE")
            active_alerts = cursor.fetchall()
            
            for alert_data in active_alerts:
                alert = Alert(
                    id=alert_data[0],
                    alert_type=alert_data[1],
                    symbol=alert_data[2],
                    condition=alert_data[3],
                    target_value=alert_data[4],
                    current_value=alert_data[5],
                    message=alert_data[6],
                    is_active=bool(alert_data[7]),
                    created_at=alert_data[8],
                    triggered_at=alert_data[9]
                )
                
                should_trigger = False
                current_value = None
                
                # Check different alert types
                if alert.alert_type == 'price':
                    current_value = self.get_current_price(alert.symbol)
                elif alert.alert_type == 'portfolio':
                    current_value = self.get_portfolio_value()
                elif alert.alert_type == 'ai_performance':
                    current_value = self.get_ai_model_accuracy(alert.symbol)
                elif alert.alert_type == 'volatility':
                    current_value = self.calculate_volatility(alert.symbol)
                
                if current_value is not None:
                    # Update current value
                    cursor.execute("""
                        UPDATE alerts SET current_value = ? WHERE id = ?
                    """, (current_value, alert.id))
                    
                    alert.current_value = current_value
                    
                    # Check condition
                    if alert.condition == 'above' and current_value >= alert.target_value:
                        should_trigger = True
                    elif alert.condition == 'below' and current_value <= alert.target_value:
                        should_trigger = True
                    elif alert.condition == 'equals' and abs(current_value - alert.target_value) < 0.01:
                        should_trigger = True
                
                if should_trigger:
                    # Trigger alert
                    now = datetime.now().isoformat()
                    
                    cursor.execute("""
                        UPDATE alerts SET is_active = FALSE, triggered_at = ? WHERE id = ?
                    """, (now, alert.id))
                    
                    cursor.execute("""
                        INSERT INTO alert_history 
                        (alert_id, symbol, alert_type, condition, target_value, actual_value, message, triggered_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (alert.id, alert.symbol, alert.alert_type, alert.condition, 
                          alert.target_value, current_value, alert.message, now))
                    
                    alert.triggered_at = now
                    alert.is_active = False
                    triggered_alerts.append(alert)
                    
                    logger.info(f"Alert triggered: {alert.message}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        alerts = []
        
        try:
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM alerts WHERE is_active = TRUE ORDER BY created_at DESC")
            
            for row in cursor.fetchall():
                alert = Alert(
                    id=row[0],
                    alert_type=row[1],
                    symbol=row[2],
                    condition=row[3],
                    target_value=row[4],
                    current_value=row[5],
                    message=row[6],
                    is_active=bool(row[7]),
                    created_at=row[8],
                    triggered_at=row[9]
                )
                alerts.append(alert)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
        
        return alerts
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history"""
        history = []
        
        try:
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, alert_type, condition, target_value, actual_value, 
                       message, triggered_at 
                FROM alert_history 
                ORDER BY triggered_at DESC 
                LIMIT ?
            """, (limit,))
            
            for row in cursor.fetchall():
                history.append({
                    'symbol': row[0],
                    'alert_type': row[1],
                    'condition': row[2],
                    'target_value': row[3],
                    'actual_value': row[4],
                    'message': row[5],
                    'triggered_at': row[6]
                })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
        
        return history
    
    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert"""
        try:
            conn = sqlite3.connect(self.alerts_db)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            
            affected_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            return affected_rows > 0
            
        except Exception as e:
            logger.error(f"Error deleting alert {alert_id}: {e}")
            return False
    
    def start_monitoring(self, check_interval: int = 30):
        """Start continuous monitoring of alerts"""
        if self.monitoring_active:
            logger.info("Alert monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            logger.info(f"Starting alert monitoring with {check_interval}s interval")
            
            while self.monitoring_active:
                try:
                    triggered = self.check_alerts()
                    if triggered:
                        logger.info(f"Triggered {len(triggered)} alerts")
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in alert monitoring loop: {e}")
                    time.sleep(check_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def create_default_alerts(self):
        """Create default alerts for the current portfolio"""
        try:
            # Portfolio value alerts
            current_portfolio = self.get_portfolio_value() or 156.92
            
            self.create_portfolio_alert('below', current_portfolio * 0.95, 
                                      f"Portfolio dropped 5% below ${current_portfolio * 0.95:.2f}")
            self.create_portfolio_alert('above', current_portfolio * 1.1, 
                                      f"Portfolio gained 10% above ${current_portfolio * 1.1:.2f}")
            
            # Price alerts for main holdings
            self.create_price_alert('PI', 'above', 2.0, "PI token reached $2.00")
            self.create_price_alert('PI', 'below', 1.0, "PI token dropped below $1.00")
            
            # AI performance alerts
            self.create_ai_alert('LSTM', 'below', 70.0, "LSTM model accuracy dropped below 70%")
            self.create_ai_alert('Ensemble', 'above', 85.0, "Ensemble model accuracy above 85%")
            
            # Volatility alerts
            self.create_volatility_alert('BTC', 5.0, "Bitcoin volatility spike above 5%")
            
            logger.info("Created default alerts for portfolio monitoring")
            
        except Exception as e:
            logger.error(f"Error creating default alerts: {e}")

def setup_alerts_system():
    """Initialize and configure the alerts system"""
    alerts = CustomAlertsSystem()
    
    # Create default alerts
    alerts.create_default_alerts()
    
    # Start monitoring
    alerts.start_monitoring(check_interval=60)  # Check every minute
    
    print("=" * 60)
    print("CUSTOM ALERTS SYSTEM INITIALIZED")
    print("=" * 60)
    
    active_alerts = alerts.get_active_alerts()
    print(f"Active Alerts: {len(active_alerts)}")
    
    for alert in active_alerts[:5]:  # Show first 5
        print(f"  {alert.alert_type.upper()}: {alert.message}")
        print(f"    Target: {alert.target_value:.2f}, Current: {alert.current_value:.2f}")
    
    print(f"\nMonitoring started - checking alerts every 60 seconds")
    print("=" * 60)
    
    return alerts

if __name__ == "__main__":
    setup_alerts_system()