#!/usr/bin/env python3
"""
Advanced Risk Management Engine
Real-time portfolio protection and risk assessment system
"""

import sqlite3
import os
import ccxt
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

class RiskManagementEngine:
    def __init__(self):
        self.exchange = self.connect_okx()
        self.risk_monitoring_active = True
        self.risk_limits = {
            'max_portfolio_risk_pct': 5.0,  # Max 5% portfolio risk per day
            'max_position_size_pct': 2.0,   # Max 2% per position
            'max_daily_trades': 20,         # Max 20 trades per day
            'max_consecutive_losses': 3,    # Stop after 3 consecutive losses
            'min_usdt_reserve_pct': 20.0,   # Keep 20% in USDT
            'volatility_threshold': 8.0     # Pause trading if volatility > 8%
        }
        
    def connect_okx(self):
        """Connect to OKX exchange"""
        try:
            return ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'rateLimit': 2000,
                'enableRateLimit': True,
            })
        except Exception as e:
            print(f"Risk engine OKX connection error: {e}")
            return None
    
    def calculate_portfolio_risk(self) -> Dict:
        """Calculate current portfolio risk metrics"""
        if not self.exchange:
            return {'total_risk': 0, 'error': 'No exchange connection'}
        
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            
            total_value = usdt_balance
            position_risks = []
            
            for currency in balance:
                if currency != 'USDT' and balance[currency]['free'] > 0:
                    amount = float(balance[currency]['free'])
                    if amount > 0:
                        try:
                            symbol = f"{currency}/USDT"
                            ticker = self.exchange.fetch_ticker(symbol)
                            price = float(ticker['last'])
                            value = amount * price
                            total_value += value
                            
                            # Calculate position risk (% of portfolio)
                            position_risk_pct = (value / total_value) * 100
                            
                            position_risks.append({
                                'symbol': currency,
                                'value': value,
                                'risk_pct': position_risk_pct,
                                'price': price,
                                'amount': amount
                            })
                        except:
                            continue
            
            # Calculate overall risk metrics
            total_crypto_value = sum([pos['value'] for pos in position_risks])
            crypto_allocation_pct = (total_crypto_value / total_value) * 100
            usdt_reserve_pct = (usdt_balance / total_value) * 100
            
            # Risk assessment
            risk_level = 'low'
            if crypto_allocation_pct > 80:
                risk_level = 'high'
            elif crypto_allocation_pct > 60:
                risk_level = 'medium'
            
            return {
                'total_portfolio_value': total_value,
                'usdt_balance': usdt_balance,
                'usdt_reserve_pct': usdt_reserve_pct,
                'crypto_allocation_pct': crypto_allocation_pct,
                'position_risks': position_risks,
                'risk_level': risk_level,
                'max_position_risk': max([pos['risk_pct'] for pos in position_risks]) if position_risks else 0
            }
            
        except Exception as e:
            return {'total_risk': 0, 'error': str(e)}
    
    def assess_trade_risk(self, symbol: str, side: str, amount_usdt: float) -> Dict:
        """Assess risk of a potential trade"""
        portfolio_risk = self.calculate_portfolio_risk()
        
        if 'error' in portfolio_risk:
            return {'allowed': False, 'reason': 'Portfolio risk calculation failed'}
        
        total_value = portfolio_risk['total_portfolio_value']
        trade_risk_pct = (amount_usdt / total_value) * 100
        
        # Check position size limit
        if trade_risk_pct > self.risk_limits['max_position_size_pct']:
            return {
                'allowed': False,
                'reason': f'Trade size {trade_risk_pct:.1f}% exceeds limit {self.risk_limits["max_position_size_pct"]}%'
            }
        
        # Check USDT reserve requirement
        if side == 'buy':
            new_usdt_pct = ((portfolio_risk['usdt_balance'] - amount_usdt) / total_value) * 100
            if new_usdt_pct < self.risk_limits['min_usdt_reserve_pct']:
                return {
                    'allowed': False,
                    'reason': f'Trade would reduce USDT reserve below {self.risk_limits["min_usdt_reserve_pct"]}%'
                }
        
        # Check daily trade limit
        daily_trades = self.get_daily_trade_count()
        if daily_trades >= self.risk_limits['max_daily_trades']:
            return {
                'allowed': False,
                'reason': f'Daily trade limit reached ({daily_trades}/{self.risk_limits["max_daily_trades"]})'
            }
        
        # Check consecutive losses
        consecutive_losses = self.get_consecutive_losses()
        if consecutive_losses >= self.risk_limits['max_consecutive_losses']:
            return {
                'allowed': False,
                'reason': f'Consecutive loss limit reached ({consecutive_losses})'
            }
        
        return {
            'allowed': True,
            'trade_risk_pct': trade_risk_pct,
            'remaining_daily_trades': self.risk_limits['max_daily_trades'] - daily_trades,
            'portfolio_impact': f'{trade_risk_pct:.2f}% of portfolio'
        }
    
    def get_daily_trade_count(self) -> int:
        """Get number of trades executed today"""
        try:
            if not os.path.exists('live_trading.db'):
                return 0
            
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT COUNT(*) FROM live_trades 
                WHERE date(timestamp) = ?
            ''', (today,))
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
            
        except Exception:
            return 0
    
    def get_consecutive_losses(self) -> int:
        """Calculate consecutive losing trades"""
        try:
            if not os.path.exists('live_trading.db'):
                return 0
            
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # Get recent trades (simplified - assumes loss tracking exists)
            cursor.execute('''
                SELECT symbol, side, amount, price, timestamp
                FROM live_trades 
                ORDER BY timestamp DESC LIMIT 10
            ''')
            
            trades = cursor.fetchall()
            conn.close()
            
            # Simple heuristic: count recent buy trades as potential losses
            # In production, this would use actual P&L data
            consecutive_losses = 0
            for trade in trades:
                if trade[1] == 'buy':  # Buy trades without corresponding sells
                    consecutive_losses += 1
                else:
                    break
            
            return min(consecutive_losses, 3)  # Cap at 3 for safety
            
        except Exception:
            return 0
    
    def check_market_volatility(self) -> Dict:
        """Check current market volatility levels"""
        if not self.exchange:
            return {'volatility_ok': True, 'reason': 'No market data'}
        
        try:
            symbols = ['BTC/USDT', 'ETH/USDT']
            volatilities = []
            
            for symbol in symbols:
                ticker = self.exchange.fetch_ticker(symbol)
                change_24h = abs(float(ticker['percentage'] or 0))
                volatilities.append(change_24h)
            
            max_volatility = max(volatilities) if volatilities else 0
            
            if max_volatility > self.risk_limits['volatility_threshold']:
                return {
                    'volatility_ok': False,
                    'max_volatility': max_volatility,
                    'reason': f'Market volatility {max_volatility:.1f}% exceeds threshold {self.risk_limits["volatility_threshold"]}%'
                }
            
            return {
                'volatility_ok': True,
                'max_volatility': max_volatility,
                'status': 'Normal market conditions'
            }
            
        except Exception as e:
            return {'volatility_ok': True, 'reason': f'Volatility check failed: {e}'}
    
    def emergency_stop_conditions(self) -> Dict:
        """Check for emergency stop conditions"""
        portfolio_risk = self.calculate_portfolio_risk()
        volatility_check = self.check_market_volatility()
        daily_trades = self.get_daily_trade_count()
        
        emergency_triggers = []
        
        # Portfolio risk too high
        if portfolio_risk.get('crypto_allocation_pct', 0) > 90:
            emergency_triggers.append('Portfolio over-allocated to crypto (>90%)')
        
        # USDT reserve too low
        if portfolio_risk.get('usdt_reserve_pct', 0) < 5:
            emergency_triggers.append('USDT reserves critically low (<5%)')
        
        # Excessive volatility
        if not volatility_check['volatility_ok']:
            emergency_triggers.append(volatility_check['reason'])
        
        # Too many trades
        if daily_trades > self.risk_limits['max_daily_trades'] * 0.8:
            emergency_triggers.append(f'Approaching daily trade limit ({daily_trades}/{self.risk_limits["max_daily_trades"]})')
        
        return {
            'emergency_stop': len(emergency_triggers) > 0,
            'triggers': emergency_triggers,
            'risk_assessment': portfolio_risk,
            'volatility_status': volatility_check
        }
    
    def log_risk_event(self, event_type: str, details: Dict):
        """Log risk management events"""
        try:
            conn = sqlite3.connect('trading_platform.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    details TEXT,
                    severity TEXT
                )
            ''')
            
            severity = 'high' if 'emergency' in event_type.lower() else 'medium'
            
            cursor.execute('''
                INSERT INTO risk_log (timestamp, event_type, details, severity)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                event_type,
                json.dumps(details),
                severity
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Risk logging error: {e}")
    
    def start_risk_monitoring(self, check_interval_seconds: int = 60):
        """Start continuous risk monitoring"""
        def risk_monitor():
            while self.risk_monitoring_active:
                try:
                    emergency_status = self.emergency_stop_conditions()
                    
                    if emergency_status['emergency_stop']:
                        print(f"⚠️  RISK ALERT: {', '.join(emergency_status['triggers'])}")
                        self.log_risk_event('emergency_alert', emergency_status)
                    
                    time.sleep(check_interval_seconds)
                    
                except Exception as e:
                    print(f"Risk monitoring error: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=risk_monitor, daemon=True)
        monitor_thread.start()
        print(f"Risk monitoring started (interval: {check_interval_seconds}s)")
        
        return monitor_thread

def main():
    """Test risk management engine"""
    risk_engine = RiskManagementEngine()
    
    print("RISK MANAGEMENT ENGINE TEST")
    print("=" * 40)
    
    # Portfolio risk assessment
    portfolio_risk = risk_engine.calculate_portfolio_risk()
    print(f"Portfolio Value: ${portfolio_risk.get('total_portfolio_value', 0):.2f}")
    print(f"Risk Level: {portfolio_risk.get('risk_level', 'unknown')}")
    print(f"USDT Reserve: {portfolio_risk.get('usdt_reserve_pct', 0):.1f}%")
    
    # Emergency conditions check
    emergency = risk_engine.emergency_stop_conditions()
    if emergency['emergency_stop']:
        print(f"⚠️  EMERGENCY TRIGGERS: {emergency['triggers']}")
    else:
        print("✅ No emergency conditions detected")
    
    print("=" * 40)

if __name__ == '__main__':
    main()