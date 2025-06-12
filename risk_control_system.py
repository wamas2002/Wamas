#!/usr/bin/env python3
"""
Automated Risk Control System
Implements stop losses, position sizing, and portfolio risk management
"""

import sqlite3
import json
import os
import ccxt
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskControlSystem:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        self.setup_risk_database()
        
        # Risk parameters
        self.max_position_percent = 25.0  # Maximum 25% per asset
        self.stop_loss_percent = 8.0      # 8% stop loss
        self.max_portfolio_risk = 15.0    # Maximum 15% total portfolio risk
        self.rebalance_threshold = 5.0    # Rebalance if >5% deviation
        
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            api_key = os.getenv('OKX_API_KEY')
            secret = os.getenv('OKX_SECRET_KEY')
            passphrase = os.getenv('OKX_PASSPHRASE')
            
            if api_key and secret and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret,
                    'password': passphrase,
                    'sandbox': False,
                    'enableRateLimit': True,
                })
                logger.info("✓ Risk control system connected to OKX")
            else:
                logger.warning("⚠ OKX credentials not configured")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_risk_database(self):
        """Setup risk management database tables"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Enhanced stop losses table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS active_stop_losses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        position_size REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_price REAL NOT NULL,
                        current_price REAL,
                        stop_loss_percent REAL NOT NULL,
                        active BOOLEAN DEFAULT 1,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        triggered_at DATETIME,
                        pnl_at_trigger REAL
                    )
                ''')
                
                # Position limits table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS position_limits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT UNIQUE NOT NULL,
                        max_position_percent REAL NOT NULL,
                        current_position_percent REAL,
                        limit_breached BOOLEAN DEFAULT 0,
                        last_check DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Risk events log
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        symbol TEXT,
                        description TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        action_taken TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("✓ Risk management database initialized")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_current_portfolio(self):
        """Get current portfolio from OKX"""
        if not self.exchange:
            return {}
        
        try:
            balance = self.exchange.fetch_balance()
            portfolio = {}
            total_value_usd = 0
            
            for symbol, amount in balance['total'].items():
                if amount > 0:
                    try:
                        if symbol == 'USDT':
                            price = 1.0
                        else:
                            ticker = self.exchange.fetch_ticker(f'{symbol}/USDT')
                            price = ticker['last']
                        
                        value_usd = amount * price
                        total_value_usd += value_usd
                        
                        portfolio[symbol] = {
                            'amount': amount,
                            'price': price,
                            'value_usd': value_usd
                        }
                    except:
                        continue
            
            # Calculate percentages
            for symbol in portfolio:
                portfolio[symbol]['percent'] = (portfolio[symbol]['value_usd'] / total_value_usd * 100) if total_value_usd > 0 else 0
            
            return portfolio, total_value_usd
            
        except Exception as e:
            logger.error(f"Portfolio fetch failed: {e}")
            return {}, 0
    
    def check_position_limits(self):
        """Check and enforce position sizing limits"""
        portfolio, total_value = self.get_current_portfolio()
        
        if not portfolio:
            return []
        
        violations = []
        current_time = datetime.now()
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                for symbol, data in portfolio.items():
                    position_percent = data['percent']
                    
                    # Update position limits table
                    cursor.execute('''
                        INSERT OR REPLACE INTO position_limits 
                        (symbol, max_position_percent, current_position_percent, limit_breached, last_check)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (symbol, self.max_position_percent, position_percent, 
                          position_percent > self.max_position_percent, current_time))
                    
                    # Check for violations
                    if position_percent > self.max_position_percent:
                        violation = {
                            'symbol': symbol,
                            'current_percent': position_percent,
                            'max_percent': self.max_position_percent,
                            'excess_percent': position_percent - self.max_position_percent,
                            'suggested_reduction': data['value_usd'] * (position_percent - self.max_position_percent) / 100
                        }
                        violations.append(violation)
                        
                        # Log risk event
                        cursor.execute('''
                            INSERT INTO risk_events 
                            (event_type, symbol, description, risk_level, action_taken)
                            VALUES (?, ?, ?, ?, ?)
                        ''', ('POSITION_LIMIT_BREACH', symbol, 
                              f'Position {position_percent:.1f}% exceeds limit {self.max_position_percent}%',
                              'HIGH', 'ALERT_GENERATED'))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Position limit check failed: {e}")
        
        return violations
    
    def create_stop_losses(self):
        """Create stop losses for current positions"""
        portfolio, total_value = self.get_current_portfolio()
        
        if not portfolio:
            return []
        
        stop_losses_created = []
        current_time = datetime.now()
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                for symbol, data in portfolio.items():
                    if symbol == 'USDT':  # Skip USDT
                        continue
                    
                    current_price = data['price']
                    position_size = data['amount']
                    
                    # Calculate stop loss price (8% below current price)
                    stop_price = current_price * (1 - self.stop_loss_percent / 100)
                    
                    # Check if stop loss already exists and is active
                    cursor.execute('''
                        SELECT id FROM active_stop_losses 
                        WHERE symbol = ? AND active = 1
                    ''', (symbol,))
                    
                    existing = cursor.fetchone()
                    
                    if not existing:
                        # Create new stop loss
                        cursor.execute('''
                            INSERT INTO active_stop_losses 
                            (symbol, position_size, entry_price, stop_price, current_price, 
                             stop_loss_percent, active)
                            VALUES (?, ?, ?, ?, ?, ?, 1)
                        ''', (symbol, position_size, current_price, stop_price, 
                              current_price, self.stop_loss_percent))
                        
                        stop_losses_created.append({
                            'symbol': symbol,
                            'position_size': position_size,
                            'current_price': current_price,
                            'stop_price': stop_price,
                            'protection_amount': position_size * (current_price - stop_price)
                        })
                        
                        # Log risk event
                        cursor.execute('''
                            INSERT INTO risk_events 
                            (event_type, symbol, description, risk_level, action_taken)
                            VALUES (?, ?, ?, ?, ?)
                        ''', ('STOP_LOSS_CREATED', symbol, 
                              f'Stop loss at ${stop_price:.4f} ({self.stop_loss_percent}% protection)',
                              'MEDIUM', 'STOP_LOSS_ACTIVATED'))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Stop loss creation failed: {e}")
        
        return stop_losses_created
    
    def monitor_stop_losses(self):
        """Monitor and trigger stop losses when needed"""
        if not self.exchange:
            return []
        
        triggered_stops = []
        current_time = datetime.now()
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Get all active stop losses
                cursor.execute('''
                    SELECT id, symbol, position_size, entry_price, stop_price, stop_loss_percent
                    FROM active_stop_losses 
                    WHERE active = 1
                ''')
                
                active_stops = cursor.fetchall()
                
                for stop in active_stops:
                    stop_id, symbol, position_size, entry_price, stop_price, stop_percent = stop
                    
                    try:
                        # Get current price
                        ticker = self.exchange.fetch_ticker(f'{symbol}/USDT')
                        current_price = ticker['last']
                        
                        # Update current price
                        cursor.execute('''
                            UPDATE active_stop_losses 
                            SET current_price = ? 
                            WHERE id = ?
                        ''', (current_price, stop_id))
                        
                        # Check if stop loss should be triggered
                        if current_price <= stop_price:
                            # Calculate P&L
                            pnl = position_size * (current_price - entry_price)
                            
                            # Mark stop loss as triggered
                            cursor.execute('''
                                UPDATE active_stop_losses 
                                SET active = 0, triggered_at = ?, pnl_at_trigger = ?
                                WHERE id = ?
                            ''', (current_time, pnl, stop_id))
                            
                            triggered_stops.append({
                                'symbol': symbol,
                                'position_size': position_size,
                                'entry_price': entry_price,
                                'trigger_price': current_price,
                                'stop_price': stop_price,
                                'pnl': pnl,
                                'loss_percent': ((current_price - entry_price) / entry_price) * 100
                            })
                            
                            # Log critical risk event
                            cursor.execute('''
                                INSERT INTO risk_events 
                                (event_type, symbol, description, risk_level, action_taken)
                                VALUES (?, ?, ?, ?, ?)
                            ''', ('STOP_LOSS_TRIGGERED', symbol, 
                                  f'Stop loss triggered at ${current_price:.4f}, P&L: ${pnl:.2f}',
                                  'CRITICAL', 'POSITION_CLOSED'))
                            
                            logger.warning(f"🚨 STOP LOSS TRIGGERED: {symbol} at ${current_price:.4f}")
                    
                    except Exception as e:
                        logger.error(f"Stop loss monitoring failed for {symbol}: {e}")
                        continue
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Stop loss monitoring failed: {e}")
        
        return triggered_stops
    
    def calculate_portfolio_risk(self):
        """Calculate overall portfolio risk metrics"""
        portfolio, total_value = self.get_current_portfolio()
        
        if not portfolio:
            return {}
        
        try:
            # Calculate concentration risk
            position_percentages = [data['percent'] for data in portfolio.values()]
            max_concentration = max(position_percentages)
            
            # Calculate diversification score (1 = perfect diversification)
            n_assets = len(portfolio)
            ideal_allocation = 100 / n_assets if n_assets > 0 else 0
            diversification_score = 100 - sum(abs(p - ideal_allocation) for p in position_percentages) / 2
            
            # Calculate total risk exposure
            total_risk_percent = sum(p for p in position_percentages if p > self.max_position_percent)
            
            risk_metrics = {
                'total_portfolio_value': total_value,
                'max_concentration': max_concentration,
                'diversification_score': diversification_score,
                'total_assets': n_assets,
                'risk_exposure_percent': total_risk_percent,
                'position_limit_breaches': sum(1 for p in position_percentages if p > self.max_position_percent),
                'risk_level': self._assess_risk_level(max_concentration, diversification_score, total_risk_percent)
            }
            
            # Save to database
            current_time = datetime.now().isoformat()
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO risk_metrics 
                    (total_portfolio_value, max_drawdown, var_1day, sharpe_ratio, 
                     diversification_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (total_value, max_concentration, total_risk_percent, 0, 
                      diversification_score, current_time))
                conn.commit()
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {}
    
    def _assess_risk_level(self, max_concentration, diversification_score, risk_exposure):
        """Assess overall portfolio risk level"""
        if max_concentration > 50 or risk_exposure > 30:
            return 'CRITICAL'
        elif max_concentration > 35 or risk_exposure > 15 or diversification_score < 50:
            return 'HIGH'
        elif max_concentration > 25 or diversification_score < 70:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_rebalancing_recommendations(self):
        """Generate portfolio rebalancing recommendations"""
        portfolio, total_value = self.get_current_portfolio()
        
        if not portfolio:
            return []
        
        recommendations = []
        target_allocation = 100 / len(portfolio)  # Equal weight target
        
        for symbol, data in portfolio.items():
            current_percent = data['percent']
            deviation = current_percent - target_allocation
            
            if abs(deviation) > self.rebalance_threshold:
                if deviation > 0:  # Overweight
                    action = 'REDUCE'
                    amount_change = data['value_usd'] * deviation / 100
                else:  # Underweight
                    action = 'INCREASE'
                    amount_change = data['value_usd'] * abs(deviation) / 100
                
                recommendations.append({
                    'symbol': symbol,
                    'action': action,
                    'current_percent': current_percent,
                    'target_percent': target_allocation,
                    'deviation_percent': deviation,
                    'amount_change_usd': amount_change,
                    'priority': 'HIGH' if abs(deviation) > 10 else 'MEDIUM'
                })
        
        return recommendations
    
    def run_risk_management_cycle(self):
        """Run complete risk management cycle"""
        logger.info("🔍 Running comprehensive risk management cycle...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'position_violations': [],
            'stop_losses_created': [],
            'stop_losses_triggered': [],
            'portfolio_risk': {},
            'rebalancing_recommendations': []
        }
        
        # 1. Check position limits
        logger.info("Checking position limits...")
        results['position_violations'] = self.check_position_limits()
        
        # 2. Create/update stop losses
        logger.info("Creating stop losses...")
        results['stop_losses_created'] = self.create_stop_losses()
        
        # 3. Monitor existing stop losses
        logger.info("Monitoring stop losses...")
        results['stop_losses_triggered'] = self.monitor_stop_losses()
        
        # 4. Calculate portfolio risk
        logger.info("Calculating portfolio risk...")
        results['portfolio_risk'] = self.calculate_portfolio_risk()
        
        # 5. Generate rebalancing recommendations
        logger.info("Generating rebalancing recommendations...")
        results['rebalancing_recommendations'] = self.generate_rebalancing_recommendations()
        
        # Print summary
        self._print_risk_summary(results)
        
        return results
    
    def _print_risk_summary(self, results):
        """Print risk management summary"""
        print("\n" + "="*60)
        print("RISK MANAGEMENT SUMMARY")
        print("="*60)
        
        # Position violations
        violations = results['position_violations']
        if violations:
            print(f"\n🚨 POSITION LIMIT VIOLATIONS ({len(violations)}):")
            for v in violations:
                print(f"  {v['symbol']}: {v['current_percent']:.1f}% (limit: {v['max_percent']:.1f}%)")
                print(f"    Suggested reduction: ${v['suggested_reduction']:,.2f}")
        else:
            print("\n✅ No position limit violations")
        
        # Stop losses
        created = results['stop_losses_created']
        if created:
            print(f"\n🛡️ STOP LOSSES CREATED ({len(created)}):")
            for sl in created:
                print(f"  {sl['symbol']}: Stop at ${sl['stop_price']:.4f} (current: ${sl['current_price']:.4f})")
        
        triggered = results['stop_losses_triggered']
        if triggered:
            print(f"\n🚨 STOP LOSSES TRIGGERED ({len(triggered)}):")
            for sl in triggered:
                print(f"  {sl['symbol']}: Loss {sl['loss_percent']:.1f}%, P&L: ${sl['pnl']:.2f}")
        
        # Portfolio risk
        risk = results['portfolio_risk']
        if risk:
            print(f"\n📊 PORTFOLIO RISK ASSESSMENT:")
            print(f"  Total Value: ${risk.get('total_portfolio_value', 0):,.2f}")
            print(f"  Max Concentration: {risk.get('max_concentration', 0):.1f}%")
            print(f"  Diversification Score: {risk.get('diversification_score', 0):.1f}%")
            print(f"  Risk Level: {risk.get('risk_level', 'UNKNOWN')}")
        
        # Rebalancing
        rebalance = results['rebalancing_recommendations']
        if rebalance:
            print(f"\n⚖️ REBALANCING RECOMMENDATIONS ({len(rebalance)}):")
            for rec in rebalance[:5]:  # Show top 5
                print(f"  {rec['symbol']}: {rec['action']} by {abs(rec['deviation_percent']):.1f}%")

def main():
    """Main risk control function"""
    risk_system = RiskControlSystem()
    
    if not risk_system.exchange:
        print("⚠️ OKX connection required for risk management")
        return
    
    # Run risk management cycle
    results = risk_system.run_risk_management_cycle()
    
    # Save results
    with open('risk_management_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📋 Risk management report saved: risk_management_report.json")
    print("✅ Risk management cycle completed!")

if __name__ == "__main__":
    main()