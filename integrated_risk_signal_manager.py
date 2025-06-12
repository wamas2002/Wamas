#!/usr/bin/env python3
"""
Integrated Risk & Signal Management System
Combines enhanced signal generation with automated risk controls
"""

import sqlite3
import json
import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedTradingManager:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        self.setup_integrated_database()
        
        # Risk control parameters
        self.max_position_percent = 25.0
        self.stop_loss_percent = 8.0
        self.rebalance_threshold = 5.0
        
        # Signal optimization parameters
        self.min_confidence_threshold = 60.0
        self.target_signals_per_day = 15
        
        # Enhanced trading pairs
        self.trading_pairs = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
            'AVAX/USDT', 'LINK/USDT', 'ATOM/USDT', 'NEAR/USDT'
        ]
        
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
                logger.info("Integrated trading manager connected to OKX")
            else:
                logger.warning("OKX credentials not configured")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_integrated_database(self):
        """Setup integrated database tables"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Create all necessary tables if they don't exist
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
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS optimized_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        rsi REAL,
                        macd REAL,
                        bb_position REAL,
                        volume_ratio REAL,
                        current_price REAL,
                        reasoning TEXT,
                        risk_level TEXT,
                        expected_move REAL,
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Integrated database initialized")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_current_portfolio(self):
        """Get current portfolio from OKX"""
        if not self.exchange:
            return {}, 0
        
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
    
    def check_and_enforce_risk_controls(self):
        """Check and enforce all risk controls"""
        portfolio, total_value = self.get_current_portfolio()
        
        if not portfolio:
            return {'violations': [], 'stop_losses': [], 'recommendations': []}
        
        violations = []
        stop_losses_created = []
        recommendations = []
        current_time = datetime.now()
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                for symbol, data in portfolio.items():
                    position_percent = data['percent']
                    
                    # Check position limits
                    if position_percent > self.max_position_percent:
                        violation = {
                            'symbol': symbol,
                            'current_percent': position_percent,
                            'max_percent': self.max_position_percent,
                            'excess_value': data['value_usd'] * (position_percent - self.max_position_percent) / 100
                        }
                        violations.append(violation)
                        
                        recommendations.append({
                            'type': 'REDUCE_POSITION',
                            'symbol': symbol,
                            'action': f'Reduce {symbol} position by ${violation["excess_value"]:.2f}',
                            'priority': 'HIGH'
                        })
                    
                    # Create/update stop losses
                    if symbol != 'USDT':
                        current_price = data['price']
                        position_size = data['amount']
                        stop_price = current_price * (1 - self.stop_loss_percent / 100)
                        
                        # Check if stop loss already exists
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
                                'stop_price': stop_price,
                                'current_price': current_price,
                                'protection_percent': self.stop_loss_percent
                            })
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Risk control check failed: {e}")
        
        return {
            'violations': violations,
            'stop_losses': stop_losses_created,
            'recommendations': recommendations
        }
    
    def generate_optimized_signal(self, symbol):
        """Generate optimized trading signal with comprehensive analysis"""
        if not self.exchange:
            return None
        
        try:
            # Get market data
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            ticker = self.exchange.fetch_ticker(symbol)
            
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd_data['MACD_12_26_9'] if 'MACD_12_26_9' in macd_data.columns else 0
            df['macd_signal'] = macd_data['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd_data.columns else 0
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb_data['BBU_20_2.0'] if 'BBU_20_2.0' in bb_data.columns else df['close']
            df['bb_lower'] = bb_data['BBL_20_2.0'] if 'BBL_20_2.0' in bb_data.columns else df['close']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Get latest values
            latest = df.iloc[-1]
            current_price = ticker['last']
            
            # Calculate signal metrics
            rsi = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            macd = latest['macd'] if not pd.isna(latest['macd']) else 0
            macd_signal = latest['macd_signal'] if not pd.isna(latest['macd_signal']) else 0
            bb_position = latest['bb_position'] if not pd.isna(latest['bb_position']) else 0.5
            volume_ratio = latest['volume_ratio'] if not pd.isna(latest['volume_ratio']) else 1
            
            # Generate signal
            signal_score = 50  # Base score
            
            # RSI contribution
            if rsi < 30:
                signal_score += 20
                action_signal = 'BUY'
            elif rsi < 40:
                signal_score += 10
                action_signal = 'BUY'
            elif rsi > 70:
                signal_score += 20
                action_signal = 'SELL'
            elif rsi > 60:
                signal_score += 10
                action_signal = 'SELL'
            else:
                signal_score += 5
                action_signal = 'HOLD'
            
            # MACD contribution
            if macd > macd_signal:
                signal_score += 10
                macd_signal_result = 'BUY'
            else:
                signal_score += 5
                macd_signal_result = 'SELL'
            
            # Bollinger Bands contribution
            if bb_position < 0.2:
                signal_score += 15
                bb_signal = 'BUY'
            elif bb_position > 0.8:
                signal_score += 15
                bb_signal = 'SELL'
            else:
                signal_score += 5
                bb_signal = 'HOLD'
            
            # Volume confirmation
            if volume_ratio > 1.2:
                signal_score += 5
            
            # Determine final action
            signals = [action_signal, macd_signal_result, bb_signal]
            buy_votes = signals.count('BUY')
            sell_votes = signals.count('SELL')
            
            if buy_votes >= 2:
                final_action = 'BUY'
            elif sell_votes >= 2:
                final_action = 'SELL'
            else:
                final_action = 'HOLD'
            
            # Calculate confidence
            confidence = min(85, signal_score)
            
            # Assess risk level
            if confidence >= 75 and volume_ratio > 1.0:
                risk_level = 'LOW'
            elif confidence >= 65:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            # Calculate expected move
            volatility = df['close'].pct_change().std() * 100
            expected_move = volatility * 2  # 2-day expected move
            
            # Generate reasoning
            reasoning = f"RSI: {rsi:.1f}, MACD: {'Bullish' if macd > macd_signal else 'Bearish'}, "
            reasoning += f"BB Position: {bb_position:.2f}, Volume: {volume_ratio:.1f}x"
            
            signal = {
                'symbol': symbol,
                'action': final_action,
                'confidence': round(confidence, 1),
                'rsi': round(rsi, 2),
                'macd': round(macd, 4),
                'bb_position': round(bb_position, 3),
                'volume_ratio': round(volume_ratio, 2),
                'current_price': current_price,
                'reasoning': reasoning,
                'risk_level': risk_level,
                'expected_move': round(expected_move, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None
    
    def save_optimized_signal(self, signal):
        """Save optimized signal to database"""
        if not signal:
            return False
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO optimized_signals 
                    (symbol, action, confidence, rsi, macd, bb_position, volume_ratio,
                     current_price, reasoning, risk_level, expected_move, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal['symbol'], signal['action'], signal['confidence'],
                    signal.get('rsi'), signal.get('macd'), signal.get('bb_position'),
                    signal.get('volume_ratio'), signal['current_price'],
                    signal['reasoning'], signal['risk_level'],
                    signal['expected_move'], signal['timestamp']
                ))
                
                # Also save to unified_signals for compatibility
                cursor.execute('''
                    INSERT INTO unified_signals 
                    (symbol, action, confidence, rsi, macd, bollinger, volume_ratio,
                     price, reasoning, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal['symbol'], signal['action'], signal['confidence'],
                    signal.get('rsi'), signal.get('macd'), signal.get('bb_position'),
                    signal.get('volume_ratio'), signal['current_price'],
                    signal['reasoning'], signal['timestamp']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Signal save failed: {e}")
            return False
    
    def run_integrated_cycle(self):
        """Run complete integrated risk and signal management cycle"""
        logger.info("Running integrated risk & signal management cycle...")
        
        if not self.exchange:
            logger.error("OKX connection required")
            return None
        
        start_time = datetime.now()
        
        # 1. Check and enforce risk controls
        logger.info("Checking risk controls...")
        risk_results = self.check_and_enforce_risk_controls()
        
        # 2. Generate optimized signals
        logger.info("Generating optimized signals...")
        signals_generated = []
        
        for symbol in self.trading_pairs:
            try:
                signal = self.generate_optimized_signal(symbol)
                
                if signal and signal['confidence'] >= self.min_confidence_threshold:
                    if self.save_optimized_signal(signal):
                        signals_generated.append(signal)
                        logger.info(f"Generated {signal['action']} signal for {symbol}: {signal['confidence']:.1f}%")
                
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
                continue
        
        # 3. Compile results
        total_signals = len(signals_generated)
        high_confidence = len([s for s in signals_generated if s['confidence'] >= 75])
        buy_signals = len([s for s in signals_generated if s['action'] == 'BUY'])
        sell_signals = len([s for s in signals_generated if s['action'] == 'SELL'])
        
        avg_confidence = sum(s['confidence'] for s in signals_generated) / total_signals if total_signals > 0 else 0
        
        execution_results = {
            'timestamp': start_time.isoformat(),
            'execution_time': (datetime.now() - start_time).total_seconds(),
            
            # Risk management results
            'risk_violations': len(risk_results['violations']),
            'stop_losses_created': len(risk_results['stop_losses']),
            'risk_recommendations': len(risk_results['recommendations']),
            
            # Signal generation results
            'total_signals': total_signals,
            'high_confidence_signals': high_confidence,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'average_confidence': round(avg_confidence, 1),
            
            # Detailed results
            'violations': risk_results['violations'][:5],
            'new_stop_losses': risk_results['stop_losses'],
            'recommendations': risk_results['recommendations'][:5],
            'top_signals': signals_generated[:10]
        }
        
        # Save execution report
        with open('integrated_execution_report.json', 'w') as f:
            json.dump(execution_results, f, indent=2, default=str)
        
        # Print summary
        self._print_execution_summary(execution_results)
        
        return execution_results
    
    def _print_execution_summary(self, results):
        """Print execution summary"""
        print("\n" + "="*70)
        print("INTEGRATED RISK & SIGNAL MANAGEMENT SUMMARY")
        print("="*70)
        print(f"Execution Time: {results['execution_time']:.1f} seconds")
        
        print(f"\nRISK MANAGEMENT:")
        print(f"  Position Violations: {results['risk_violations']}")
        print(f"  Stop Losses Created: {results['stop_losses_created']}")
        print(f"  Recommendations: {results['risk_recommendations']}")
        
        print(f"\nSIGNAL GENERATION:")
        print(f"  Total Signals: {results['total_signals']}")
        print(f"  High Confidence (≥75%): {results['high_confidence_signals']}")
        print(f"  Average Confidence: {results['average_confidence']:.1f}%")
        print(f"  Distribution: {results['buy_signals']} BUY, {results['sell_signals']} SELL")
        
        if results['violations']:
            print(f"\nRISK VIOLATIONS:")
            for v in results['violations']:
                print(f"  {v['symbol']}: {v['current_percent']:.1f}% (limit: {v['max_percent']:.1f}%)")
        
        if results['new_stop_losses']:
            print(f"\nNEW STOP LOSSES:")
            for sl in results['new_stop_losses']:
                print(f"  {sl['symbol']}: Stop at ${sl['stop_price']:.4f} ({sl['protection_percent']}% protection)")
        
        if results['top_signals']:
            print(f"\nTOP SIGNALS:")
            for signal in results['top_signals'][:5]:
                print(f"  {signal['symbol']}: {signal['action']} at {signal['confidence']:.1f}% ({signal['risk_level']} risk)")
        
        print(f"\nExecution report saved: integrated_execution_report.json")

def main():
    """Main integrated management function"""
    manager = IntegratedTradingManager()
    
    if not manager.exchange:
        print("OKX connection required for integrated management")
        return
    
    # Run integrated cycle
    results = manager.run_integrated_cycle()
    
    print("Integrated risk & signal management completed!")
    return results

if __name__ == "__main__":
    main()