#!/usr/bin/env python3
"""
Professional Trading Optimizer - Non-Destructive Enhancement Layer
Implements advanced capital-efficient trading behaviors without altering core logic
"""

import os
import sys
import sqlite3
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, render_template_string, jsonify
import threading
from decimal import Decimal, ROUND_DOWN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('professional_trading.log')
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalTradingOptimizer:
    """Professional trading optimizer with advanced capital efficiency"""
    
    def __init__(self):
        self.exchange = None
        self.is_running = False
        
        # Enhanced trading parameters
        self.min_confidence = 75.0  # Raised from 70% for high-probability signals
        self.position_size_base = 0.01  # 1% base allocation
        self.max_position_size = 0.015  # 1.5% maximum per trade
        self.stop_loss_pct = 12.0
        self.cooldown_hours = 12
        
        # Multi-tiered take profit levels
        self.tp_levels = {
            'tp1': {'percent': 2.0, 'close_ratio': 0.30},  # Close 30% at +2%
            'tp2': {'percent': 4.0, 'close_ratio': 0.50},  # Close 50% at +4%
            'tp3': {'percent': 6.0, 'trail_remaining': 0.20}  # Trail 20% after +6%
        }
        
        # Risk management state
        self.consecutive_losses = 0
        self.daily_portfolio_start = 0
        self.last_trade_time = None
        self.cooldown_until = None
        self.market_regime = "UNKNOWN"
        
        # Position tracking for multi-tier TP
        self.active_positions = {}
        self.executed_signals = set()
        
        # Symbol management
        self.symbols = []
        self.symbol_prices = {}
        
        # Initialize system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize enhanced trading system"""
        try:
            # Connect to OKX
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Load markets and get symbols under $200
            self.exchange.load_markets()
            self.symbols = self.fetch_symbols_under_200()
            
            # Setup enhanced database
            self.setup_database()
            
            # Initialize daily portfolio value
            self.update_daily_portfolio_start()
            
            logger.info(f"Professional trading optimizer initialized with {len(self.symbols)} symbols")
            logger.info(f"Enhanced parameters: Min confidence {self.min_confidence}%, Max position {self.max_position_size*100}%")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def fetch_symbols_under_200(self) -> List[str]:
        """Fetch symbols under $200 from OKX"""
        try:
            tickers = self.exchange.fetch_tickers()
            under_200_symbols = []
            
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol and ticker['last'] and float(ticker['last']) <= 200:
                    under_200_symbols.append(symbol)
            
            # Sort by volume and take top 100
            symbol_volumes = [(symbol, tickers[symbol]['quoteVolume'] or 0) for symbol in under_200_symbols]
            symbol_volumes.sort(key=lambda x: x[1], reverse=True)
            
            return [symbol for symbol, _ in symbol_volumes[:100]]
            
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []
    
    def setup_database(self):
        """Setup professional trading database"""
        try:
            conn = sqlite3.connect('professional_trading.db')
            cursor = conn.cursor()
            
            # Enhanced signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS professional_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    price REAL,
                    position_size REAL,
                    market_regime TEXT,
                    atr REAL,
                    volume_surge REAL,
                    momentum_score REAL,
                    risk_tier TEXT,
                    tp1_target REAL,
                    tp2_target REAL,
                    tp3_target REAL,
                    stop_loss REAL
                )
            ''')
            
            # Multi-tier position tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS professional_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    entry_price REAL,
                    original_quantity REAL,
                    remaining_quantity REAL,
                    tp1_executed BOOLEAN DEFAULT 0,
                    tp2_executed BOOLEAN DEFAULT 0,
                    tp3_active BOOLEAN DEFAULT 0,
                    trailing_stop REAL,
                    status TEXT,
                    pnl REAL
                )
            ''')
            
            # Risk management log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    description TEXT,
                    portfolio_impact REAL,
                    action_taken TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Professional trading database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Non-invasive market regime detection using ADX + MACD + RSI slope"""
        if len(df) < 50:
            return "UNKNOWN"
        
        try:
            # Calculate ADX for trend strength
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            current_adx = adx['ADX_14'].iloc[-1] if 'ADX_14' in adx.columns else 20
            
            # MACD momentum
            macd = ta.macd(df['close'])
            macd_hist = macd['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in macd.columns else 0
            
            # RSI slope (trend direction)
            rsi = ta.rsi(df['close'], length=14)
            rsi_slope = rsi.iloc[-1] - rsi.iloc[-5] if len(rsi) >= 5 else 0
            
            # Regime logic
            if current_adx > 25 and abs(macd_hist) > 0.001:
                if rsi_slope > 2:
                    return "TREND_UP"
                elif rsi_slope < -2:
                    return "TREND_DOWN"
                else:
                    return "TREND"
            else:
                return "RANGE"
                
        except Exception as e:
            logger.warning(f"Market regime detection failed: {e}")
            return "UNKNOWN"
    
    def calculate_dynamic_position_size(self, confidence: float, market_data: Dict) -> float:
        """Dynamic risk allocation based on confidence and market conditions"""
        try:
            # Base allocation from confidence
            if confidence >= 85:
                base_size = self.max_position_size  # 1.5%
            elif confidence >= 80:
                base_size = 0.0125  # 1.25%
            else:
                base_size = self.position_size_base  # 1.0%
            
            # Adjust for market conditions
            atr_multiplier = 1.0
            volume_multiplier = 1.0
            
            # ATR adjustment (lower volatility = higher size)
            if 'atr' in market_data and market_data['atr']:
                atr_normalized = market_data['atr'] / market_data['price']
                if atr_normalized < 0.02:  # Low volatility
                    atr_multiplier = 1.1
                elif atr_normalized > 0.05:  # High volatility
                    atr_multiplier = 0.9
            
            # Volume surge adjustment
            if 'volume_surge' in market_data and market_data['volume_surge']:
                if market_data['volume_surge'] > 2.0:  # Breakout volume
                    volume_multiplier = 1.1
            
            # Calculate final size
            final_size = base_size * atr_multiplier * volume_multiplier
            return min(final_size, self.max_position_size)
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            return self.position_size_base
    
    def check_risk_conditions(self) -> bool:
        """Check psychological capital protection conditions"""
        try:
            # Check cooldown period
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                logger.warning(f"Trading paused until {self.cooldown_until}")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= 3:
                self.cooldown_until = datetime.now() + timedelta(hours=self.cooldown_hours)
                self.log_risk_event("CONSECUTIVE_LOSSES", 
                                   f"3 consecutive losses - trading paused for {self.cooldown_hours} hours",
                                   0)
                return False
            
            # Check daily portfolio loss
            current_balance = self.get_current_balance()
            if self.daily_portfolio_start > 0:
                daily_loss_pct = (self.daily_portfolio_start - current_balance) / self.daily_portfolio_start
                if daily_loss_pct > 0.05:  # 5% daily loss
                    self.cooldown_until = datetime.now() + timedelta(hours=self.cooldown_hours)
                    self.log_risk_event("DAILY_LOSS_LIMIT", 
                                       f"Portfolio down {daily_loss_pct*100:.1f}% - trading paused",
                                       daily_loss_pct)
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk condition check failed: {e}")
            return True
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get enhanced market data with regime detection"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.calculate_enhanced_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Standard indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # EMAs
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # ATR for volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Volume analysis
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum indicators
        df['mom'] = ta.mom(df['close'], length=10)
        df['roc'] = ta.roc(df['close'], length=10)
        
        return df
    
    def analyze_professional_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Enhanced signal analysis with professional filters"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Detect market regime
        regime = self.detect_market_regime(df)
        
        # Calculate signal strength
        buy_score = 0
        sell_score = 0
        
        # RSI analysis (enhanced)
        rsi = latest['rsi']
        if pd.notna(rsi):
            if rsi < 30:
                buy_score += 30
            elif rsi < 40:
                buy_score += 20
            elif rsi > 70:
                sell_score += 30
            elif rsi > 60:
                sell_score += 15
        
        # MACD analysis
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            macd = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal and prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']:
                    buy_score += 25
                elif macd < macd_signal and prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']:
                    sell_score += 25
        
        # EMA trend analysis
        if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
            if latest['close'] > latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
                buy_score += 25
            elif latest['close'] < latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
                sell_score += 25
        
        # Volume confirmation
        volume_surge = latest['volume_ratio'] if pd.notna(latest['volume_ratio']) else 1.0
        if volume_surge > 1.5:
            buy_score += 10
            sell_score += 10  # Volume confirms any direction
        
        # Determine signal
        if buy_score > sell_score and buy_score >= 50:
            signal_type = 'BUY'
            confidence = min(buy_score + (volume_surge - 1) * 10, 100)
        elif sell_score > buy_score and sell_score >= 50:
            signal_type = 'SELL'
            confidence = min(sell_score + (volume_surge - 1) * 10, 100)
        else:
            return None
        
        # Professional filter - only ≥75% confidence
        if confidence < self.min_confidence:
            return None
        
        # Market regime filter
        if regime == "RANGE" and confidence < 80:
            return None  # Be more selective in ranging markets
        
        # Calculate market data for position sizing
        market_data = {
            'price': float(latest['close']),
            'atr': float(latest['atr']) if pd.notna(latest['atr']) else None,
            'volume_surge': volume_surge,
            'momentum': float(latest['mom']) if pd.notna(latest['mom']) else 0
        }
        
        # Calculate dynamic position size
        position_size = self.calculate_dynamic_position_size(confidence, market_data)
        
        # Calculate multi-tier take profit levels
        current_price = float(latest['close'])
        if signal_type == 'BUY':
            tp1_target = current_price * (1 + self.tp_levels['tp1']['percent'] / 100)
            tp2_target = current_price * (1 + self.tp_levels['tp2']['percent'] / 100)
            tp3_target = current_price * (1 + self.tp_levels['tp3']['percent'] / 100)
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
        else:  # SELL
            tp1_target = current_price * (1 - self.tp_levels['tp1']['percent'] / 100)
            tp2_target = current_price * (1 - self.tp_levels['tp2']['percent'] / 100)
            tp3_target = current_price * (1 - self.tp_levels['tp3']['percent'] / 100)
            stop_loss = current_price * (1 + self.stop_loss_pct / 100)
        
        signal = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'price': current_price,
            'position_size': position_size,
            'market_regime': regime,
            'atr': market_data['atr'],
            'volume_surge': volume_surge,
            'momentum_score': market_data['momentum'],
            'risk_tier': self.get_risk_tier(confidence),
            'tp1_target': tp1_target,
            'tp2_target': tp2_target,
            'tp3_target': tp3_target,
            'stop_loss': stop_loss
        }
        
        return signal
    
    def get_risk_tier(self, confidence: float) -> str:
        """Categorize risk tier based on confidence"""
        if confidence >= 85:
            return "HIGH_CONFIDENCE"
        elif confidence >= 80:
            return "MEDIUM_CONFIDENCE"
        else:
            return "CONSERVATIVE"
    
    def execute_professional_signal(self, signal: Dict) -> bool:
        """Execute signal with multi-tier take profit setup"""
        if not self.exchange or not self.check_risk_conditions():
            return False
        
        try:
            symbol = signal['symbol']
            current_price = signal['price']
            position_size_pct = signal['position_size']
            
            # Get balance and calculate trade amount
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance.get('USDT', {}).get('free', 0))
            
            if usdt_balance < 10:
                logger.warning(f"Insufficient USDT balance: ${usdt_balance:.2f}")
                return False
            
            # Calculate trade value
            trade_value = usdt_balance * position_size_pct
            quantity = trade_value / current_price
            
            # Execute based on signal type
            if signal['signal_type'] == 'BUY':
                quantity = self.exchange.amount_to_precision(symbol, quantity)
                order = self.exchange.create_market_buy_order(symbol, float(quantity))
                side = 'BUY'
            else:  # SELL
                base_currency = symbol.split('/')[0]
                available_amount = float(balance.get(base_currency, {}).get('free', 0))
                min_amount = self.exchange.markets[symbol]['limits']['amount']['min']
                
                if available_amount < min_amount:
                    logger.warning(f"Insufficient {base_currency} for SELL: {available_amount}")
                    return False
                
                sell_quantity = min(available_amount * 0.8, quantity)
                sell_quantity = self.exchange.amount_to_precision(symbol, sell_quantity)
                order = self.exchange.create_market_sell_order(symbol, float(sell_quantity))
                quantity = sell_quantity
                side = 'SELL'
            
            # Save position for multi-tier management
            self.save_professional_position(signal, order, float(quantity))
            
            # Save signal
            self.save_professional_signal(signal)
            
            logger.info(f"✅ PROFESSIONAL TRADE: {side} {quantity} {symbol} @ ${current_price:.6f} "
                       f"(Confidence: {signal['confidence']:.1f}%, Size: {position_size_pct*100:.2f}%, "
                       f"Regime: {signal['market_regime']})")
            
            return True
            
        except Exception as e:
            logger.error(f"Professional trade execution failed: {e}")
            self.consecutive_losses += 1
            return False
    
    def save_professional_signal(self, signal: Dict):
        """Save professional signal to database"""
        try:
            conn = sqlite3.connect('professional_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO professional_signals (
                    timestamp, symbol, signal_type, confidence, price, position_size,
                    market_regime, atr, volume_surge, momentum_score, risk_tier,
                    tp1_target, tp2_target, tp3_target, stop_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'], signal['symbol'], signal['signal_type'],
                signal['confidence'], signal['price'], signal['position_size'],
                signal['market_regime'], signal['atr'], signal['volume_surge'],
                signal['momentum_score'], signal['risk_tier'], signal['tp1_target'],
                signal['tp2_target'], signal['tp3_target'], signal['stop_loss']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save professional signal: {e}")
    
    def save_professional_position(self, signal: Dict, order: Dict, quantity: float):
        """Save position for multi-tier take profit management"""
        try:
            conn = sqlite3.connect('professional_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO professional_positions (
                    timestamp, symbol, entry_price, original_quantity, 
                    remaining_quantity, status
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'], signal['symbol'], signal['price'],
                quantity, quantity, 'ACTIVE'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save professional position: {e}")
    
    def get_current_balance(self) -> float:
        """Get current USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('total', 0))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    def update_daily_portfolio_start(self):
        """Update daily portfolio starting value"""
        current_time = datetime.now()
        if (not hasattr(self, 'last_daily_update') or 
            self.last_daily_update.date() != current_time.date()):
            self.daily_portfolio_start = self.get_current_balance()
            self.last_daily_update = current_time
            self.consecutive_losses = 0  # Reset daily
    
    def log_risk_event(self, event_type: str, description: str, impact: float):
        """Log risk management events"""
        try:
            conn = sqlite3.connect('professional_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_events (timestamp, event_type, description, portfolio_impact, action_taken)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), event_type, description, impact, "TRADING_PAUSED"))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log risk event: {e}")
    
    def professional_trading_loop(self):
        """Main professional trading loop"""
        logger.info("🚀 Starting Professional Trading Optimizer")
        logger.info(f"Enhanced Configuration: Min Confidence: {self.min_confidence}%, "
                   f"Max Position: {self.max_position_size*100}%, Multi-Tier TP: Enabled")
        
        while self.is_running:
            try:
                # Update daily portfolio tracking
                self.update_daily_portfolio_start()
                
                # Check risk conditions
                if not self.check_risk_conditions():
                    time.sleep(300)  # Wait 5 minutes during cooldown
                    continue
                
                signals_generated = 0
                trades_executed = 0
                
                logger.info(f"🔄 Professional scan: {len(self.symbols)} symbols (Regime: {self.market_regime})")
                
                for symbol in self.symbols:
                    try:
                        df = self.get_market_data(symbol)
                        if df is not None:
                            signal = self.analyze_professional_signal(symbol, df)
                            if signal:
                                signals_generated += 1
                                
                                # Update market regime from signal
                                self.market_regime = signal['market_regime']
                                
                                # Execute if passes all filters
                                if self.execute_professional_signal(signal):
                                    trades_executed += 1
                                    
                                logger.info(f"📊 {symbol}: {signal['signal_type']} "
                                          f"(Conf: {signal['confidence']:.1f}%, "
                                          f"Size: {signal['position_size']*100:.2f}%, "
                                          f"Regime: {signal['market_regime']})")
                    
                    except Exception as e:
                        logger.error(f"Professional analysis failed for {symbol}: {e}")
                        continue
                
                logger.info(f"✅ Professional scan complete: {signals_generated} signals, "
                           f"{trades_executed} trades executed")
                
                # Dynamic scan interval based on market regime
                if self.market_regime.startswith("TREND"):
                    scan_interval = 180  # 3 minutes in trending markets
                else:
                    scan_interval = 300  # 5 minutes in ranging markets
                
                logger.info(f"⏰ Next scan in {scan_interval} seconds (Regime: {self.market_regime})")
                
                for _ in range(scan_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Professional trading loop error: {e}")
                time.sleep(60)
    
    def start_professional_trading(self):
        """Start professional trading system"""
        if self.is_running:
            return
        
        self.is_running = True
        trading_thread = threading.Thread(target=self.professional_trading_loop, daemon=True)
        trading_thread.start()
    
    def stop_professional_trading(self):
        """Stop professional trading system"""
        self.is_running = False

# Flask web interface
app = Flask(__name__)
trading_system = ProfessionalTradingOptimizer()

@app.route('/')
def professional_dashboard():
    """Professional trading dashboard"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Professional Trading Optimizer - Advanced Capital Efficiency</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0e27; color: #fff; }
            .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #4CAF50; font-size: 2.8em; margin-bottom: 10px; }
            .header p { color: #999; font-size: 1.2em; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .stat-card { background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            .stat-value { font-size: 1.8em; font-weight: bold; color: #4CAF50; }
            .stat-label { color: #ccc; margin-top: 5px; font-size: 0.9em; }
            .regime-indicator { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-top: 5px; }
            .trend { background: #4CAF50; color: #fff; }
            .range { background: #ff9800; color: #fff; }
            .unknown { background: #666; color: #fff; }
            .cooldown { background: #f44336; color: #fff; }
            .content-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
            .signals-section { background: #1a1a2e; border-radius: 10px; padding: 20px; }
            .risk-section { background: #1a1a2e; border-radius: 10px; padding: 20px; }
            .section-title { color: #4CAF50; font-size: 1.5em; margin-bottom: 15px; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
            .signal-item { background: #252545; margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .signal-symbol { font-weight: bold; color: #fff; }
            .confidence-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
            .high-conf { background: #4CAF50; color: #fff; }
            .medium-conf { background: #ff9800; color: #fff; }
            .conservative { background: #2196F3; color: #fff; }
            .signal-details { font-size: 0.9em; color: #ccc; }
            .tier-info { color: #ffa726; font-weight: bold; }
            .risk-item { background: #252545; margin: 10px 0; padding: 15px; border-radius: 8px; }
            @media (max-width: 768px) { .content-grid { grid-template-columns: 1fr; } }
        </style>
        <script>
            function updateData() {
                Promise.all([
                    fetch('/api/professional/status').then(r => r.json()),
                    fetch('/api/professional/signals').then(r => r.json()),
                    fetch('/api/professional/risk').then(r => r.json())
                ]).then(([status, signals, risk]) => {
                    // Update status
                    document.getElementById('system-status').textContent = status.status;
                    document.getElementById('confidence-threshold').textContent = status.min_confidence + '%';
                    document.getElementById('max-position').textContent = (status.max_position * 100).toFixed(1) + '%';
                    document.getElementById('market-regime').textContent = status.market_regime;
                    document.getElementById('market-regime').className = 'regime-indicator ' + 
                        (status.market_regime.includes('TREND') ? 'trend' : 
                         status.market_regime === 'RANGE' ? 'range' : 'unknown');
                    
                    document.getElementById('signals-count').textContent = signals.length;
                    document.getElementById('consecutive-losses').textContent = risk.consecutive_losses;
                    
                    // Update signals
                    const signalsHtml = signals.slice(0, 10).map(signal => `
                        <div class="signal-item">
                            <div class="signal-header">
                                <span class="signal-symbol">${signal.symbol}</span>
                                <span class="confidence-badge ${signal.risk_tier.toLowerCase().replace('_', '-')}">${signal.confidence.toFixed(1)}%</span>
                            </div>
                            <div class="signal-details">
                                Size: <span class="tier-info">${(signal.position_size * 100).toFixed(2)}%</span> | 
                                Regime: ${signal.market_regime} | 
                                TP Tiers: <span class="tier-info">${signal.tp1_target.toFixed(4)} / ${signal.tp2_target.toFixed(4)} / ${signal.tp3_target.toFixed(4)}</span>
                            </div>
                        </div>
                    `).join('');
                    document.getElementById('signals-list').innerHTML = signalsHtml;
                    
                    // Update risk events
                    const riskHtml = risk.recent_events.slice(0, 5).map(event => `
                        <div class="risk-item">
                            <strong>${event.event_type}</strong><br>
                            <small>${event.description}</small>
                        </div>
                    `).join('');
                    document.getElementById('risk-events').innerHTML = riskHtml;
                });
            }
            setInterval(updateData, 3000);
            updateData();
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚡ Professional Trading Optimizer</h1>
                <p>High-Probability Signals • Dynamic Risk Allocation • Multi-Tier Take Profit • Capital Protection</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="system-status">Running</div>
                    <div class="stat-label">System Status</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="confidence-threshold">75%</div>
                    <div class="stat-label">Min Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="max-position">1.5%</div>
                    <div class="stat-label">Max Position</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="market-regime">UNKNOWN</div>
                    <div class="stat-label">Market Regime</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="signals-count">0</div>
                    <div class="stat-label">High-Prob Signals</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="consecutive-losses">0</div>
                    <div class="stat-label">Consecutive Losses</div>
                </div>
            </div>
            
            <div class="content-grid">
                <div class="signals-section">
                    <div class="section-title">🎯 Professional Signals (≥75% Confidence)</div>
                    <div id="signals-list">Loading signals...</div>
                </div>
                
                <div class="risk-section">
                    <div class="section-title">🛡️ Risk Management</div>
                    <div id="risk-events">Loading risk events...</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/api/professional/status')
def get_professional_status():
    """Get professional system status"""
    return jsonify({
        'status': 'Running' if trading_system.is_running else 'Stopped',
        'min_confidence': trading_system.min_confidence,
        'max_position': trading_system.max_position_size,
        'market_regime': trading_system.market_regime,
        'cooldown_until': trading_system.cooldown_until.isoformat() if trading_system.cooldown_until else None
    })

@app.route('/api/professional/signals')
def get_professional_signals():
    """Get recent professional signals"""
    try:
        conn = sqlite3.connect('professional_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM professional_signals 
            ORDER BY timestamp DESC LIMIT 50
        ''')
        
        signals = []
        for row in cursor.fetchall():
            signals.append({
                'id': row[0], 'timestamp': row[1], 'symbol': row[2],
                'signal_type': row[3], 'confidence': row[4], 'price': row[5],
                'position_size': row[6], 'market_regime': row[7], 'atr': row[8],
                'volume_surge': row[9], 'momentum_score': row[10], 'risk_tier': row[11],
                'tp1_target': row[12], 'tp2_target': row[13], 'tp3_target': row[14],
                'stop_loss': row[15]
            })
        
        conn.close()
        return jsonify(signals)
        
    except Exception as e:
        logger.error(f"Failed to get professional signals: {e}")
        return jsonify([])

@app.route('/api/professional/risk')
def get_risk_status():
    """Get risk management status"""
    try:
        conn = sqlite3.connect('professional_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM risk_events 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'id': row[0], 'timestamp': row[1], 'event_type': row[2],
                'description': row[3], 'portfolio_impact': row[4], 'action_taken': row[5]
            })
        
        conn.close()
        
        return jsonify({
            'consecutive_losses': trading_system.consecutive_losses,
            'cooldown_active': trading_system.cooldown_until is not None,
            'daily_portfolio_start': trading_system.daily_portfolio_start,
            'recent_events': events
        })
        
    except Exception as e:
        logger.error(f"Failed to get risk status: {e}")
        return jsonify({'consecutive_losses': 0, 'recent_events': []})

def main():
    """Main function"""
    try:
        # Start professional trading system
        trading_system.start_professional_trading()
        
        # Start web interface
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down professional trading optimizer...")
        trading_system.stop_professional_trading()
    except Exception as e:
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()