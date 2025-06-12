#!/usr/bin/env python3
"""
Enhanced Signal Generation Optimizer
Optimizes AI signal generation with improved algorithms and lower thresholds
"""

import sqlite3
import json
import os
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSignalOptimizer:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        self.setup_signal_database()
        
        # Optimized parameters based on audit
        self.min_confidence_threshold = 60.0  # Lowered from 75%
        self.target_signals_per_day = 15       # Increased from 5
        self.lookback_periods = {
            'short': 14,   # RSI period
            'medium': 26,  # MACD fast
            'long': 50     # EMA period
        }
        
        # Extended trading pairs for more opportunities
        self.trading_pairs = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
            'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT', 'ATOM/USDT', 'NEAR/USDT'
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
                logger.info("Enhanced signal optimizer connected to OKX")
            else:
                logger.warning("OKX credentials not configured")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_signal_database(self):
        """Setup enhanced signal database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Enhanced signals table with more metrics
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS enhanced_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        base_confidence REAL NOT NULL,
                        
                        -- Technical indicators
                        rsi REAL,
                        macd REAL,
                        macd_signal REAL,
                        macd_histogram REAL,
                        ema_20 REAL,
                        ema_50 REAL,
                        bollinger_upper REAL,
                        bollinger_lower REAL,
                        bollinger_position REAL,
                        
                        -- Volume metrics
                        volume_ratio REAL,
                        volume_sma REAL,
                        volume_trend TEXT,
                        
                        -- Price metrics
                        current_price REAL,
                        price_change_1h REAL,
                        price_change_24h REAL,
                        support_level REAL,
                        resistance_level REAL,
                        
                        -- Signal quality metrics
                        signal_strength TEXT,
                        market_condition TEXT,
                        trend_direction TEXT,
                        volatility_level TEXT,
                        
                        -- Execution data
                        reasoning TEXT,
                        risk_level TEXT,
                        expected_move REAL,
                        time_horizon TEXT,
                        
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Signal performance tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signal_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id INTEGER,
                        symbol TEXT NOT NULL,
                        initial_confidence REAL,
                        actual_performance REAL,
                        time_to_target REAL,
                        success BOOLEAN,
                        performance_category TEXT,
                        lessons_learned TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (signal_id) REFERENCES enhanced_signals (id)
                    )
                ''')
                
                conn.commit()
                logger.info("Enhanced signal database initialized")
                
        except Exception as e:
            logger.error(f"Signal database setup failed: {e}")
    
    def get_market_data(self, symbol, timeframe='1h', limit=100):
        """Get comprehensive market data for analysis"""
        if not self.exchange:
            return None
        
        try:
            # Get OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Get current ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            return df, ticker
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        try:
            # Price-based indicators
            df['rsi'] = ta.rsi(df['close'], length=self.lookback_periods['short'])
            
            # MACD
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd_data['MACD_12_26_9']
            df['macd_signal'] = macd_data['MACDs_12_26_9']
            df['macd_histogram'] = macd_data['MACDh_12_26_9']
            
            # Moving averages
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['sma_20'] = ta.sma(df['close'], length=20)
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb_data['BBU_20_2.0']
            df['bb_lower'] = bb_data['BBL_20_2.0']
            df['bb_middle'] = bb_data['BBM_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # Volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Support and resistance levels
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            
            return df
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            return df
    
    def analyze_market_condition(self, df, ticker):
        """Analyze current market conditions"""
        try:
            latest = df.iloc[-1]
            
            # Trend analysis
            if latest['ema_20'] > latest['ema_50']:
                trend = 'UPTREND'
            elif latest['ema_20'] < latest['ema_50']:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'
            
            # Volatility assessment
            volatility_percentile = (latest['volatility'] - df['volatility'].quantile(0.2)) / (df['volatility'].quantile(0.8) - df['volatility'].quantile(0.2))
            
            if volatility_percentile > 0.7:
                volatility = 'HIGH'
            elif volatility_percentile < 0.3:
                volatility = 'LOW'
            else:
                volatility = 'MEDIUM'
            
            # Market condition
            rsi = latest['rsi']
            bb_position = latest['bb_position']
            
            if rsi > 70 and bb_position > 0.8:
                condition = 'OVERBOUGHT'
            elif rsi < 30 and bb_position < 0.2:
                condition = 'OVERSOLD'
            elif 40 <= rsi <= 60 and 0.3 <= bb_position <= 0.7:
                condition = 'NEUTRAL'
            else:
                condition = 'TRANSITIONAL'
            
            # Volume trend
            recent_volume_avg = df['volume'].tail(5).mean()
            older_volume_avg = df['volume'].tail(20).head(15).mean()
            
            if recent_volume_avg > older_volume_avg * 1.2:
                volume_trend = 'INCREASING'
            elif recent_volume_avg < older_volume_avg * 0.8:
                volume_trend = 'DECREASING'
            else:
                volume_trend = 'STABLE'
            
            return {
                'trend': trend,
                'volatility': volatility,
                'condition': condition,
                'volume_trend': volume_trend,
                'price_change_1h': ticker.get('percentage', 0) if ticker.get('percentage') else 0,
                'price_change_24h': ticker.get('change', 0) if ticker.get('change') else 0
            }
            
        except Exception as e:
            logger.error(f"Market condition analysis failed: {e}")
            return {
                'trend': 'UNKNOWN',
                'volatility': 'MEDIUM',
                'condition': 'NEUTRAL',
                'volume_trend': 'STABLE',
                'price_change_1h': 0,
                'price_change_24h': 0
            }
    
    def generate_enhanced_signal(self, symbol):
        """Generate enhanced trading signal with comprehensive analysis"""
        try:
            # Get market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                return None
            
            df, ticker = market_data
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Analyze market conditions
            market_analysis = self.analyze_market_condition(df, ticker)
            
            # Get latest values
            latest = df.iloc[-1]
            current_price = ticker['last']
            
            # Initialize signal
            signal = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate base confidence and action
            base_confidence, action, reasoning = self._calculate_signal_metrics(latest, market_analysis)
            
            # Apply confidence modifiers based on market conditions
            adjusted_confidence = self._apply_confidence_modifiers(
                base_confidence, market_analysis, latest
            )
            
            # Determine signal strength
            if adjusted_confidence >= 80:
                signal_strength = 'STRONG'
            elif adjusted_confidence >= 70:
                signal_strength = 'MODERATE'
            elif adjusted_confidence >= 60:
                signal_strength = 'WEAK'
            else:
                signal_strength = 'VERY_WEAK'
            
            # Calculate expected move and risk
            expected_move = self._calculate_expected_move(latest, market_analysis)
            risk_level = self._assess_risk_level(adjusted_confidence, market_analysis, latest)
            
            # Populate signal data
            signal.update({
                'action': action,
                'confidence': round(adjusted_confidence, 1),
                'base_confidence': round(base_confidence, 1),
                
                # Technical indicators
                'rsi': round(latest['rsi'], 2) if not pd.isna(latest['rsi']) else None,
                'macd': round(latest['macd'], 4) if not pd.isna(latest['macd']) else None,
                'macd_signal': round(latest['macd_signal'], 4) if not pd.isna(latest['macd_signal']) else None,
                'macd_histogram': round(latest['macd_histogram'], 4) if not pd.isna(latest['macd_histogram']) else None,
                'ema_20': round(latest['ema_20'], 4) if not pd.isna(latest['ema_20']) else None,
                'ema_50': round(latest['ema_50'], 4) if not pd.isna(latest['ema_50']) else None,
                'bollinger_upper': round(latest['bb_upper'], 4) if not pd.isna(latest['bb_upper']) else None,
                'bollinger_lower': round(latest['bb_lower'], 4) if not pd.isna(latest['bb_lower']) else None,
                'bollinger_position': round(latest['bb_position'], 3) if not pd.isna(latest['bb_position']) else None,
                
                # Volume metrics
                'volume_ratio': round(latest['volume_ratio'], 2) if not pd.isna(latest['volume_ratio']) else None,
                'volume_sma': round(latest['volume_sma'], 0) if not pd.isna(latest['volume_sma']) else None,
                'volume_trend': market_analysis['volume_trend'],
                
                # Price metrics
                'price_change_1h': round(market_analysis['price_change_1h'], 2),
                'price_change_24h': round(market_analysis['price_change_24h'], 2),
                'support_level': round(latest['support'], 4) if not pd.isna(latest['support']) else None,
                'resistance_level': round(latest['resistance'], 4) if not pd.isna(latest['resistance']) else None,
                
                # Analysis results
                'signal_strength': signal_strength,
                'market_condition': market_analysis['condition'],
                'trend_direction': market_analysis['trend'],
                'volatility_level': market_analysis['volatility'],
                
                # Execution data
                'reasoning': reasoning,
                'risk_level': risk_level,
                'expected_move': round(expected_move, 2),
                'time_horizon': self._determine_time_horizon(market_analysis, adjusted_confidence)
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Enhanced signal generation failed for {symbol}: {e}")
            return None
    
    def _calculate_signal_metrics(self, latest, market_analysis):
        """Calculate base signal metrics"""
        try:
            rsi = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            macd = latest['macd'] if not pd.isna(latest['macd']) else 0
            macd_signal = latest['macd_signal'] if not pd.isna(latest['macd_signal']) else 0
            bb_position = latest['bb_position'] if not pd.isna(latest['bb_position']) else 0.5
            
            # RSI signals
            rsi_score = 0
            if rsi < 30:
                rsi_score = 25  # Strong buy signal
                rsi_signal = 'BUY'
            elif rsi < 40:
                rsi_score = 15  # Moderate buy signal
                rsi_signal = 'BUY'
            elif rsi > 70:
                rsi_score = 25  # Strong sell signal
                rsi_signal = 'SELL'
            elif rsi > 60:
                rsi_score = 15  # Moderate sell signal
                rsi_signal = 'SELL'
            else:
                rsi_score = 5   # Neutral
                rsi_signal = 'HOLD'
            
            # MACD signals
            macd_score = 0
            if macd > macd_signal and macd > 0:
                macd_score = 20  # Strong bullish
                macd_signal_result = 'BUY'
            elif macd > macd_signal:
                macd_score = 10  # Moderate bullish
                macd_signal_result = 'BUY'
            elif macd < macd_signal and macd < 0:
                macd_score = 20  # Strong bearish
                macd_signal_result = 'SELL'
            elif macd < macd_signal:
                macd_score = 10  # Moderate bearish
                macd_signal_result = 'SELL'
            else:
                macd_score = 5   # Neutral
                macd_signal_result = 'HOLD'
            
            # Bollinger Band signals
            bb_score = 0
            if bb_position < 0.2:
                bb_score = 15  # Near lower band - buy signal
                bb_signal = 'BUY'
            elif bb_position > 0.8:
                bb_score = 15  # Near upper band - sell signal
                bb_signal = 'SELL'
            else:
                bb_score = 5   # In middle range
                bb_signal = 'HOLD'
            
            # Trend confirmation
            trend_score = 0
            if market_analysis['trend'] == 'UPTREND':
                trend_score = 15
                trend_signal = 'BUY'
            elif market_analysis['trend'] == 'DOWNTREND':
                trend_score = 15
                trend_signal = 'SELL'
            else:
                trend_score = 5
                trend_signal = 'HOLD'
            
            # Combine signals
            signals = [rsi_signal, macd_signal_result, bb_signal, trend_signal]
            buy_votes = signals.count('BUY')
            sell_votes = signals.count('SELL')
            
            if buy_votes >= 3:
                action = 'BUY'
                base_confidence = min(85, 50 + rsi_score + macd_score + bb_score + trend_score)
            elif sell_votes >= 3:
                action = 'SELL'
                base_confidence = min(85, 50 + rsi_score + macd_score + bb_score + trend_score)
            elif buy_votes > sell_votes:
                action = 'BUY'
                base_confidence = min(75, 40 + rsi_score + macd_score + bb_score + trend_score)
            elif sell_votes > buy_votes:
                action = 'SELL'
                base_confidence = min(75, 40 + rsi_score + macd_score + bb_score + trend_score)
            else:
                action = 'HOLD'
                base_confidence = min(65, 30 + rsi_score + macd_score + bb_score + trend_score)
            
            # Generate reasoning
            reasoning = f"Technical analysis: RSI {rsi:.1f} ({rsi_signal}), MACD {macd_signal_result}, "
            reasoning += f"Bollinger position {bb_position:.2f} ({bb_signal}), Trend: {market_analysis['trend']}"
            
            return base_confidence, action, reasoning
            
        except Exception as e:
            logger.error(f"Signal metrics calculation failed: {e}")
            return 50, 'HOLD', 'Technical analysis failed'
    
    def _apply_confidence_modifiers(self, base_confidence, market_analysis, latest):
        """Apply confidence modifiers based on market conditions"""
        try:
            adjusted_confidence = base_confidence
            
            # Volume confirmation
            volume_ratio = latest['volume_ratio'] if not pd.isna(latest['volume_ratio']) else 1
            if volume_ratio > 1.5:
                adjusted_confidence += 5  # High volume confirmation
            elif volume_ratio < 0.5:
                adjusted_confidence -= 5  # Low volume warning
            
            # Volatility adjustment
            if market_analysis['volatility'] == 'HIGH':
                adjusted_confidence -= 5  # Higher risk in volatile markets
            elif market_analysis['volatility'] == 'LOW':
                adjusted_confidence += 3  # More predictable in low volatility
            
            # Market condition adjustment
            if market_analysis['condition'] in ['OVERBOUGHT', 'OVERSOLD']:
                adjusted_confidence += 8  # Strong reversal signals
            elif market_analysis['condition'] == 'NEUTRAL':
                adjusted_confidence += 2  # Stable conditions
            
            # Price momentum
            price_change_24h = abs(market_analysis['price_change_24h'])
            if price_change_24h > 10:
                adjusted_confidence -= 3  # Large moves may reverse
            elif 2 <= price_change_24h <= 5:
                adjusted_confidence += 3  # Moderate momentum
            
            return max(0, min(100, adjusted_confidence))
            
        except Exception as e:
            logger.error(f"Confidence modifier application failed: {e}")
            return base_confidence
    
    def _calculate_expected_move(self, latest, market_analysis):
        """Calculate expected price move percentage"""
        try:
            # Base expectation on volatility and trend strength
            volatility_factor = {
                'LOW': 2.0,
                'MEDIUM': 4.0,
                'HIGH': 8.0
            }.get(market_analysis['volatility'], 4.0)
            
            # Adjust for trend strength
            if market_analysis['trend'] in ['UPTREND', 'DOWNTREND']:
                trend_factor = 1.2
            else:
                trend_factor = 0.8
            
            expected_move = volatility_factor * trend_factor
            return expected_move
            
        except Exception as e:
            logger.error(f"Expected move calculation failed: {e}")
            return 3.0
    
    def _assess_risk_level(self, confidence, market_analysis, latest):
        """Assess signal risk level"""
        try:
            risk_factors = 0
            
            # Low confidence increases risk
            if confidence < 65:
                risk_factors += 2
            elif confidence < 70:
                risk_factors += 1
            
            # High volatility increases risk
            if market_analysis['volatility'] == 'HIGH':
                risk_factors += 2
            elif market_analysis['volatility'] == 'MEDIUM':
                risk_factors += 1
            
            # Extreme RSI levels reduce risk (clear signals)
            rsi = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            if rsi < 25 or rsi > 75:
                risk_factors -= 1
            
            # Market condition affects risk
            if market_analysis['condition'] in ['OVERBOUGHT', 'OVERSOLD']:
                risk_factors -= 1  # Clear reversal signals
            elif market_analysis['condition'] == 'TRANSITIONAL':
                risk_factors += 1  # Uncertain conditions
            
            # Determine risk level
            if risk_factors <= 0:
                return 'LOW'
            elif risk_factors <= 2:
                return 'MEDIUM'
            else:
                return 'HIGH'
                
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return 'MEDIUM'
    
    def _determine_time_horizon(self, market_analysis, confidence):
        """Determine appropriate time horizon for signal"""
        if confidence >= 75 and market_analysis['volatility'] != 'HIGH':
            return 'MEDIUM_TERM'  # 1-7 days
        elif confidence >= 65:
            return 'SHORT_TERM'   # 1-24 hours
        else:
            return 'IMMEDIATE'    # 1-4 hours
    
    def save_enhanced_signal(self, signal):
        """Save enhanced signal to database"""
        if not signal:
            return False
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO enhanced_signals (
                        symbol, action, confidence, base_confidence,
                        rsi, macd, macd_signal, macd_histogram,
                        ema_20, ema_50, bollinger_upper, bollinger_lower, bollinger_position,
                        volume_ratio, volume_sma, volume_trend,
                        current_price, price_change_1h, price_change_24h,
                        support_level, resistance_level,
                        signal_strength, market_condition, trend_direction, volatility_level,
                        reasoning, risk_level, expected_move, time_horizon, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal['symbol'], signal['action'], signal['confidence'], signal['base_confidence'],
                    signal.get('rsi'), signal.get('macd'), signal.get('macd_signal'), signal.get('macd_histogram'),
                    signal.get('ema_20'), signal.get('ema_50'), signal.get('bollinger_upper'), 
                    signal.get('bollinger_lower'), signal.get('bollinger_position'),
                    signal.get('volume_ratio'), signal.get('volume_sma'), signal.get('volume_trend'),
                    signal['current_price'], signal.get('price_change_1h'), signal.get('price_change_24h'),
                    signal.get('support_level'), signal.get('resistance_level'),
                    signal['signal_strength'], signal['market_condition'], signal['trend_direction'], 
                    signal['volatility_level'], signal['reasoning'], signal['risk_level'], 
                    signal['expected_move'], signal['time_horizon'], signal['timestamp']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Signal save failed: {e}")
            return False
    
    def generate_optimized_signals_batch(self):
        """Generate optimized signals for all trading pairs"""
        if not self.exchange:
            logger.error("Exchange connection required")
            return []
        
        logger.info(f"Generating enhanced signals for {len(self.trading_pairs)} pairs...")
        
        signals_generated = []
        high_confidence_count = 0
        
        for symbol in self.trading_pairs:
            try:
                signal = self.generate_enhanced_signal(symbol)
                
                if signal and signal['confidence'] >= self.min_confidence_threshold:
                    # Save to enhanced signals table
                    if self.save_enhanced_signal(signal):
                        signals_generated.append(signal)
                        
                        if signal['confidence'] >= 75:
                            high_confidence_count += 1
                        
                        logger.info(f"Generated {signal['action']} signal for {symbol}: {signal['confidence']:.1f}%")
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
                continue
        
        # Update unified_signals table for compatibility
        self._update_unified_signals(signals_generated)
        
        logger.info(f"Signal generation complete: {len(signals_generated)} total, {high_confidence_count} high confidence")
        
        return signals_generated
    
    def _update_unified_signals(self, enhanced_signals):
        """Update unified_signals table for backward compatibility"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                for signal in enhanced_signals:
                    cursor.execute('''
                        INSERT INTO unified_signals 
                        (symbol, action, confidence, rsi, macd, bollinger, volume_ratio, 
                         price, reasoning, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        signal['symbol'], signal['action'], signal['confidence'],
                        signal.get('rsi'), signal.get('macd'), signal.get('bollinger_position'),
                        signal.get('volume_ratio'), signal['current_price'],
                        signal['reasoning'], signal['timestamp']
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Unified signals update failed: {e}")
    
    def run_signal_optimization_cycle(self):
        """Run complete signal optimization cycle"""
        logger.info("Running enhanced signal optimization cycle...")
        
        start_time = datetime.now()
        
        # Generate optimized signals
        signals = self.generate_optimized_signals_batch()
        
        # Calculate performance metrics
        total_signals = len(signals)
        high_confidence = len([s for s in signals if s['confidence'] >= 75])
        buy_signals = len([s for s in signals if s['action'] == 'BUY'])
        sell_signals = len([s for s in signals if s['action'] == 'SELL'])
        
        avg_confidence = sum(s['confidence'] for s in signals) / total_signals if total_signals > 0 else 0
        
        optimization_results = {
            'timestamp': start_time.isoformat(),
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'total_signals': total_signals,
            'high_confidence_signals': high_confidence,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'average_confidence': round(avg_confidence, 1),
            'min_confidence_threshold': self.min_confidence_threshold,
            'trading_pairs_analyzed': len(self.trading_pairs),
            'signals': signals[:10]  # Store first 10 for review
        }
        
        # Save optimization report
        with open('signal_optimization_report.json', 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        # Print summary
        self._print_optimization_summary(optimization_results)
        
        return optimization_results
    
    def _print_optimization_summary(self, results):
        """Print signal optimization summary"""
        print("\n" + "="*60)
        print("SIGNAL OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Execution Time: {results['execution_time']:.1f} seconds")
        print(f"Total Signals Generated: {results['total_signals']}")
        print(f"High Confidence (≥75%): {results['high_confidence_signals']}")
        print(f"Average Confidence: {results['average_confidence']:.1f}%")
        print(f"Signal Distribution: {results['buy_signals']} BUY, {results['sell_signals']} SELL")
        print(f"Trading Pairs Analyzed: {results['trading_pairs_analyzed']}")
        
        if results['total_signals'] > 0:
            print(f"\nTop Signals Generated:")
            for signal in results['signals'][:5]:
                print(f"  {signal['symbol']}: {signal['action']} at {signal['confidence']:.1f}% confidence")
                print(f"    Risk: {signal['risk_level']}, Expected move: {signal['expected_move']:.1f}%")
        
        print(f"\nOptimization report saved: signal_optimization_report.json")

def main():
    """Main signal optimization function"""
    optimizer = EnhancedSignalOptimizer()
    
    if not optimizer.exchange:
        print("OKX connection required for signal optimization")
        return
    
    # Run optimization cycle
    results = optimizer.run_signal_optimization_cycle()
    
    print("Enhanced signal optimization completed!")
    return results

if __name__ == "__main__":
    main()