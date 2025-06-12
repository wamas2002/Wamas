#!/usr/bin/env python3
"""
TradingView Integration for Advanced Chart Analysis
Professional charting and technical analysis integration
"""

import sqlite3
import json
import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingViewAnalyzer:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        self.setup_analysis_database()
        
        # TradingView-style indicators
        self.indicators = {
            'trend': ['EMA', 'SMA', 'MACD', 'ADX'],
            'momentum': ['RSI', 'Stochastic', 'Williams %R', 'CCI'],
            'volatility': ['Bollinger Bands', 'ATR', 'Keltner Channels'],
            'volume': ['OBV', 'Volume Profile', 'VWAP', 'Money Flow Index'],
            'support_resistance': ['Pivot Points', 'Fibonacci', 'Support/Resistance Lines']
        }
        
        # Timeframes for multi-timeframe analysis
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        
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
                logger.info("TradingView analyzer connected to OKX")
            else:
                logger.warning("OKX credentials not configured")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_analysis_database(self):
        """Setup TradingView analysis database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tradingview_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        
                        -- Price action
                        current_price REAL,
                        price_change_24h REAL,
                        volume_24h REAL,
                        
                        -- Trend indicators
                        ema_20 REAL,
                        ema_50 REAL,
                        ema_200 REAL,
                        sma_20 REAL,
                        sma_50 REAL,
                        macd REAL,
                        macd_signal REAL,
                        macd_histogram REAL,
                        adx REAL,
                        
                        -- Momentum indicators
                        rsi REAL,
                        stoch_k REAL,
                        stoch_d REAL,
                        williams_r REAL,
                        cci REAL,
                        
                        -- Volatility indicators
                        bb_upper REAL,
                        bb_middle REAL,
                        bb_lower REAL,
                        bb_width REAL,
                        atr REAL,
                        
                        -- Volume indicators
                        obv REAL,
                        vwap REAL,
                        mfi REAL,
                        
                        -- Support/Resistance
                        pivot_point REAL,
                        resistance_1 REAL,
                        resistance_2 REAL,
                        support_1 REAL,
                        support_2 REAL,
                        
                        -- Analysis results
                        trend_direction TEXT,
                        trend_strength TEXT,
                        momentum_signal TEXT,
                        volatility_level TEXT,
                        volume_confirmation TEXT,
                        
                        -- Trading signals
                        overall_signal TEXT,
                        signal_strength REAL,
                        risk_level TEXT,
                        target_price REAL,
                        stop_loss REAL,
                        
                        -- Pattern recognition
                        chart_pattern TEXT,
                        pattern_confidence REAL,
                        breakout_potential TEXT,
                        
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("TradingView analysis database initialized")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_multi_timeframe_data(self, symbol, limit=200):
        """Get data for multiple timeframes"""
        if not self.exchange:
            return {}
        
        timeframe_data = {}
        
        for tf in self.timeframes:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                if ohlcv and len(ohlcv) >= 50:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    timeframe_data[tf] = df
                    logger.info(f"Loaded {len(df)} candles for {symbol} {tf}")
            except Exception as e:
                logger.error(f"Failed to get {tf} data for {symbol}: {e}")
                continue
        
        return timeframe_data
    
    def calculate_tradingview_indicators(self, df):
        """Calculate comprehensive TradingView-style indicators"""
        try:
            # Trend indicators
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            
            # MACD
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if not macd_data.empty:
                df['macd'] = macd_data.iloc[:, 0] if len(macd_data.columns) > 0 else 0
                df['macd_signal'] = macd_data.iloc[:, 1] if len(macd_data.columns) > 1 else 0
                df['macd_histogram'] = macd_data.iloc[:, 2] if len(macd_data.columns) > 2 else 0
            
            # ADX
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if not adx_data.empty:
                df['adx'] = adx_data.iloc[:, 0] if len(adx_data.columns) > 0 else 50
            
            # Momentum indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Stochastic
            stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if not stoch_data.empty:
                df['stoch_k'] = stoch_data.iloc[:, 0] if len(stoch_data.columns) > 0 else 50
                df['stoch_d'] = stoch_data.iloc[:, 1] if len(stoch_data.columns) > 1 else 50
            
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20, std=2)
            if not bb_data.empty:
                df['bb_upper'] = bb_data.iloc[:, 0] if len(bb_data.columns) > 0 else df['close']
                df['bb_middle'] = bb_data.iloc[:, 1] if len(bb_data.columns) > 1 else df['close']
                df['bb_lower'] = bb_data.iloc[:, 2] if len(bb_data.columns) > 2 else df['close']
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Volume indicators
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            
            # Calculate pivot points
            df = self._calculate_pivot_points(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return df
    
    def _calculate_pivot_points(self, df):
        """Calculate pivot points and support/resistance levels"""
        try:
            # Classic pivot points calculation
            df['pivot_point'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            df['resistance_1'] = 2 * df['pivot_point'] - df['low'].shift(1)
            df['support_1'] = 2 * df['pivot_point'] - df['high'].shift(1)
            df['resistance_2'] = df['pivot_point'] + (df['high'].shift(1) - df['low'].shift(1))
            df['support_2'] = df['pivot_point'] - (df['high'].shift(1) - df['low'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Pivot point calculation failed: {e}")
            return df
    
    def analyze_trend_direction(self, df):
        """Analyze trend direction using multiple indicators"""
        try:
            latest = df.iloc[-1]
            
            trend_signals = []
            
            # EMA trend
            if latest['ema_20'] > latest['ema_50'] > latest['ema_200']:
                trend_signals.append('STRONG_BULLISH')
            elif latest['ema_20'] > latest['ema_50']:
                trend_signals.append('BULLISH')
            elif latest['ema_20'] < latest['ema_50'] < latest['ema_200']:
                trend_signals.append('STRONG_BEARISH')
            elif latest['ema_20'] < latest['ema_50']:
                trend_signals.append('BEARISH')
            else:
                trend_signals.append('NEUTRAL')
            
            # MACD trend
            if latest['macd'] > latest['macd_signal'] and latest['macd'] > 0:
                trend_signals.append('BULLISH')
            elif latest['macd'] < latest['macd_signal'] and latest['macd'] < 0:
                trend_signals.append('BEARISH')
            else:
                trend_signals.append('NEUTRAL')
            
            # ADX strength
            adx_value = latest['adx'] if not pd.isna(latest['adx']) else 25
            if adx_value > 50:
                strength = 'VERY_STRONG'
            elif adx_value > 25:
                strength = 'STRONG'
            elif adx_value > 20:
                strength = 'MODERATE'
            else:
                strength = 'WEAK'
            
            # Determine overall trend
            bullish_count = trend_signals.count('BULLISH') + trend_signals.count('STRONG_BULLISH')
            bearish_count = trend_signals.count('BEARISH') + trend_signals.count('STRONG_BEARISH')
            
            if bullish_count > bearish_count:
                direction = 'BULLISH'
            elif bearish_count > bullish_count:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            return direction, strength
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return 'NEUTRAL', 'WEAK'
    
    def analyze_momentum_signals(self, df):
        """Analyze momentum using multiple oscillators"""
        try:
            latest = df.iloc[-1]
            
            momentum_signals = []
            
            # RSI analysis
            rsi = latest['rsi'] if not pd.isna(latest['rsi']) else 50
            if rsi > 70:
                momentum_signals.append('OVERBOUGHT')
            elif rsi < 30:
                momentum_signals.append('OVERSOLD')
            elif rsi > 55:
                momentum_signals.append('BULLISH')
            elif rsi < 45:
                momentum_signals.append('BEARISH')
            else:
                momentum_signals.append('NEUTRAL')
            
            # Stochastic analysis
            stoch_k = latest['stoch_k'] if not pd.isna(latest['stoch_k']) else 50
            stoch_d = latest['stoch_d'] if not pd.isna(latest['stoch_d']) else 50
            
            if stoch_k > 80 and stoch_d > 80:
                momentum_signals.append('OVERBOUGHT')
            elif stoch_k < 20 and stoch_d < 20:
                momentum_signals.append('OVERSOLD')
            elif stoch_k > stoch_d and stoch_k > 50:
                momentum_signals.append('BULLISH')
            elif stoch_k < stoch_d and stoch_k < 50:
                momentum_signals.append('BEARISH')
            else:
                momentum_signals.append('NEUTRAL')
            
            # Williams %R analysis
            williams_r = latest['williams_r'] if not pd.isna(latest['williams_r']) else -50
            if williams_r > -20:
                momentum_signals.append('OVERBOUGHT')
            elif williams_r < -80:
                momentum_signals.append('OVERSOLD')
            
            # Determine overall momentum
            if momentum_signals.count('OVERBOUGHT') >= 2:
                return 'OVERBOUGHT'
            elif momentum_signals.count('OVERSOLD') >= 2:
                return 'OVERSOLD'
            elif momentum_signals.count('BULLISH') > momentum_signals.count('BEARISH'):
                return 'BULLISH'
            elif momentum_signals.count('BEARISH') > momentum_signals.count('BULLISH'):
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return 'NEUTRAL'
    
    def analyze_volatility_level(self, df):
        """Analyze volatility using Bollinger Bands and ATR"""
        try:
            latest = df.iloc[-1]
            
            # Bollinger Band width analysis
            bb_width = latest['bb_width'] if not pd.isna(latest['bb_width']) else 5
            bb_width_avg = df['bb_width'].tail(20).mean()
            
            # ATR analysis
            atr = latest['atr'] if not pd.isna(latest['atr']) else 0
            atr_avg = df['atr'].tail(20).mean()
            
            volatility_factors = []
            
            if bb_width > bb_width_avg * 1.5:
                volatility_factors.append('HIGH')
            elif bb_width < bb_width_avg * 0.7:
                volatility_factors.append('LOW')
            else:
                volatility_factors.append('MEDIUM')
            
            if atr > atr_avg * 1.3:
                volatility_factors.append('HIGH')
            elif atr < atr_avg * 0.8:
                volatility_factors.append('LOW')
            else:
                volatility_factors.append('MEDIUM')
            
            # Determine overall volatility
            if volatility_factors.count('HIGH') >= 1:
                return 'HIGH'
            elif volatility_factors.count('LOW') >= 1:
                return 'LOW'
            else:
                return 'MEDIUM'
                
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return 'MEDIUM'
    
    def analyze_volume_confirmation(self, df):
        """Analyze volume confirmation"""
        try:
            latest = df.iloc[-1]
            
            # Volume analysis
            volume_avg = df['volume'].tail(20).mean()
            current_volume = latest['volume']
            
            # OBV trend
            obv_trend = 'NEUTRAL'
            if len(df) >= 5:
                obv_recent = df['obv'].tail(5).mean()
                obv_older = df['obv'].tail(20).head(15).mean()
                
                if obv_recent > obv_older * 1.05:
                    obv_trend = 'INCREASING'
                elif obv_recent < obv_older * 0.95:
                    obv_trend = 'DECREASING'
            
            # MFI analysis
            mfi = latest['mfi'] if not pd.isna(latest['mfi']) else 50
            
            if current_volume > volume_avg * 1.5 and obv_trend == 'INCREASING' and mfi > 50:
                return 'STRONG_BULLISH'
            elif current_volume > volume_avg * 1.5 and obv_trend == 'DECREASING' and mfi < 50:
                return 'STRONG_BEARISH'
            elif current_volume > volume_avg * 1.2:
                return 'CONFIRMING'
            elif current_volume < volume_avg * 0.7:
                return 'WEAK'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return 'NEUTRAL'
    
    def detect_chart_patterns(self, df):
        """Detect chart patterns"""
        try:
            if len(df) < 50:
                return 'NONE', 0
            
            # Simple pattern detection based on price action
            highs = df['high'].tail(20)
            lows = df['low'].tail(20)
            closes = df['close'].tail(20)
            
            # Double top/bottom detection
            recent_high = highs.max()
            recent_low = lows.min()
            
            high_touches = (highs > recent_high * 0.99).sum()
            low_touches = (lows < recent_low * 1.01).sum()
            
            if high_touches >= 2:
                return 'DOUBLE_TOP', 75
            elif low_touches >= 2:
                return 'DOUBLE_BOTTOM', 75
            
            # Triangle pattern detection
            if highs.iloc[-1] < highs.iloc[0] and lows.iloc[-1] > lows.iloc[0]:
                return 'SYMMETRICAL_TRIANGLE', 65
            elif highs.iloc[-1] < highs.iloc[0] and lows.iloc[-1] <= lows.iloc[0]:
                return 'DESCENDING_TRIANGLE', 70
            elif highs.iloc[-1] >= highs.iloc[0] and lows.iloc[-1] > lows.iloc[0]:
                return 'ASCENDING_TRIANGLE', 70
            
            return 'NONE', 0
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return 'NONE', 0
    
    def generate_tradingview_analysis(self, symbol, timeframe='1h'):
        """Generate comprehensive TradingView-style analysis"""
        try:
            # Get multi-timeframe data
            timeframe_data = self.get_multi_timeframe_data(symbol)
            
            if timeframe not in timeframe_data:
                logger.error(f"No data available for {symbol} {timeframe}")
                return None
            
            df = timeframe_data[timeframe]
            
            # Calculate all indicators
            df = self.calculate_tradingview_indicators(df)
            
            if df.empty:
                return None
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Perform analysis
            trend_direction, trend_strength = self.analyze_trend_direction(df)
            momentum_signal = self.analyze_momentum_signals(df)
            volatility_level = self.analyze_volatility_level(df)
            volume_confirmation = self.analyze_volume_confirmation(df)
            chart_pattern, pattern_confidence = self.detect_chart_patterns(df)
            
            # Generate overall signal
            overall_signal, signal_strength = self._generate_overall_signal(
                trend_direction, momentum_signal, volume_confirmation
            )
            
            # Calculate targets and stops
            target_price, stop_loss = self._calculate_targets_stops(df, overall_signal)
            
            # Get latest indicator values
            latest = df.iloc[-1]
            
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'price_change_24h': ((current_price - df['close'].iloc[-25]) / df['close'].iloc[-25] * 100) if len(df) >= 25 else 0,
                'volume_24h': df['volume'].tail(24).sum(),
                
                # Technical indicators
                'ema_20': latest['ema_20'] if not pd.isna(latest['ema_20']) else None,
                'ema_50': latest['ema_50'] if not pd.isna(latest['ema_50']) else None,
                'ema_200': latest['ema_200'] if not pd.isna(latest['ema_200']) else None,
                'sma_20': latest['sma_20'] if not pd.isna(latest['sma_20']) else None,
                'sma_50': latest['sma_50'] if not pd.isna(latest['sma_50']) else None,
                'macd': latest['macd'] if not pd.isna(latest['macd']) else None,
                'macd_signal': latest['macd_signal'] if not pd.isna(latest['macd_signal']) else None,
                'macd_histogram': latest['macd_histogram'] if not pd.isna(latest['macd_histogram']) else None,
                'adx': latest['adx'] if not pd.isna(latest['adx']) else None,
                'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else None,
                'stoch_k': latest['stoch_k'] if not pd.isna(latest['stoch_k']) else None,
                'stoch_d': latest['stoch_d'] if not pd.isna(latest['stoch_d']) else None,
                'williams_r': latest['williams_r'] if not pd.isna(latest['williams_r']) else None,
                'cci': latest['cci'] if not pd.isna(latest['cci']) else None,
                'bb_upper': latest['bb_upper'] if not pd.isna(latest['bb_upper']) else None,
                'bb_middle': latest['bb_middle'] if not pd.isna(latest['bb_middle']) else None,
                'bb_lower': latest['bb_lower'] if not pd.isna(latest['bb_lower']) else None,
                'bb_width': latest['bb_width'] if not pd.isna(latest['bb_width']) else None,
                'atr': latest['atr'] if not pd.isna(latest['atr']) else None,
                'obv': latest['obv'] if not pd.isna(latest['obv']) else None,
                'vwap': latest['vwap'] if not pd.isna(latest['vwap']) else None,
                'mfi': latest['mfi'] if not pd.isna(latest['mfi']) else None,
                'pivot_point': latest['pivot_point'] if not pd.isna(latest['pivot_point']) else None,
                'resistance_1': latest['resistance_1'] if not pd.isna(latest['resistance_1']) else None,
                'resistance_2': latest['resistance_2'] if not pd.isna(latest['resistance_2']) else None,
                'support_1': latest['support_1'] if not pd.isna(latest['support_1']) else None,
                'support_2': latest['support_2'] if not pd.isna(latest['support_2']) else None,
                
                # Analysis results
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'momentum_signal': momentum_signal,
                'volatility_level': volatility_level,
                'volume_confirmation': volume_confirmation,
                'overall_signal': overall_signal,
                'signal_strength': signal_strength,
                'risk_level': self._assess_risk_level(signal_strength, volatility_level),
                'target_price': target_price,
                'stop_loss': stop_loss,
                'chart_pattern': chart_pattern,
                'pattern_confidence': pattern_confidence,
                'breakout_potential': self._assess_breakout_potential(df, chart_pattern, pattern_confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"TradingView analysis failed for {symbol}: {e}")
            return None
    
    def _generate_overall_signal(self, trend, momentum, volume):
        """Generate overall trading signal"""
        score = 50  # Base score
        
        # Trend contribution
        if trend == 'STRONG_BULLISH':
            score += 25
        elif trend == 'BULLISH':
            score += 15
        elif trend == 'STRONG_BEARISH':
            score -= 25
        elif trend == 'BEARISH':
            score -= 15
        
        # Momentum contribution
        if momentum == 'OVERSOLD':
            score += 20
        elif momentum == 'BULLISH':
            score += 10
        elif momentum == 'OVERBOUGHT':
            score -= 20
        elif momentum == 'BEARISH':
            score -= 10
        
        # Volume confirmation
        if volume == 'STRONG_BULLISH':
            score += 15
        elif volume == 'CONFIRMING':
            score += 5
        elif volume == 'STRONG_BEARISH':
            score -= 15
        elif volume == 'WEAK':
            score -= 5
        
        # Determine signal
        if score >= 75:
            return 'STRONG_BUY', score
        elif score >= 60:
            return 'BUY', score
        elif score <= 25:
            return 'STRONG_SELL', score
        elif score <= 40:
            return 'SELL', score
        else:
            return 'HOLD', score
    
    def _calculate_targets_stops(self, df, signal):
        """Calculate target and stop loss prices"""
        try:
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else current_price * 0.02
            
            if 'BUY' in signal:
                target_price = current_price + (atr * 2)
                stop_loss = current_price - (atr * 1.5)
            elif 'SELL' in signal:
                target_price = current_price - (atr * 2)
                stop_loss = current_price + (atr * 1.5)
            else:
                target_price = current_price
                stop_loss = current_price - (atr * 1)
            
            return round(target_price, 4), round(stop_loss, 4)
            
        except Exception as e:
            logger.error(f"Target/stop calculation failed: {e}")
            return None, None
    
    def _assess_risk_level(self, signal_strength, volatility):
        """Assess risk level"""
        if signal_strength >= 75 and volatility == 'LOW':
            return 'LOW'
        elif signal_strength >= 60 and volatility != 'HIGH':
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _assess_breakout_potential(self, df, pattern, confidence):
        """Assess breakout potential"""
        if pattern in ['ASCENDING_TRIANGLE', 'SYMMETRICAL_TRIANGLE'] and confidence > 60:
            return 'HIGH'
        elif pattern == 'DOUBLE_BOTTOM' and confidence > 70:
            return 'HIGH'
        elif pattern != 'NONE':
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def save_tradingview_analysis(self, analysis):
        """Save TradingView analysis to database"""
        if not analysis:
            return False
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO tradingview_analysis (
                        symbol, timeframe, current_price, price_change_24h, volume_24h,
                        ema_20, ema_50, ema_200, sma_20, sma_50,
                        macd, macd_signal, macd_histogram, adx,
                        rsi, stoch_k, stoch_d, williams_r, cci,
                        bb_upper, bb_middle, bb_lower, bb_width, atr,
                        obv, vwap, mfi,
                        pivot_point, resistance_1, resistance_2, support_1, support_2,
                        trend_direction, trend_strength, momentum_signal, volatility_level, volume_confirmation,
                        overall_signal, signal_strength, risk_level, target_price, stop_loss,
                        chart_pattern, pattern_confidence, breakout_potential, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis['symbol'], analysis['timeframe'], analysis['current_price'],
                    analysis.get('price_change_24h'), analysis.get('volume_24h'),
                    analysis.get('ema_20'), analysis.get('ema_50'), analysis.get('ema_200'),
                    analysis.get('sma_20'), analysis.get('sma_50'),
                    analysis.get('macd'), analysis.get('macd_signal'), analysis.get('macd_histogram'),
                    analysis.get('adx'), analysis.get('rsi'), analysis.get('stoch_k'), analysis.get('stoch_d'),
                    analysis.get('williams_r'), analysis.get('cci'),
                    analysis.get('bb_upper'), analysis.get('bb_middle'), analysis.get('bb_lower'),
                    analysis.get('bb_width'), analysis.get('atr'),
                    analysis.get('obv'), analysis.get('vwap'), analysis.get('mfi'),
                    analysis.get('pivot_point'), analysis.get('resistance_1'), analysis.get('resistance_2'),
                    analysis.get('support_1'), analysis.get('support_2'),
                    analysis['trend_direction'], analysis['trend_strength'], analysis['momentum_signal'],
                    analysis['volatility_level'], analysis['volume_confirmation'],
                    analysis['overall_signal'], analysis['signal_strength'], analysis['risk_level'],
                    analysis.get('target_price'), analysis.get('stop_loss'),
                    analysis['chart_pattern'], analysis['pattern_confidence'], analysis['breakout_potential'],
                    analysis['timestamp']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Analysis save failed: {e}")
            return False
    
    def run_tradingview_analysis_suite(self, symbols=None):
        """Run comprehensive TradingView analysis for multiple symbols"""
        if not self.exchange:
            logger.error("Exchange connection required")
            return []
        
        if not symbols:
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        
        logger.info(f"Running TradingView analysis for {len(symbols)} symbols...")
        
        analyses = []
        
        for symbol in symbols:
            try:
                # Analyze multiple timeframes
                for timeframe in ['1h', '4h', '1d']:
                    analysis = self.generate_tradingview_analysis(symbol, timeframe)
                    
                    if analysis:
                        if self.save_tradingview_analysis(analysis):
                            analyses.append(analysis)
                            logger.info(f"TradingView analysis complete: {symbol} {timeframe} - {analysis['overall_signal']} ({analysis['signal_strength']:.1f})")
                
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
                continue
        
        # Generate summary report
        summary = self._generate_analysis_summary(analyses)
        
        # Save comprehensive report
        with open('tradingview_analysis_report.json', 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_analyses': analyses[:20]  # Store top 20
            }, f, indent=2, default=str)
        
        # Print summary
        self._print_analysis_summary(summary, analyses)
        
        return analyses
    
    def _generate_analysis_summary(self, analyses):
        """Generate analysis summary"""
        if not analyses:
            return {}
        
        strong_buy = len([a for a in analyses if a['overall_signal'] == 'STRONG_BUY'])
        buy = len([a for a in analyses if a['overall_signal'] == 'BUY'])
        hold = len([a for a in analyses if a['overall_signal'] == 'HOLD'])
        sell = len([a for a in analyses if a['overall_signal'] == 'SELL'])
        strong_sell = len([a for a in analyses if a['overall_signal'] == 'STRONG_SELL'])
        
        avg_signal_strength = sum(a['signal_strength'] for a in analyses) / len(analyses)
        
        return {
            'total_analyses': len(analyses),
            'signal_distribution': {
                'STRONG_BUY': strong_buy,
                'BUY': buy,
                'HOLD': hold,
                'SELL': sell,
                'STRONG_SELL': strong_sell
            },
            'average_signal_strength': round(avg_signal_strength, 1),
            'bullish_sentiment': round((strong_buy + buy) / len(analyses) * 100, 1),
            'bearish_sentiment': round((strong_sell + sell) / len(analyses) * 100, 1)
        }
    
    def _print_analysis_summary(self, summary, analyses):
        """Print analysis summary"""
        print("\n" + "="*70)
        print("TRADINGVIEW ANALYSIS SUMMARY")
        print("="*70)
        
        if summary:
            print(f"Total Analyses: {summary['total_analyses']}")
            print(f"Average Signal Strength: {summary['average_signal_strength']:.1f}")
            print(f"Bullish Sentiment: {summary['bullish_sentiment']:.1f}%")
            print(f"Bearish Sentiment: {summary['bearish_sentiment']:.1f}%")
            
            print(f"\nSignal Distribution:")
            dist = summary['signal_distribution']
            print(f"  STRONG BUY: {dist['STRONG_BUY']}")
            print(f"  BUY: {dist['BUY']}")
            print(f"  HOLD: {dist['HOLD']}")
            print(f"  SELL: {dist['SELL']}")
            print(f"  STRONG SELL: {dist['STRONG_SELL']}")
        
        # Show top analyses
        if analyses:
            top_analyses = sorted(analyses, key=lambda x: x['signal_strength'], reverse=True)[:5]
            print(f"\nTop Signal Opportunities:")
            for analysis in top_analyses:
                print(f"  {analysis['symbol']} ({analysis['timeframe']}): {analysis['overall_signal']}")
                print(f"    Strength: {analysis['signal_strength']:.1f}, Risk: {analysis['risk_level']}")
                if analysis.get('target_price'):
                    print(f"    Target: ${analysis['target_price']:.4f}, Stop: ${analysis['stop_loss']:.4f}")
        
        print(f"\nTradingView analysis report saved: tradingview_analysis_report.json")

def main():
    """Main TradingView analysis function"""
    analyzer = TradingViewAnalyzer()
    
    if not analyzer.exchange:
        print("OKX connection required for TradingView analysis")
        return
    
    # Run comprehensive analysis
    analyses = analyzer.run_tradingview_analysis_suite()
    
    print("TradingView analysis suite completed!")
    return analyses

if __name__ == "__main__":
    main()