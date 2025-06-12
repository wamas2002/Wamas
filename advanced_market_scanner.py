#!/usr/bin/env python3
"""
Advanced Market Scanner with AI Integration
Comprehensive market analysis with multiple AI models and advanced indicators
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMarketScanner:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        self.setup_scanner_database()
        
        # AI Models
        self.trend_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.momentum_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Comprehensive 100 cryptocurrency scanning universe
        self.scan_symbols = [
            # Major cryptocurrencies
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'TRX/USDT', 'TON/USDT', 'LINK/USDT',
            'MATIC/USDT', 'LTC/USDT', 'DOT/USDT', 'AVAX/USDT', 'UNI/USDT',
            
            # DeFi tokens
            'AAVE/USDT', 'MKR/USDT', 'COMP/USDT', 'SUSHI/USDT', 'CRV/USDT',
            'YFI/USDT', 'SNX/USDT', '1INCH/USDT', 'BAL/USDT', 'REN/USDT',
            
            # Layer 1 blockchains
            'ATOM/USDT', 'ALGO/USDT', 'XLM/USDT', 'VET/USDT', 'FIL/USDT',
            'EOS/USDT', 'TEZOS/USDT', 'IOTA/USDT', 'NEO/USDT', 'WAVES/USDT',
            
            # Smart contract platforms
            'NEAR/USDT', 'FTM/USDT', 'ONE/USDT', 'LUNA/USDT', 'EGLD/USDT',
            'THETA/USDT', 'HBAR/USDT', 'ICP/USDT', 'FLOW/USDT', 'MINA/USDT',
            
            # Gaming & NFT tokens
            'AXS/USDT', 'SAND/USDT', 'MANA/USDT', 'ENJ/USDT', 'CHZ/USDT',
            'GALA/USDT', 'IMX/USDT', 'APE/USDT', 'LRC/USDT', 'GMT/USDT',
            
            # Metaverse & Web3
            'CRO/USDT', 'HNT/USDT', 'ROSE/USDT', 'AR/USDT', 'STORJ/USDT',
            'BAT/USDT', 'GRT/USDT', 'OCEAN/USDT', 'FET/USDT', 'RNDR/USDT',
            
            # Privacy coins
            'XMR/USDT', 'ZEC/USDT', 'DASH/USDT', 'DCR/USDT', 'BEAM/USDT',
            
            # Exchange tokens
            'KCS/USDT', 'HT/USDT', 'OKB/USDT', 'FTT/USDT', 'LEO/USDT',
            
            # Stablecoins & derivatives
            'USDC/USDT', 'DAI/USDT', 'TUSD/USDT', 'USDP/USDT', 'FRAX/USDT',
            
            # Layer 2 solutions
            'MATIC/USDT', 'LRC/USDT', 'OMG/USDT', 'SKL/USDT', 'CTSI/USDT',
            
            # Emerging technologies
            'QNT/USDT', 'HOLO/USDT', 'ICX/USDT', 'ZIL/USDT', 'QTUM/USDT',
            'ONT/USDT', 'KAVA/USDT', 'BAND/USDT', 'RSR/USDT', 'RVN/USDT',
            
            # Additional promising projects
            'CELO/USDT', 'ZEN/USDT', 'REP/USDT', 'KNC/USDT', 'LSK/USDT',
            'SC/USDT', 'DGB/USDT', 'NKN/USDT', 'ANKR/USDT', 'CELR/USDT',
            'DENT/USDT', 'WAN/USDT', 'HOT/USDT', 'DUSK/USDT', 'ARDR/USDT'
        ]
        
        # Advanced indicator suite
        self.indicators = {
            'trend': ['EMA_9', 'EMA_21', 'EMA_50', 'EMA_200', 'MACD', 'ADX', 'Parabolic_SAR'],
            'momentum': ['RSI', 'Stochastic', 'Williams_R', 'CCI', 'ROC', 'MFI'],
            'volatility': ['Bollinger_Bands', 'ATR', 'Keltner_Channels', 'Donchian_Channels'],
            'volume': ['OBV', 'VWAP', 'Volume_Profile', 'Accumulation_Distribution', 'Chaikin_MF'],
            'market_structure': ['Pivot_Points', 'Fibonacci', 'Support_Resistance', 'Market_Profile']
        }
        
        # Scanner configurations
        self.scan_configs = {
            'breakout_scanner': {
                'timeframes': ['15m', '1h', '4h'],
                'min_volume_ratio': 1.5,
                'breakout_threshold': 0.02,
                'confirmation_candles': 2
            },
            'momentum_scanner': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_divergence': True,
                'volume_confirmation': True
            },
            'mean_reversion_scanner': {
                'bb_position_threshold': [0.1, 0.9],
                'rsi_extremes': [20, 80],
                'price_deviation': 2.0
            },
            'trend_following_scanner': {
                'ema_alignment': True,
                'adx_threshold': 25,
                'momentum_confirmation': True
            }
        }
        
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
                logger.info("Advanced market scanner connected to OKX")
            else:
                logger.warning("OKX credentials not configured")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_scanner_database(self):
        """Setup advanced scanner database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Comprehensive market scan results
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_scan_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        scan_type TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        
                        -- Price data
                        current_price REAL,
                        price_change_1h REAL,
                        price_change_4h REAL,
                        price_change_24h REAL,
                        volume_24h REAL,
                        volume_ratio REAL,
                        
                        -- Trend indicators
                        ema_9 REAL,
                        ema_21 REAL,
                        ema_50 REAL,
                        ema_200 REAL,
                        macd REAL,
                        macd_signal REAL,
                        macd_histogram REAL,
                        adx REAL,
                        sar REAL,
                        
                        -- Momentum indicators
                        rsi REAL,
                        stoch_k REAL,
                        stoch_d REAL,
                        williams_r REAL,
                        cci REAL,
                        roc REAL,
                        mfi REAL,
                        
                        -- Volatility indicators
                        bb_upper REAL,
                        bb_middle REAL,
                        bb_lower REAL,
                        bb_width REAL,
                        bb_position REAL,
                        atr REAL,
                        keltner_upper REAL,
                        keltner_lower REAL,
                        
                        -- Volume indicators
                        obv REAL,
                        vwap REAL,
                        ad_line REAL,
                        chaikin_mf REAL,
                        
                        -- Support/Resistance
                        pivot_point REAL,
                        resistance_1 REAL,
                        resistance_2 REAL,
                        support_1 REAL,
                        support_2 REAL,
                        
                        -- AI Analysis
                        trend_prediction TEXT,
                        trend_confidence REAL,
                        momentum_prediction TEXT,
                        momentum_confidence REAL,
                        ai_composite_score REAL,
                        
                        -- Scan results
                        scan_score REAL,
                        signal_strength TEXT,
                        entry_signal TEXT,
                        target_price REAL,
                        stop_loss REAL,
                        risk_reward_ratio REAL,
                        
                        -- Pattern recognition
                        chart_pattern TEXT,
                        pattern_confidence REAL,
                        breakout_probability REAL,
                        
                        -- Market structure
                        market_regime TEXT,
                        volatility_regime TEXT,
                        liquidity_score REAL,
                        
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # AI model training data
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        features TEXT NOT NULL,
                        target_trend REAL,
                        target_momentum REAL,
                        actual_outcome REAL,
                        prediction_accuracy REAL,
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Scanner performance tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS scanner_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_type TEXT NOT NULL,
                        total_scans INTEGER,
                        successful_signals INTEGER,
                        success_rate REAL,
                        avg_return REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Advanced scanner database initialized")
                
        except Exception as e:
            logger.error(f"Scanner database setup failed: {e}")
    
    def get_comprehensive_market_data(self, symbol, timeframe='1h', limit=500):
        """Get comprehensive market data with all timeframes"""
        if not self.exchange:
            return None
        
        try:
            # Primary timeframe data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 100:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Get ticker for additional data
            ticker = self.exchange.fetch_ticker(symbol)
            
            return df, ticker
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol} {timeframe}: {e}")
            return None
    
    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        try:
            # Trend indicators
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # MACD
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if not macd_data.empty and len(macd_data.columns) >= 3:
                df['macd'] = macd_data.iloc[:, 0]
                df['macd_signal'] = macd_data.iloc[:, 1]
                df['macd_histogram'] = macd_data.iloc[:, 2]
            else:
                df['macd'] = df['macd_signal'] = df['macd_histogram'] = 0
            
            # ADX
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if not adx_data.empty:
                df['adx'] = adx_data.iloc[:, 0] if len(adx_data.columns) > 0 else 25
            else:
                df['adx'] = 25
            
            # Parabolic SAR
            sar_data = ta.psar(df['high'], df['low'], df['close'])
            if not sar_data.empty:
                df['sar'] = sar_data.iloc[:, 0] if len(sar_data.columns) > 0 else df['close']
            else:
                df['sar'] = df['close']
            
            # Momentum indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Stochastic
            stoch_data = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if not stoch_data.empty and len(stoch_data.columns) >= 2:
                df['stoch_k'] = stoch_data.iloc[:, 0]
                df['stoch_d'] = stoch_data.iloc[:, 1]
            else:
                df['stoch_k'] = df['stoch_d'] = 50
            
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # ROC
            df['roc'] = ta.roc(df['close'], length=10)
            
            # MFI
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20, std=2)
            if not bb_data.empty and len(bb_data.columns) >= 3:
                df['bb_upper'] = bb_data.iloc[:, 0]
                df['bb_middle'] = bb_data.iloc[:, 1]
                df['bb_lower'] = bb_data.iloc[:, 2]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            else:
                df['bb_upper'] = df['bb_middle'] = df['bb_lower'] = df['close']
                df['bb_width'] = 2.0
                df['bb_position'] = 0.5
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Keltner Channels
            kc_data = ta.kc(df['high'], df['low'], df['close'], length=20)
            if not kc_data.empty and len(kc_data.columns) >= 3:
                df['keltner_upper'] = kc_data.iloc[:, 0]
                df['keltner_lower'] = kc_data.iloc[:, 2]
            else:
                df['keltner_upper'] = df['keltner_lower'] = df['close']
            
            # Volume indicators
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            df['ad_line'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
            df['chaikin_mf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
            
            # Volume ratio
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Calculate pivot points
            df = self._calculate_pivot_points(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Advanced indicator calculation failed: {e}")
            return df
    
    def _calculate_pivot_points(self, df):
        """Calculate pivot points and support/resistance levels"""
        try:
            # Standard pivot points
            df['pivot_point'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            df['resistance_1'] = 2 * df['pivot_point'] - df['low'].shift(1)
            df['support_1'] = 2 * df['pivot_point'] - df['high'].shift(1)
            df['resistance_2'] = df['pivot_point'] + (df['high'].shift(1) - df['low'].shift(1))
            df['support_2'] = df['pivot_point'] - (df['high'].shift(1) - df['low'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Pivot points calculation failed: {e}")
            return df
    
    def train_ai_models(self, df):
        """Train AI models for trend and momentum prediction"""
        try:
            if len(df) < 100:
                return False
            
            # Prepare features
            feature_columns = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'adx', 
                             'stoch_k', 'williams_r', 'cci', 'roc', 'mfi']
            
            # Clean data
            df_clean = df[feature_columns].dropna()
            if len(df_clean) < 50:
                return False
            
            # Create targets (future price movement)
            df_clean['future_return_1h'] = df['close'].shift(-1) / df['close'] - 1
            df_clean['future_return_4h'] = df['close'].shift(-4) / df['close'] - 1
            
            # Remove rows with NaN targets
            df_clean = df_clean.dropna()
            if len(df_clean) < 30:
                return False
            
            # Prepare training data
            X = df_clean[feature_columns].values
            y_trend = (df_clean['future_return_4h'] > 0.01).astype(int)  # Bullish trend
            y_momentum = (df_clean['future_return_1h'] > 0.005).astype(int)  # Short-term momentum
            
            if len(X) < 20:
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.trend_model.fit(X_scaled, y_trend)
            self.momentum_model.fit(X_scaled, y_momentum)
            
            return True
            
        except Exception as e:
            logger.error(f"AI model training failed: {e}")
            return False
    
    def get_ai_predictions(self, latest_data):
        """Get AI model predictions"""
        try:
            feature_columns = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'adx', 
                             'stoch_k', 'williams_r', 'cci', 'roc', 'mfi']
            
            # Prepare features
            features = []
            for col in feature_columns:
                value = latest_data.get(col, 50 if col == 'rsi' else 0)
                if pd.isna(value):
                    value = 50 if col == 'rsi' else 0
                features.append(float(value))
            
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Get predictions
            trend_prob = self.trend_model.predict_proba(features_scaled)[0]
            momentum_prob = self.momentum_model.predict_proba(features_scaled)[0]
            
            trend_prediction = 'BULLISH' if trend_prob[1] > 0.6 else 'BEARISH' if trend_prob[0] > 0.6 else 'NEUTRAL'
            momentum_prediction = 'BULLISH' if momentum_prob[1] > 0.6 else 'BEARISH' if momentum_prob[0] > 0.6 else 'NEUTRAL'
            
            trend_confidence = max(trend_prob) * 100
            momentum_confidence = max(momentum_prob) * 100
            
            # Composite AI score
            ai_composite_score = (trend_confidence + momentum_confidence) / 2
            
            return {
                'trend_prediction': trend_prediction,
                'trend_confidence': trend_confidence,
                'momentum_prediction': momentum_prediction,
                'momentum_confidence': momentum_confidence,
                'ai_composite_score': ai_composite_score
            }
            
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            return {
                'trend_prediction': 'NEUTRAL',
                'trend_confidence': 50.0,
                'momentum_prediction': 'NEUTRAL',
                'momentum_confidence': 50.0,
                'ai_composite_score': 50.0
            }
    
    def breakout_scanner(self, symbol, timeframe='1h'):
        """Advanced breakout detection scanner"""
        try:
            market_data = self.get_comprehensive_market_data(symbol, timeframe)
            if not market_data:
                return None
            
            df, ticker = market_data
            df = self.calculate_advanced_indicators(df)
            
            if len(df) < 50:
                return None
            
            latest = df.iloc[-1]
            
            # Breakout conditions
            breakout_score = 0
            
            # Volume breakout
            volume_ratio = latest['volume_ratio']
            if volume_ratio > 2.0:
                breakout_score += 30
            elif volume_ratio > 1.5:
                breakout_score += 20
            
            # Price breakout above resistance
            resistance = latest['resistance_1']
            if latest['close'] > resistance * 1.01:
                breakout_score += 25
            
            # Bollinger Band breakout
            bb_position = latest['bb_position']
            if bb_position > 0.95:
                breakout_score += 20
            elif bb_position < 0.05:
                breakout_score += 20
            
            # Momentum confirmation
            if latest['rsi'] > 60 and latest['macd'] > latest['macd_signal']:
                breakout_score += 15
            
            # Trend alignment
            if (latest['ema_9'] > latest['ema_21'] > latest['ema_50'] and 
                latest['close'] > latest['ema_9']):
                breakout_score += 10
            
            if breakout_score >= 50:
                return {
                    'scan_type': 'breakout',
                    'score': breakout_score,
                    'signal': 'BUY' if bb_position > 0.5 else 'SELL',
                    'confidence': min(95, breakout_score),
                    'volume_confirmation': volume_ratio > 1.5,
                    'price_target': resistance * 1.05 if bb_position > 0.5 else latest['support_1'] * 0.95
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Breakout scanner failed for {symbol}: {e}")
            return None
    
    def momentum_scanner(self, symbol, timeframe='1h'):
        """Advanced momentum detection scanner"""
        try:
            market_data = self.get_comprehensive_market_data(symbol, timeframe)
            if not market_data:
                return None
            
            df, ticker = market_data
            df = self.calculate_advanced_indicators(df)
            
            if len(df) < 50:
                return None
            
            latest = df.iloc[-1]
            
            # Momentum scoring
            momentum_score = 0
            signal_direction = 'HOLD'
            
            # RSI momentum
            rsi = latest['rsi']
            if rsi < 30:
                momentum_score += 25
                signal_direction = 'BUY'
            elif rsi > 70:
                momentum_score += 25
                signal_direction = 'SELL'
            elif 40 <= rsi <= 60:
                momentum_score += 10
            
            # MACD momentum
            if latest['macd'] > latest['macd_signal'] and latest['macd_histogram'] > 0:
                momentum_score += 20
                if signal_direction == 'HOLD':
                    signal_direction = 'BUY'
            elif latest['macd'] < latest['macd_signal'] and latest['macd_histogram'] < 0:
                momentum_score += 20
                if signal_direction == 'HOLD':
                    signal_direction = 'SELL'
            
            # Stochastic momentum
            if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
                momentum_score += 15
                if signal_direction == 'HOLD':
                    signal_direction = 'BUY'
            elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                momentum_score += 15
                if signal_direction == 'HOLD':
                    signal_direction = 'SELL'
            
            # Volume confirmation
            if latest['volume_ratio'] > 1.2:
                momentum_score += 10
            
            # ROC momentum
            if latest['roc'] > 2:
                momentum_score += 10
                if signal_direction == 'HOLD':
                    signal_direction = 'BUY'
            elif latest['roc'] < -2:
                momentum_score += 10
                if signal_direction == 'HOLD':
                    signal_direction = 'SELL'
            
            if momentum_score >= 50:
                return {
                    'scan_type': 'momentum',
                    'score': momentum_score,
                    'signal': signal_direction,
                    'confidence': min(95, momentum_score),
                    'rsi_signal': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral',
                    'macd_signal': 'bullish' if latest['macd'] > latest['macd_signal'] else 'bearish'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Momentum scanner failed for {symbol}: {e}")
            return None
    
    def mean_reversion_scanner(self, symbol, timeframe='1h'):
        """Advanced mean reversion detection scanner"""
        try:
            market_data = self.get_comprehensive_market_data(symbol, timeframe)
            if not market_data:
                return None
            
            df, ticker = market_data
            df = self.calculate_advanced_indicators(df)
            
            if len(df) < 50:
                return None
            
            latest = df.iloc[-1]
            
            # Mean reversion scoring
            reversion_score = 0
            signal_direction = 'HOLD'
            
            # Bollinger Band reversion
            bb_position = latest['bb_position']
            if bb_position < 0.1:  # Near lower band
                reversion_score += 30
                signal_direction = 'BUY'
            elif bb_position > 0.9:  # Near upper band
                reversion_score += 30
                signal_direction = 'SELL'
            
            # RSI extremes
            rsi = latest['rsi']
            if rsi < 20:
                reversion_score += 25
                if signal_direction == 'HOLD':
                    signal_direction = 'BUY'
            elif rsi > 80:
                reversion_score += 25
                if signal_direction == 'HOLD':
                    signal_direction = 'SELL'
            
            # Williams %R extremes
            williams_r = latest['williams_r']
            if williams_r < -90:
                reversion_score += 15
                if signal_direction == 'HOLD':
                    signal_direction = 'BUY'
            elif williams_r > -10:
                reversion_score += 15
                if signal_direction == 'HOLD':
                    signal_direction = 'SELL'
            
            # Price deviation from VWAP
            vwap = latest['vwap']
            price_deviation = abs(latest['close'] - vwap) / vwap
            if price_deviation > 0.03:  # 3% deviation
                reversion_score += 15
            
            # Low volatility environment (good for mean reversion)
            if latest['bb_width'] < 3:  # Low volatility
                reversion_score += 10
            
            if reversion_score >= 50:
                return {
                    'scan_type': 'mean_reversion',
                    'score': reversion_score,
                    'signal': signal_direction,
                    'confidence': min(95, reversion_score),
                    'bb_position': bb_position,
                    'price_deviation': price_deviation,
                    'target_price': vwap
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Mean reversion scanner failed for {symbol}: {e}")
            return None
    
    def comprehensive_market_scan(self):
        """Run comprehensive market scan across all strategies"""
        if not self.exchange:
            logger.error("Exchange connection required for market scanning")
            return []
        
        logger.info(f"Running comprehensive market scan on {len(self.scan_symbols)} cryptocurrency pairs...")
        
        scan_results = []
        ai_trained = False
        
        for symbol in self.scan_symbols:
            try:
                # Get comprehensive data
                market_data = self.get_comprehensive_market_data(symbol, '1h')
                if not market_data:
                    continue
                
                df, ticker = market_data
                df = self.calculate_advanced_indicators(df)
                
                if len(df) < 100:
                    continue
                
                # Train AI models (once per scan)
                if not ai_trained:
                    ai_trained = self.train_ai_models(df)
                
                latest = df.iloc[-1]
                current_price = ticker['last']
                
                # Prepare latest data for AI
                latest_data = latest.to_dict()
                latest_data['current_price'] = current_price
                
                # Get AI predictions
                ai_predictions = self.get_ai_predictions(latest_data)
                
                # Run all scanner types
                scanners = [
                    self.breakout_scanner(symbol, '1h'),
                    self.momentum_scanner(symbol, '1h'),
                    self.mean_reversion_scanner(symbol, '1h')
                ]
                
                # Process scanner results
                for scan_result in scanners:
                    if scan_result and scan_result['confidence'] >= 60:
                        
                        # Calculate comprehensive metrics
                        scan_score = self._calculate_composite_score(scan_result, ai_predictions, latest_data)
                        
                        # Determine signal strength
                        if scan_score >= 80:
                            signal_strength = 'STRONG'
                        elif scan_score >= 70:
                            signal_strength = 'MODERATE'
                        elif scan_score >= 60:
                            signal_strength = 'WEAK'
                        else:
                            continue
                        
                        # Calculate targets and stops
                        target_price, stop_loss, risk_reward = self._calculate_targets_stops(
                            latest_data, scan_result['signal'], scan_result['scan_type']
                        )
                        
                        # Detect patterns
                        pattern, pattern_confidence = self._detect_chart_patterns(df.tail(50))
                        
                        # Assess market regime
                        market_regime, volatility_regime = self._assess_market_regime(df.tail(100))
                        
                        # Create comprehensive result
                        comprehensive_result = {
                            'symbol': symbol,
                            'scan_type': scan_result['scan_type'],
                            'timeframe': '1h',
                            'current_price': current_price,
                            'price_change_1h': ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) >= 2 else 0,
                            'price_change_4h': ((current_price - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100) if len(df) >= 5 else 0,
                            'price_change_24h': ticker.get('percentage', 0),
                            'volume_24h': ticker.get('quoteVolume', 0),
                            'volume_ratio': latest_data.get('volume_ratio', 1),
                            
                            # Technical indicators
                            **{k: v for k, v in latest_data.items() if not pd.isna(v)},
                            
                            # AI predictions
                            **ai_predictions,
                            
                            # Scan results
                            'scan_score': scan_score,
                            'signal_strength': signal_strength,
                            'entry_signal': scan_result['signal'],
                            'target_price': target_price,
                            'stop_loss': stop_loss,
                            'risk_reward_ratio': risk_reward,
                            
                            # Pattern analysis
                            'chart_pattern': pattern,
                            'pattern_confidence': pattern_confidence,
                            'breakout_probability': self._calculate_breakout_probability(df.tail(20)),
                            
                            # Market structure
                            'market_regime': market_regime,
                            'volatility_regime': volatility_regime,
                            'liquidity_score': self._calculate_liquidity_score(latest_data),
                            
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Save to database
                        if self._save_scan_result(comprehensive_result):
                            scan_results.append(comprehensive_result)
                            logger.info(f"Scan hit: {symbol} {scan_result['scan_type']} - {signal_strength} {scan_result['signal']} ({scan_score:.1f})")
                
            except Exception as e:
                logger.error(f"Comprehensive scan failed for {symbol}: {e}")
                continue
        
        # Sort by scan score
        scan_results.sort(key=lambda x: x['scan_score'], reverse=True)
        
        # Generate scan summary
        summary = self._generate_scan_summary(scan_results)
        
        # Save comprehensive report
        with open('comprehensive_market_scan_report.json', 'w') as f:
            json.dump({
                'summary': summary,
                'scan_results': scan_results[:50],  # Top 50 results
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"Comprehensive market scan complete: {len(scan_results)} opportunities found")
        
        return scan_results
    
    def _calculate_composite_score(self, scan_result, ai_predictions, latest_data):
        """Calculate composite score combining technical and AI analysis"""
        try:
            base_score = scan_result['confidence']
            
            # AI enhancement
            ai_boost = (ai_predictions['ai_composite_score'] - 50) * 0.3
            base_score += ai_boost
            
            # Volume confirmation
            volume_ratio = latest_data.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                base_score += 5
            elif volume_ratio > 1.2:
                base_score += 2
            
            # Trend alignment bonus
            ema_9 = latest_data.get('ema_9', 0)
            ema_21 = latest_data.get('ema_21', 0)
            ema_50 = latest_data.get('ema_50', 0)
            
            if ema_9 > ema_21 > ema_50:  # Strong uptrend
                if scan_result['signal'] == 'BUY':
                    base_score += 8
            elif ema_9 < ema_21 < ema_50:  # Strong downtrend
                if scan_result['signal'] == 'SELL':
                    base_score += 8
            
            # ADX trend strength
            adx = latest_data.get('adx', 25)
            if adx > 30:
                base_score += 5
            elif adx > 25:
                base_score += 3
            
            return min(100, max(0, base_score))
            
        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            return scan_result['confidence']
    
    def _calculate_targets_stops(self, latest_data, signal, scan_type):
        """Calculate target price, stop loss, and risk/reward ratio"""
        try:
            current_price = latest_data['current_price']
            atr = latest_data.get('atr', current_price * 0.02)
            
            if signal == 'BUY':
                if scan_type == 'breakout':
                    target_price = current_price + (atr * 3)
                    stop_loss = current_price - (atr * 1.5)
                elif scan_type == 'momentum':
                    target_price = current_price + (atr * 2.5)
                    stop_loss = current_price - (atr * 1.2)
                else:  # mean_reversion
                    target_price = current_price + (atr * 1.5)
                    stop_loss = current_price - (atr * 0.8)
            
            else:  # SELL
                if scan_type == 'breakout':
                    target_price = current_price - (atr * 3)
                    stop_loss = current_price + (atr * 1.5)
                elif scan_type == 'momentum':
                    target_price = current_price - (atr * 2.5)
                    stop_loss = current_price + (atr * 1.2)
                else:  # mean_reversion
                    target_price = current_price - (atr * 1.5)
                    stop_loss = current_price + (atr * 0.8)
            
            # Calculate risk/reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(target_price - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return round(target_price, 4), round(stop_loss, 4), round(risk_reward, 2)
            
        except Exception as e:
            logger.error(f"Target/stop calculation failed: {e}")
            return None, None, 0
    
    def _detect_chart_patterns(self, df):
        """Detect chart patterns in recent price action"""
        try:
            if len(df) < 20:
                return 'NONE', 0
            
            highs = df['high']
            lows = df['low']
            closes = df['close']
            
            # Double top/bottom detection
            recent_high = highs.max()
            recent_low = lows.min()
            
            high_touches = (highs > recent_high * 0.99).sum()
            low_touches = (lows < recent_low * 1.01).sum()
            
            if high_touches >= 2:
                return 'DOUBLE_TOP', 75
            elif low_touches >= 2:
                return 'DOUBLE_BOTTOM', 75
            
            # Triangle patterns
            if (highs.iloc[-1] < highs.iloc[0] * 0.99 and 
                lows.iloc[-1] > lows.iloc[0] * 1.01):
                return 'SYMMETRICAL_TRIANGLE', 65
            
            # Flag pattern
            recent_range = highs.tail(10).max() - lows.tail(10).min()
            if recent_range < (recent_high - recent_low) * 0.3:
                return 'FLAG', 60
            
            return 'NONE', 0
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return 'NONE', 0
    
    def _assess_market_regime(self, df):
        """Assess current market regime"""
        try:
            if len(df) < 50:
                return 'UNKNOWN', 'UNKNOWN'
            
            # Trend assessment
            ema_20 = df['ema_21'].iloc[-1] if 'ema_21' in df.columns else df['close'].mean()
            ema_50 = df['ema_50'].iloc[-1] if 'ema_50' in df.columns else df['close'].mean()
            
            if ema_20 > ema_50 * 1.02:
                market_regime = 'TRENDING_UP'
            elif ema_20 < ema_50 * 0.98:
                market_regime = 'TRENDING_DOWN'
            else:
                market_regime = 'RANGING'
            
            # Volatility assessment
            volatility = df['close'].pct_change().std() * 100
            if volatility > 5:
                volatility_regime = 'HIGH_VOLATILITY'
            elif volatility < 2:
                volatility_regime = 'LOW_VOLATILITY'
            else:
                volatility_regime = 'NORMAL_VOLATILITY'
            
            return market_regime, volatility_regime
            
        except Exception as e:
            logger.error(f"Market regime assessment failed: {e}")
            return 'UNKNOWN', 'UNKNOWN'
    
    def _calculate_breakout_probability(self, df):
        """Calculate probability of price breakout"""
        try:
            if len(df) < 10:
                return 0.5
            
            # Volume analysis
            volume_trend = df['volume'].tail(5).mean() / df['volume'].head(5).mean()
            
            # Volatility compression
            volatility_current = df['close'].tail(5).std()
            volatility_avg = df['close'].std()
            compression = volatility_current / volatility_avg
            
            # Bollinger Band squeeze
            bb_width = df['bb_width'].iloc[-1] if 'bb_width' in df.columns else 2
            bb_squeeze = 1 / (bb_width + 0.1)  # Inverse relationship
            
            # Combine factors
            breakout_prob = (volume_trend * 0.4 + compression * 0.3 + bb_squeeze * 0.3) / 3
            
            return min(1.0, max(0.0, breakout_prob))
            
        except Exception as e:
            logger.error(f"Breakout probability calculation failed: {e}")
            return 0.5
    
    def _calculate_liquidity_score(self, latest_data):
        """Calculate liquidity score based on volume and spread"""
        try:
            volume_ratio = latest_data.get('volume_ratio', 1)
            volume_24h = latest_data.get('volume_24h', 0)
            
            # Volume component
            volume_score = min(100, volume_ratio * 50)
            
            # 24h volume component (normalized)
            volume_24h_score = min(50, np.log10(volume_24h + 1) * 10) if volume_24h > 0 else 0
            
            liquidity_score = (volume_score + volume_24h_score) / 2
            
            return round(liquidity_score, 1)
            
        except Exception as e:
            logger.error(f"Liquidity score calculation failed: {e}")
            return 50.0
    
    def _save_scan_result(self, result):
        """Save scan result to database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO market_scan_results (
                        symbol, scan_type, timeframe, current_price, price_change_1h, price_change_4h, price_change_24h,
                        volume_24h, volume_ratio, ema_9, ema_21, ema_50, ema_200, macd, macd_signal, macd_histogram,
                        adx, sar, rsi, stoch_k, stoch_d, williams_r, cci, roc, mfi, bb_upper, bb_middle, bb_lower,
                        bb_width, bb_position, atr, keltner_upper, keltner_lower, obv, vwap, ad_line, chaikin_mf,
                        pivot_point, resistance_1, resistance_2, support_1, support_2, trend_prediction, trend_confidence,
                        momentum_prediction, momentum_confidence, ai_composite_score, scan_score, signal_strength,
                        entry_signal, target_price, stop_loss, risk_reward_ratio, chart_pattern, pattern_confidence,
                        breakout_probability, market_regime, volatility_regime, liquidity_score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['symbol'], result['scan_type'], result['timeframe'], result['current_price'],
                    result.get('price_change_1h'), result.get('price_change_4h'), result.get('price_change_24h'),
                    result.get('volume_24h'), result.get('volume_ratio'), result.get('ema_9'), result.get('ema_21'),
                    result.get('ema_50'), result.get('ema_200'), result.get('macd'), result.get('macd_signal'),
                    result.get('macd_histogram'), result.get('adx'), result.get('sar'), result.get('rsi'),
                    result.get('stoch_k'), result.get('stoch_d'), result.get('williams_r'), result.get('cci'),
                    result.get('roc'), result.get('mfi'), result.get('bb_upper'), result.get('bb_middle'),
                    result.get('bb_lower'), result.get('bb_width'), result.get('bb_position'), result.get('atr'),
                    result.get('keltner_upper'), result.get('keltner_lower'), result.get('obv'), result.get('vwap'),
                    result.get('ad_line'), result.get('chaikin_mf'), result.get('pivot_point'), result.get('resistance_1'),
                    result.get('resistance_2'), result.get('support_1'), result.get('support_2'),
                    result.get('trend_prediction'), result.get('trend_confidence'), result.get('momentum_prediction'),
                    result.get('momentum_confidence'), result.get('ai_composite_score'), result['scan_score'],
                    result['signal_strength'], result['entry_signal'], result.get('target_price'),
                    result.get('stop_loss'), result.get('risk_reward_ratio'), result['chart_pattern'],
                    result['pattern_confidence'], result['breakout_probability'], result['market_regime'],
                    result['volatility_regime'], result['liquidity_score'], result['timestamp']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Scan result save failed: {e}")
            return False
    
    def _generate_scan_summary(self, scan_results):
        """Generate comprehensive scan summary"""
        if not scan_results:
            return {}
        
        total_scans = len(scan_results)
        strong_signals = len([r for r in scan_results if r['signal_strength'] == 'STRONG'])
        buy_signals = len([r for r in scan_results if r['entry_signal'] == 'BUY'])
        sell_signals = len([r for r in scan_results if r['entry_signal'] == 'SELL'])
        
        avg_score = sum(r['scan_score'] for r in scan_results) / total_scans
        avg_ai_score = sum(r['ai_composite_score'] for r in scan_results) / total_scans
        
        # Scan type distribution
        scan_types = {}
        for result in scan_results:
            scan_type = result['scan_type']
            scan_types[scan_type] = scan_types.get(scan_type, 0) + 1
        
        # Market regime analysis
        market_regimes = {}
        for result in scan_results:
            regime = result['market_regime']
            market_regimes[regime] = market_regimes.get(regime, 0) + 1
        
        return {
            'total_opportunities': total_scans,
            'strong_signals': strong_signals,
            'signal_distribution': {'BUY': buy_signals, 'SELL': sell_signals},
            'average_scan_score': round(avg_score, 1),
            'average_ai_score': round(avg_ai_score, 1),
            'scan_type_distribution': scan_types,
            'market_regime_distribution': market_regimes,
            'top_opportunities': scan_results[:10]
        }

def main():
    """Main scanner function"""
    scanner = AdvancedMarketScanner()
    
    if not scanner.exchange:
        print("OKX connection required for advanced market scanning")
        return
    
    # Run comprehensive market scan
    results = scanner.comprehensive_market_scan()
    
    print(f"\nAdvanced market scan completed: {len(results)} opportunities identified")
    
    return results

if __name__ == "__main__":
    main()