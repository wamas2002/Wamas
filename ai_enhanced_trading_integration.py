#!/usr/bin/env python3
"""
AI-Enhanced Trading System Integration
Integrates advanced market scanner with unified trading platform
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

class AIEnhancedTradingIntegration:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        self.setup_ai_enhanced_database()
        
        # AI Models for enhanced analysis
        self.trend_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.momentum_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        
        # Comprehensive 100 cryptocurrency trading pairs for AI analysis
        self.trading_pairs = [
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
            'LRC/USDT', 'OMG/USDT', 'SKL/USDT', 'CTSI/USDT', 'METIS/USDT',
            
            # Emerging technologies
            'QNT/USDT', 'HOLO/USDT', 'ICX/USDT', 'ZIL/USDT', 'QTUM/USDT',
            'ONT/USDT', 'KAVA/USDT', 'BAND/USDT', 'RSR/USDT', 'RVN/USDT',
            
            # Additional promising projects
            'CELO/USDT', 'ZEN/USDT', 'REP/USDT', 'KNC/USDT', 'LSK/USDT',
            'SC/USDT', 'DGB/USDT', 'NKN/USDT', 'ANKR/USDT', 'CELR/USDT',
            'DENT/USDT', 'WAN/USDT', 'HOT/USDT', 'DUSK/USDT', 'ARDR/USDT'
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
                logger.info("AI-Enhanced Trading Integration connected to OKX")
            else:
                logger.warning("OKX credentials not configured")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_ai_enhanced_database(self):
        """Setup AI-enhanced trading database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Enhanced signals table with AI integration
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_enhanced_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        ai_score REAL DEFAULT 0,
                        scan_type TEXT DEFAULT 'enhanced',
                        
                        -- Technical indicators
                        rsi REAL,
                        macd REAL,
                        macd_signal REAL,
                        bb_position REAL,
                        volume_ratio REAL,
                        adx REAL,
                        stoch_k REAL,
                        williams_r REAL,
                        
                        -- AI predictions
                        trend_prediction TEXT,
                        trend_confidence REAL,
                        momentum_prediction TEXT,
                        momentum_confidence REAL,
                        
                        -- Pattern analysis
                        pattern_detected TEXT DEFAULT 'none',
                        pattern_confidence REAL DEFAULT 0,
                        breakout_probability REAL DEFAULT 0,
                        
                        -- Risk management
                        target_price REAL,
                        stop_loss REAL,
                        risk_reward_ratio REAL DEFAULT 0,
                        
                        -- Market structure
                        market_regime TEXT DEFAULT 'unknown',
                        volatility_regime TEXT DEFAULT 'normal',
                        liquidity_score REAL DEFAULT 50,
                        
                        reasoning TEXT,
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Portfolio optimization table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_optimization (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_value_usd REAL,
                        risk_score REAL,
                        diversification_score REAL,
                        performance_score REAL,
                        optimization_suggestions TEXT,
                        timestamp TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # AI model performance tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_type TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        samples_trained INTEGER,
                        last_trained DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("AI-enhanced database initialized")
                
        except Exception as e:
            logger.error(f"AI database setup failed: {e}")
    
    def get_enhanced_market_data(self, symbol, timeframe='1h', limit=200):
        """Get enhanced market data with comprehensive analysis"""
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
            
            # Get ticker for additional data
            ticker = self.exchange.fetch_ticker(symbol)
            
            return df, ticker
            
        except Exception as e:
            logger.error(f"Enhanced market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_enhanced_indicators(self, df):
        """Calculate comprehensive technical indicators with error handling"""
        try:
            if len(df) < 50:
                return df
            
            # Trend indicators
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            
            # MACD with error handling
            try:
                macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if macd_data is not None and not macd_data.empty and len(macd_data.columns) >= 3:
                    df['macd'] = macd_data.iloc[:, 0]
                    df['macd_signal'] = macd_data.iloc[:, 1]
                    df['macd_histogram'] = macd_data.iloc[:, 2]
                else:
                    df['macd'] = df['macd_signal'] = df['macd_histogram'] = 0
            except:
                df['macd'] = df['macd_signal'] = df['macd_histogram'] = 0
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # ADX with error handling
            try:
                adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
                if adx_data is not None and not adx_data.empty:
                    df['adx'] = adx_data.iloc[:, 0] if len(adx_data.columns) > 0 else 25
                else:
                    df['adx'] = 25
            except:
                df['adx'] = 25
            
            # Stochastic with error handling
            try:
                stoch_data = ta.stoch(df['high'], df['low'], df['close'])
                if stoch_data is not None and not stoch_data.empty and len(stoch_data.columns) >= 2:
                    df['stoch_k'] = stoch_data.iloc[:, 0]
                    df['stoch_d'] = stoch_data.iloc[:, 1]
                else:
                    df['stoch_k'] = df['stoch_d'] = 50
            except:
                df['stoch_k'] = df['stoch_d'] = 50
            
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # Bollinger Bands with error handling
            try:
                bb_data = ta.bbands(df['close'], length=20, std=2)
                if bb_data is not None and not bb_data.empty and len(bb_data.columns) >= 3:
                    df['bb_upper'] = bb_data.iloc[:, 0]
                    df['bb_middle'] = bb_data.iloc[:, 1]
                    df['bb_lower'] = bb_data.iloc[:, 2]
                    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                else:
                    df['bb_upper'] = df['bb_middle'] = df['bb_lower'] = df['close']
                    df['bb_position'] = 0.5
            except:
                df['bb_upper'] = df['bb_middle'] = df['bb_lower'] = df['close']
                df['bb_position'] = 0.5
            
            # Volume analysis
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Fill any NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Enhanced indicator calculation failed: {e}")
            return df
    
    def train_ai_models(self, df):
        """Train AI models with enhanced features"""
        try:
            if len(df) < 100:
                return False
            
            # Enhanced feature set
            feature_columns = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'adx', 'stoch_k', 'williams_r']
            
            # Clean data and ensure we have required columns
            available_features = [col for col in feature_columns if col in df.columns]
            if len(available_features) < 5:
                return False
            
            df_clean = df[available_features].dropna()
            if len(df_clean) < 50:
                return False
            
            # Create targets
            df_clean['future_return_1h'] = df['close'].shift(-1) / df['close'] - 1
            df_clean['future_return_4h'] = df['close'].shift(-4) / df['close'] - 1
            
            # Remove NaN targets
            df_clean = df_clean.dropna()
            if len(df_clean) < 30:
                return False
            
            # Prepare training data
            X = df_clean[available_features].values
            y_trend = (df_clean['future_return_4h'] > 0.005).astype(int)
            y_momentum = (df_clean['future_return_1h'] > 0.002).astype(int)
            
            if len(X) < 20:
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.trend_model.fit(X_scaled, y_trend)
            self.momentum_model.fit(X_scaled, y_momentum)
            
            # Save model performance
            trend_score = self.trend_model.score(X_scaled, y_trend)
            momentum_score = self.momentum_model.score(X_scaled, y_momentum)
            
            self.save_ai_model_performance('trend_model', trend_score, len(X))
            self.save_ai_model_performance('momentum_model', momentum_score, len(X))
            
            return True
            
        except Exception as e:
            logger.error(f"AI model training failed: {e}")
            return False
    
    def get_ai_predictions(self, latest_data, available_features):
        """Get AI predictions with enhanced analysis"""
        try:
            # Prepare features
            features = []
            for col in available_features:
                value = latest_data.get(col, 50 if col == 'rsi' else 0)
                if pd.isna(value):
                    value = 50 if col == 'rsi' else 0
                features.append(float(value))
            
            if len(features) < 5:
                return self._get_default_predictions()
            
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
            ai_score = (trend_confidence + momentum_confidence) / 2
            
            return {
                'trend_prediction': trend_prediction,
                'trend_confidence': trend_confidence,
                'momentum_prediction': momentum_prediction,
                'momentum_confidence': momentum_confidence,
                'ai_score': ai_score
            }
            
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            return self._get_default_predictions()
    
    def _get_default_predictions(self):
        """Default AI predictions when models fail"""
        return {
            'trend_prediction': 'NEUTRAL',
            'trend_confidence': 50.0,
            'momentum_prediction': 'NEUTRAL',
            'momentum_confidence': 50.0,
            'ai_score': 50.0
        }
    
    def analyze_market_conditions(self, df):
        """Analyze current market conditions"""
        try:
            latest = df.iloc[-1]
            
            # Market regime analysis
            ema_9 = latest.get('ema_9', latest['close'])
            ema_21 = latest.get('ema_21', latest['close'])
            ema_50 = latest.get('ema_50', latest['close'])
            
            if ema_9 > ema_21 > ema_50:
                market_regime = 'TRENDING_UP'
            elif ema_9 < ema_21 < ema_50:
                market_regime = 'TRENDING_DOWN'
            else:
                market_regime = 'RANGING'
            
            # Volatility analysis
            volatility = df['close'].pct_change().tail(20).std() * 100
            if volatility > 5:
                volatility_regime = 'HIGH_VOLATILITY'
            elif volatility < 2:
                volatility_regime = 'LOW_VOLATILITY'
            else:
                volatility_regime = 'NORMAL_VOLATILITY'
            
            # Liquidity score
            volume_ratio = latest.get('volume_ratio', 1)
            liquidity_score = min(100, max(0, volume_ratio * 50))
            
            return {
                'market_regime': market_regime,
                'volatility_regime': volatility_regime,
                'liquidity_score': liquidity_score
            }
            
        except Exception as e:
            logger.error(f"Market condition analysis failed: {e}")
            return {
                'market_regime': 'UNKNOWN',
                'volatility_regime': 'NORMAL_VOLATILITY',
                'liquidity_score': 50.0
            }
    
    def generate_enhanced_signals(self):
        """Generate enhanced trading signals with AI integration"""
        if not self.exchange:
            logger.error("Exchange connection required for signal generation")
            return []
        
        logger.info("Generating AI-enhanced trading signals...")
        
        enhanced_signals = []
        ai_trained = False
        
        for symbol in self.trading_pairs[:10]:  # Process top 10 pairs
            try:
                # Get market data
                market_data = self.get_enhanced_market_data(symbol, '1h')
                if not market_data:
                    continue
                
                df, ticker = market_data
                df = self.calculate_enhanced_indicators(df)
                
                if len(df) < 50:
                    continue
                
                # Train AI models once
                if not ai_trained:
                    ai_trained = self.train_ai_models(df)
                
                latest = df.iloc[-1]
                current_price = ticker['last']
                
                # Enhanced signal analysis
                signal_data = self.analyze_enhanced_signal(df, latest, current_price, symbol)
                
                if signal_data and signal_data['confidence'] >= 60:
                    # Save enhanced signal
                    if self.save_enhanced_signal(signal_data):
                        enhanced_signals.append(signal_data)
                        logger.info(f"Enhanced signal: {symbol} {signal_data['signal']} ({signal_data['confidence']:.1f}%)")
                
            except Exception as e:
                logger.error(f"Enhanced signal generation failed for {symbol}: {e}")
                continue
        
        logger.info(f"Generated {len(enhanced_signals)} enhanced signals")
        return enhanced_signals
    
    def analyze_enhanced_signal(self, df, latest, current_price, symbol):
        """Analyze and generate enhanced signal with AI integration"""
        try:
            # Get available features
            feature_columns = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'adx', 'stoch_k', 'williams_r']
            available_features = [col for col in feature_columns if col in latest.index and not pd.isna(latest[col])]
            
            if len(available_features) < 5:
                return None
            
            # Get AI predictions
            latest_data = latest.to_dict()
            latest_data['current_price'] = current_price
            ai_predictions = self.get_ai_predictions(latest_data, available_features)
            
            # Analyze market conditions
            market_conditions = self.analyze_market_conditions(df)
            
            # Generate signal based on multiple factors
            signal_strength = 0
            signal_direction = 'HOLD'
            reasoning = []
            
            # RSI analysis
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                signal_strength += 25
                signal_direction = 'BUY'
                reasoning.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signal_strength += 25
                signal_direction = 'SELL'
                reasoning.append(f"RSI overbought ({rsi:.1f})")
            
            # MACD analysis
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            if macd > macd_signal:
                signal_strength += 15
                if signal_direction == 'HOLD':
                    signal_direction = 'BUY'
                reasoning.append("MACD bullish crossover")
            elif macd < macd_signal:
                signal_strength += 15
                if signal_direction == 'HOLD':
                    signal_direction = 'SELL'
                reasoning.append("MACD bearish crossover")
            
            # AI prediction boost
            ai_boost = (ai_predictions['ai_score'] - 50) * 0.3
            signal_strength += ai_boost
            
            if ai_predictions['trend_prediction'] == 'BULLISH':
                reasoning.append(f"AI trend bullish ({ai_predictions['trend_confidence']:.1f}%)")
            elif ai_predictions['trend_prediction'] == 'BEARISH':
                reasoning.append(f"AI trend bearish ({ai_predictions['trend_confidence']:.1f}%)")
            
            # Volume confirmation
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 1.2:
                signal_strength += 10
                reasoning.append(f"High volume ({volume_ratio:.1f}x)")
            
            # Market regime adjustment
            if market_conditions['market_regime'] == 'TRENDING_UP' and signal_direction == 'BUY':
                signal_strength += 5
                reasoning.append("Uptrend confirmation")
            elif market_conditions['market_regime'] == 'TRENDING_DOWN' and signal_direction == 'SELL':
                signal_strength += 5
                reasoning.append("Downtrend confirmation")
            
            # Ensure minimum confidence
            confidence = min(95, max(signal_strength, 0))
            
            if confidence >= 60:
                return {
                    'symbol': symbol,
                    'signal': signal_direction,
                    'confidence': confidence,
                    'ai_score': ai_predictions['ai_score'],
                    'scan_type': 'ai_enhanced',
                    
                    # Technical indicators
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'bb_position': latest.get('bb_position', 0.5),
                    'volume_ratio': volume_ratio,
                    'adx': latest.get('adx', 25),
                    'stoch_k': latest.get('stoch_k', 50),
                    'williams_r': latest.get('williams_r', -50),
                    
                    # AI predictions
                    'trend_prediction': ai_predictions['trend_prediction'],
                    'trend_confidence': ai_predictions['trend_confidence'],
                    'momentum_prediction': ai_predictions['momentum_prediction'],
                    'momentum_confidence': ai_predictions['momentum_confidence'],
                    
                    # Market analysis
                    'market_regime': market_conditions['market_regime'],
                    'volatility_regime': market_conditions['volatility_regime'],
                    'liquidity_score': market_conditions['liquidity_score'],
                    
                    'reasoning': ' | '.join(reasoning),
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced signal analysis failed: {e}")
            return None
    
    def save_enhanced_signal(self, signal_data):
        """Save enhanced signal to database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO ai_enhanced_signals (
                        symbol, signal, confidence, ai_score, scan_type,
                        rsi, macd, macd_signal, bb_position, volume_ratio, adx, stoch_k, williams_r,
                        trend_prediction, trend_confidence, momentum_prediction, momentum_confidence,
                        market_regime, volatility_regime, liquidity_score,
                        reasoning, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['symbol'], signal_data['signal'], signal_data['confidence'],
                    signal_data['ai_score'], signal_data['scan_type'],
                    signal_data['rsi'], signal_data['macd'], signal_data['macd_signal'],
                    signal_data['bb_position'], signal_data['volume_ratio'], signal_data['adx'],
                    signal_data['stoch_k'], signal_data['williams_r'],
                    signal_data['trend_prediction'], signal_data['trend_confidence'],
                    signal_data['momentum_prediction'], signal_data['momentum_confidence'],
                    signal_data['market_regime'], signal_data['volatility_regime'],
                    signal_data['liquidity_score'], signal_data['reasoning'], signal_data['timestamp']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Enhanced signal save failed: {e}")
            return False
    
    def save_ai_model_performance(self, model_type, accuracy, samples):
        """Save AI model performance metrics"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO ai_model_performance (
                        model_type, accuracy, samples_trained
                    ) VALUES (?, ?, ?)
                ''', (model_type, accuracy, samples))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"AI model performance save failed: {e}")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'ai_integration': 'ACTIVE',
                'exchange_connection': 'CONNECTED' if self.exchange else 'DISCONNECTED',
                'enhanced_signals': 0,
                'ai_models_trained': False
            }
            
            # Check recent signals
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT COUNT(*) FROM ai_enhanced_signals 
                    WHERE datetime(timestamp) >= datetime('now', '-1 hour')
                ''')
                recent_signals = cursor.fetchone()[0]
                status['enhanced_signals'] = recent_signals
                
                # Check AI model performance
                cursor.execute('''
                    SELECT COUNT(*) FROM ai_model_performance 
                    WHERE datetime(last_trained) >= datetime('now', '-24 hours')
                ''')
                recent_models = cursor.fetchone()[0]
                status['ai_models_trained'] = recent_models > 0
            
            return status
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {'error': str(e)}

def main():
    """Main AI integration function"""
    integration = AIEnhancedTradingIntegration()
    
    print("AI-Enhanced Trading System Integration")
    print("=====================================")
    
    # Generate enhanced signals
    signals = integration.generate_enhanced_signals()
    
    # Get system status
    status = integration.get_system_status()
    
    print(f"Enhanced signals generated: {len(signals)}")
    print(f"System status: {status}")
    
    return signals, status

if __name__ == "__main__":
    main()