#!/usr/bin/env python3
"""
Advanced System Enhancements
Next-level improvements for the AI trading system
"""

import sqlite3
import ccxt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSystemEnhancements:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.exchange = None
        self.initialize_exchange()
        self.setup_enhanced_features()
        
    def initialize_exchange(self):
        """Initialize OKX connection"""
        try:
            api_key = os.environ.get('OKX_API_KEY')
            secret_key = os.environ.get('OKX_SECRET_KEY')
            passphrase = os.environ.get('OKX_PASSPHRASE')
            
            if api_key and secret_key and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret_key,
                    'password': passphrase,
                    'sandbox': False,
                    'rateLimit': 800,
                    'enableRateLimit': True,
                })
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def setup_enhanced_features(self):
        """Setup enhanced feature databases"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Market regime detection table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    regime TEXT,
                    volatility REAL,
                    trend_strength REAL,
                    volume_profile TEXT,
                    correlation_cluster INTEGER
                )
            ''')
            
            # Advanced performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_performance (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    model_type TEXT,
                    symbol TEXT,
                    prediction_accuracy REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    avg_trade_duration REAL
                )
            ''')
            
            # Risk management alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    symbol TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    action_taken TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Multi-timeframe analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS multi_timeframe_signals (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    timeframe_1m TEXT,
                    timeframe_5m TEXT,
                    timeframe_15m TEXT,
                    timeframe_1h TEXT,
                    timeframe_4h TEXT,
                    timeframe_1d TEXT,
                    consensus_signal TEXT,
                    consensus_strength REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced feature databases initialized")
            
        except Exception as e:
            logger.error(f"Enhanced setup error: {e}")

    def implement_dynamic_stop_losses(self):
        """Implement ATR-based dynamic stop losses"""
        try:
            crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
            
            for symbol in crypto_symbols:
                # Get market data
                ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', limit=50)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate ATR (Average True Range)
                df['high_low'] = df['high'] - df['low']
                df['high_close'] = np.abs(df['high'] - df['close'].shift())
                df['low_close'] = np.abs(df['low'] - df['close'].shift())
                df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
                df['atr'] = df['tr'].rolling(window=14).mean()
                
                current_price = df['close'].iloc[-1]
                current_atr = df['atr'].iloc[-1]
                
                # Dynamic stop loss levels
                dynamic_stop_buy = current_price - (current_atr * 2.0)  # 2x ATR below
                dynamic_stop_sell = current_price + (current_atr * 2.0)  # 2x ATR above
                
                # Save to database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS dynamic_stops (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        symbol TEXT,
                        current_price REAL,
                        atr_value REAL,
                        stop_loss_buy REAL,
                        stop_loss_sell REAL,
                        volatility_adjusted BOOLEAN DEFAULT TRUE
                    )
                ''')
                
                cursor.execute('''
                    INSERT INTO dynamic_stops 
                    (timestamp, symbol, current_price, atr_value, stop_loss_buy, stop_loss_sell)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    symbol,
                    current_price,
                    current_atr,
                    dynamic_stop_buy,
                    dynamic_stop_sell
                ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"{symbol}: ATR=${current_atr:.2f}, Stop Buy=${dynamic_stop_buy:.2f}, Stop Sell=${dynamic_stop_sell:.2f}")
                
        except Exception as e:
            logger.error(f"Dynamic stop loss error: {e}")

    def implement_sentiment_analysis(self):
        """Implement market sentiment analysis"""
        try:
            # Fear & Greed Index simulation (in production, use real API)
            fear_greed_score = np.random.normal(50, 20)  # Simulate market sentiment
            fear_greed_score = max(0, min(100, fear_greed_score))
            
            if fear_greed_score < 25:
                sentiment = "Extreme Fear"
                signal_bias = "BUY"  # Contrarian signal
            elif fear_greed_score < 45:
                sentiment = "Fear"
                signal_bias = "BUY"
            elif fear_greed_score < 55:
                sentiment = "Neutral"
                signal_bias = "HOLD"
            elif fear_greed_score < 75:
                sentiment = "Greed"
                signal_bias = "SELL"
            else:
                sentiment = "Extreme Greed"
                signal_bias = "SELL"
            
            # Save sentiment analysis
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_sentiment (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    fear_greed_score REAL,
                    sentiment_label TEXT,
                    signal_bias TEXT,
                    volume_sentiment REAL,
                    news_sentiment REAL DEFAULT 50.0
                )
            ''')
            
            cursor.execute('''
                INSERT INTO market_sentiment 
                (timestamp, fear_greed_score, sentiment_label, signal_bias, volume_sentiment)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                fear_greed_score,
                sentiment,
                signal_bias,
                np.random.normal(50, 15)  # Volume sentiment
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Market Sentiment: {sentiment} ({fear_greed_score:.1f}/100) - Bias: {signal_bias}")
            return sentiment, signal_bias
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "Neutral", "HOLD"

    def implement_correlation_analysis(self):
        """Implement inter-asset correlation analysis"""
        try:
            crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
            correlation_matrix = {}
            
            # Get price data for all symbols
            price_data = {}
            for symbol in crypto_symbols:
                ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                price_data[symbol] = df['close'].pct_change().dropna()
                time.sleep(0.2)  # Rate limiting
            
            # Calculate correlation matrix
            correlation_df = pd.DataFrame(price_data)
            correlation_matrix = correlation_df.corr().to_dict()
            
            # Identify correlation clusters
            high_correlation_pairs = []
            for symbol1 in crypto_symbols:
                for symbol2 in crypto_symbols:
                    if symbol1 != symbol2:
                        corr = correlation_matrix[symbol1][symbol2]
                        if abs(corr) > 0.7:  # High correlation threshold
                            high_correlation_pairs.append((symbol1, symbol2, corr))
            
            # Save correlation analysis
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS correlation_analysis (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    symbol_pair TEXT,
                    correlation_coefficient REAL,
                    risk_level TEXT
                )
            ''')
            
            for symbol1, symbol2, corr in high_correlation_pairs:
                risk_level = "HIGH" if abs(corr) > 0.8 else "MEDIUM"
                cursor.execute('''
                    INSERT INTO correlation_analysis 
                    (timestamp, symbol_pair, correlation_coefficient, risk_level)
                    VALUES (?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    f"{symbol1}-{symbol2}",
                    corr,
                    risk_level
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Correlation Analysis: {len(high_correlation_pairs)} high correlation pairs identified")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return {}

    def implement_advanced_risk_management(self):
        """Implement sophisticated risk management"""
        try:
            # Get current portfolio
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT symbol, quantity, current_price FROM portfolio WHERE quantity > 0")
            positions = cursor.fetchall()
            
            total_portfolio_value = 0
            risk_alerts = []
            
            for symbol, quantity, price in positions:
                position_value = quantity * price
                total_portfolio_value += position_value
            
            # Risk checks
            for symbol, quantity, price in positions:
                position_value = quantity * price
                position_weight = (position_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
                
                # Position size risk
                if position_weight > 40:  # Single position > 40% of portfolio
                    risk_alerts.append({
                        'type': 'POSITION_SIZE',
                        'severity': 'HIGH',
                        'symbol': symbol,
                        'current_value': position_weight,
                        'threshold': 40,
                        'message': f'{symbol} represents {position_weight:.1f}% of portfolio'
                    })
                
                # Volatility risk
                if symbol in ['SOL', 'ADA', 'DOT', 'AVAX'] and position_weight > 15:
                    risk_alerts.append({
                        'type': 'VOLATILITY_RISK',
                        'severity': 'MEDIUM',
                        'symbol': symbol,
                        'current_value': position_weight,
                        'threshold': 15,
                        'message': f'High volatility asset {symbol} at {position_weight:.1f}%'
                    })
            
            # Save risk alerts
            for alert in risk_alerts:
                cursor.execute('''
                    INSERT INTO risk_alerts 
                    (timestamp, alert_type, severity, symbol, current_value, threshold_value, action_taken)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    alert['type'],
                    alert['severity'],
                    alert['symbol'],
                    alert['current_value'],
                    alert['threshold'],
                    'MONITORING'
                ))
            
            conn.commit()
            conn.close()
            
            if risk_alerts:
                logger.info(f"Risk Management: {len(risk_alerts)} alerts generated")
                for alert in risk_alerts:
                    logger.warning(f"{alert['severity']} {alert['type']}: {alert['message']}")
            
            return risk_alerts
            
        except Exception as e:
            logger.error(f"Risk management error: {e}")
            return []

    def implement_machine_learning_optimization(self):
        """Implement ML-based signal optimization"""
        try:
            # Get historical signal performance
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Join signals with trade results
            cursor.execute('''
                SELECT s.symbol, s.signal, s.confidence, s.reasoning, s.timestamp,
                       t.pnl, t.side, t.price
                FROM ai_signals s
                LEFT JOIN (
                    SELECT symbol, side, price, pnl, timestamp,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp) as rn
                    FROM live_trades 
                    WHERE timestamp > datetime('now', '-7 days')
                ) t ON s.symbol = REPLACE(t.symbol, '/USDT', '') 
                    AND s.signal = UPPER(t.side)
                    AND t.rn = 1
                WHERE s.timestamp > datetime('now', '-7 days')
                ORDER BY s.timestamp DESC
            ''')
            
            signal_performance = cursor.fetchall()
            conn.close()
            
            if len(signal_performance) >= 10:
                # Prepare training data
                features = []
                labels = []
                
                for row in signal_performance:
                    if row[5] is not None:  # Has PnL data
                        # Features: confidence, hour of day, day of week
                        timestamp = datetime.fromisoformat(row[4])
                        feature_vector = [
                            float(row[2]),  # confidence
                            timestamp.hour,  # hour of day
                            timestamp.weekday(),  # day of week
                            len(row[3])  # reasoning length
                        ]
                        features.append(feature_vector)
                        
                        # Label: profitable (1) or not (0)
                        labels.append(1 if float(row[5]) > 0 else 0)
                
                if len(features) >= 5:
                    # Train ML model for signal quality prediction
                    X = np.array(features)
                    y = np.array(labels)
                    
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    
                    # Calculate feature importance
                    feature_names = ['confidence', 'hour', 'day_of_week', 'reasoning_length']
                    importance = model.feature_importances_
                    
                    logger.info("ML Optimization - Feature Importance:")
                    for name, imp in zip(feature_names, importance):
                        logger.info(f"  {name}: {imp:.3f}")
                    
                    # Save model
                    model_path = 'signal_quality_model.pkl'
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    return model
            
            return None
            
        except Exception as e:
            logger.error(f"ML optimization error: {e}")
            return None

    def run_comprehensive_enhancements(self):
        """Run all enhancement modules"""
        logger.info("🚀 Running Advanced System Enhancements")
        
        # 1. Dynamic Stop Losses
        logger.info("📊 Implementing Dynamic Stop Losses...")
        self.implement_dynamic_stop_losses()
        
        # 2. Sentiment Analysis
        logger.info("🧠 Running Market Sentiment Analysis...")
        sentiment, bias = self.implement_sentiment_analysis()
        
        # 3. Correlation Analysis
        logger.info("🔗 Analyzing Asset Correlations...")
        correlations = self.implement_correlation_analysis()
        
        # 4. Advanced Risk Management
        logger.info("⚠️ Running Advanced Risk Management...")
        risk_alerts = self.implement_advanced_risk_management()
        
        # 5. ML Optimization
        logger.info("🤖 Optimizing with Machine Learning...")
        ml_model = self.implement_machine_learning_optimization()
        
        logger.info("✅ Advanced Enhancements Complete")
        
        return {
            'sentiment': sentiment,
            'bias': bias,
            'correlations': correlations,
            'risk_alerts': risk_alerts,
            'ml_model': ml_model is not None
        }

def run_system_enhancements():
    """Main function to run system enhancements"""
    enhancer = AdvancedSystemEnhancements()
    return enhancer.run_comprehensive_enhancements()

if __name__ == "__main__":
    run_system_enhancements()