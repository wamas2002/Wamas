"""
AI Model Evaluation & Auto-Switching System
Evaluates inactive models every 12h and switches if Sharpe Ratio improves ≥ 5%
"""

import os
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import ccxt
import pandas_ta as ta
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    model_name: str
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    profit_factor: float
    max_drawdown: float
    avg_return: float
    last_evaluated: datetime
    is_active: bool

class AIModelEvaluator:
    def __init__(self):
        self.config_file = "ai/active_model.json"
        self.db_path = "ai/model_evaluation.db"
        self.exchange = None
        self.ensure_directories()
        self.setup_database()
        self.initialize_exchange()
        
        # Available models for evaluation
        self.available_models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'lstm_basic': None,  # Placeholder for LSTM implementation
            'ensemble_hybrid': None  # Placeholder for ensemble model
        }
        
        # Load current active model
        self.active_model_config = self.load_active_model_config()
    
    def ensure_directories(self):
        """Ensure AI directory exists"""
        os.makedirs("ai", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def initialize_exchange(self):
        """Initialize OKX exchange for backtesting"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True
            })
            logger.info("Model evaluator connected to OKX")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_database(self):
        """Setup model evaluation database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sharpe_ratio REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    avg_return REAL NOT NULL,
                    backtest_period_days INTEGER NOT NULL,
                    is_active BOOLEAN DEFAULT FALSE,
                    performance_data TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_switches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    switch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    from_model TEXT NOT NULL,
                    to_model TEXT NOT NULL,
                    reason TEXT,
                    performance_improvement REAL,
                    sharpe_improvement REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Model evaluation database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def load_active_model_config(self) -> Dict:
        """Load current active model configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                default_config = {
                    'active_model': 'random_forest',
                    'last_switch': datetime.now().isoformat(),
                    'last_evaluation': None,
                    'evaluation_interval_hours': 12,
                    'min_improvement_threshold': 5.0  # 5% Sharpe ratio improvement
                }
                self.save_active_model_config(default_config)
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            return {}
    
    def save_active_model_config(self, config: Dict):
        """Save active model configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model config: {e}")
    
    def should_evaluate_models(self) -> bool:
        """Check if it's time to evaluate models"""
        try:
            last_eval = self.active_model_config.get('last_evaluation')
            if not last_eval:
                return True
            
            last_eval_time = datetime.fromisoformat(last_eval)
            interval_hours = self.active_model_config.get('evaluation_interval_hours', 12)
            
            return datetime.now() - last_eval_time >= timedelta(hours=interval_hours)
            
        except Exception as e:
            logger.error(f"Error checking evaluation schedule: {e}")
            return False
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data for backtesting"""
        try:
            if not self.exchange:
                return None
            
            # Calculate required candles (assuming 1h timeframe)
            limit = days * 24
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=limit)
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for model evaluation"""
        try:
            # Basic indicators
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df['macd'] = macd.iloc[:, 0] if len(macd.columns) > 0 else 0
                df['macd_signal'] = macd.iloc[:, 1] if len(macd.columns) > 1 else 0
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            if bb is not None and not bb.empty:
                df['bb_upper'] = bb.iloc[:, 0] if len(bb.columns) > 0 else df['close']
                df['bb_lower'] = bb.iloc[:, 2] if len(bb.columns) > 2 else df['close']
            
            # Volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Volume
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['price_change'] = df['close'].pct_change()
            df['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
            
            return df.fillna(0)
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for model training"""
        try:
            # Feature columns
            features = [
                'rsi', 'macd', 'macd_signal', 'volume_ratio', 'momentum',
                'price_change'
            ]
            
            # Ensure all feature columns exist
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Create target (future return)
            df['future_return'] = df['close'].shift(-5) / df['close'] - 1
            df['target'] = np.where(df['future_return'] > 0.02, 1,  # Buy signal
                                  np.where(df['future_return'] < -0.02, -1, 0))  # Sell signal, Hold
            
            # Prepare data
            X = df[features].fillna(0).iloc[:-5]
            y = df['target'].fillna(0).iloc[:-5]
            
            # Remove invalid data
            valid_mask = ~np.isnan(y) & ~np.isinf(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            return X.values, y.values
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return np.array([]), np.array([])
    
    def evaluate_model(self, model_name: str, symbols: List[str], days: int = 30) -> Optional[ModelPerformance]:
        """Evaluate a single model's performance"""
        try:
            if model_name not in self.available_models:
                logger.warning(f"Model {model_name} not available")
                return None
            
            model_class = self.available_models[model_name]
            if model_class is None:
                logger.warning(f"Model {model_name} not implemented yet")
                return None
            
            all_predictions = []
            all_returns = []
            total_trades = 0
            
            for symbol in symbols:
                df = self.get_historical_data(symbol, days)
                if df is None or len(df) < 50:
                    continue
                
                X, y = self.prepare_training_data(df)
                if len(X) == 0:
                    continue
                
                # Split data for training and testing
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                if len(X_train) < 10 or len(X_test) < 5:
                    continue
                
                # Train model
                model = model_class(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Calculate returns for each prediction
                test_returns = df['future_return'].iloc[split_idx:split_idx+len(predictions)]
                
                for pred, ret in zip(predictions, test_returns):
                    if abs(pred) > 0:  # Only count actual signals
                        all_predictions.append(pred)
                        # Simulate trade return based on prediction
                        trade_return = ret * pred if pred != 0 else 0
                        all_returns.append(trade_return)
                        total_trades += 1
            
            if len(all_returns) == 0:
                return None
            
            # Calculate performance metrics
            returns_series = pd.Series(all_returns)
            
            # Sharpe ratio
            if returns_series.std() > 0:
                sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Win rate
            win_rate = (returns_series > 0).mean() * 100
            
            # Profit factor
            gross_profit = returns_series[returns_series > 0].sum()
            gross_loss = abs(returns_series[returns_series < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Max drawdown
            cumulative_returns = (1 + returns_series).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) * 100
            
            # Average return
            avg_return = returns_series.mean() * 100
            
            performance = ModelPerformance(
                model_name=model_name,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                total_trades=total_trades,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                avg_return=avg_return,
                last_evaluated=datetime.now(),
                is_active=(model_name == self.active_model_config.get('active_model'))
            )
            
            # Save performance to database
            self.save_model_performance(performance, days)
            
            return performance
            
        except Exception as e:
            logger.error(f"Model evaluation failed for {model_name}: {e}")
            return None
    
    def save_model_performance(self, performance: ModelPerformance, backtest_days: int):
        """Save model performance to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance 
                (model_name, sharpe_ratio, win_rate, total_trades, profit_factor,
                 max_drawdown, avg_return, backtest_period_days, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance.model_name,
                performance.sharpe_ratio,
                performance.win_rate,
                performance.total_trades,
                performance.profit_factor,
                performance.max_drawdown,
                performance.avg_return,
                backtest_days,
                performance.is_active
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save model performance: {e}")
    
    def evaluate_all_models(self) -> Dict[str, ModelPerformance]:
        """Evaluate all available models"""
        logger.info("Starting comprehensive model evaluation")
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
        performances = {}
        
        for model_name in self.available_models.keys():
            if self.available_models[model_name] is not None:  # Skip unimplemented models
                performance = self.evaluate_model(model_name, symbols, days=30)
                if performance:
                    performances[model_name] = performance
                    logger.info(f"Evaluated {model_name}: Sharpe={performance.sharpe_ratio:.3f}, "
                              f"Win Rate={performance.win_rate:.1f}%")
        
        # Update last evaluation time
        self.active_model_config['last_evaluation'] = datetime.now().isoformat()
        self.save_active_model_config(self.active_model_config)
        
        return performances
    
    def should_switch_model(self, performances: Dict[str, ModelPerformance]) -> Optional[str]:
        """Determine if model should be switched based on performance"""
        try:
            current_model = self.active_model_config.get('active_model')
            if current_model not in performances:
                return None
            
            current_performance = performances[current_model]
            min_improvement = self.active_model_config.get('min_improvement_threshold', 5.0)
            
            best_model = None
            best_sharpe = current_performance.sharpe_ratio
            best_improvement = 0
            
            for model_name, performance in performances.items():
                if model_name == current_model:
                    continue
                
                # Calculate improvement percentage
                if current_performance.sharpe_ratio != 0:
                    improvement = ((performance.sharpe_ratio - current_performance.sharpe_ratio) / 
                                 abs(current_performance.sharpe_ratio)) * 100
                else:
                    improvement = performance.sharpe_ratio * 100
                
                # Check if improvement meets threshold and other criteria
                if (improvement >= min_improvement and 
                    performance.sharpe_ratio > best_sharpe and
                    performance.total_trades >= 10 and  # Minimum trades for reliability
                    performance.win_rate >= 45):  # Minimum win rate
                    
                    best_model = model_name
                    best_sharpe = performance.sharpe_ratio
                    best_improvement = improvement
            
            if best_model:
                logger.info(f"Model switch recommended: {current_model} -> {best_model} "
                          f"(Improvement: {best_improvement:.1f}%)")
                return best_model
            
            return None
            
        except Exception as e:
            logger.error(f"Error determining model switch: {e}")
            return None
    
    def switch_active_model(self, new_model: str, reason: str, improvement: float):
        """Switch to a new active model"""
        try:
            old_model = self.active_model_config.get('active_model', 'unknown')
            
            # Update active model config
            self.active_model_config['active_model'] = new_model
            self.active_model_config['last_switch'] = datetime.now().isoformat()
            self.save_active_model_config(self.active_model_config)
            
            # Log the switch
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_switches 
                (from_model, to_model, reason, performance_improvement, sharpe_improvement)
                VALUES (?, ?, ?, ?, ?)
            ''', (old_model, new_model, reason, improvement, improvement))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Model switched from {old_model} to {new_model}: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
    
    def run_evaluation_cycle(self):
        """Run complete model evaluation cycle"""
        try:
            if not self.should_evaluate_models():
                logger.info("Model evaluation not due yet")
                return
            
            logger.info("Starting 12-hour model evaluation cycle")
            
            # Evaluate all models
            performances = self.evaluate_all_models()
            
            if not performances:
                logger.warning("No model performances available")
                return
            
            # Check if model switch is needed
            recommended_model = self.should_switch_model(performances)
            
            if recommended_model:
                current_model = self.active_model_config.get('active_model')
                current_sharpe = performances[current_model].sharpe_ratio
                new_sharpe = performances[recommended_model].sharpe_ratio
                
                improvement = ((new_sharpe - current_sharpe) / abs(current_sharpe)) * 100
                
                self.switch_active_model(
                    recommended_model,
                    f"Sharpe ratio improvement: {improvement:.1f}%",
                    improvement
                )
            else:
                logger.info("No model switch needed - current model performing optimally")
            
            # Generate performance summary
            self.generate_evaluation_report(performances)
            
        except Exception as e:
            logger.error(f"Model evaluation cycle failed: {e}")
    
    def generate_evaluation_report(self, performances: Dict[str, ModelPerformance]):
        """Generate evaluation report"""
        try:
            report = {
                'evaluation_date': datetime.now().isoformat(),
                'active_model': self.active_model_config.get('active_model'),
                'model_performances': {}
            }
            
            for model_name, performance in performances.items():
                report['model_performances'][model_name] = {
                    'sharpe_ratio': performance.sharpe_ratio,
                    'win_rate': performance.win_rate,
                    'total_trades': performance.total_trades,
                    'profit_factor': performance.profit_factor,
                    'max_drawdown': performance.max_drawdown,
                    'avg_return': performance.avg_return,
                    'is_active': performance.is_active
                }
            
            # Save report
            report_file = f"logs/model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Evaluation report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
    
    def get_current_active_model(self) -> str:
        """Get currently active model name"""
        return self.active_model_config.get('active_model', 'random_forest')
    
    def get_model_performance_history(self, model_name: str = None) -> List[Dict]:
        """Get historical performance data for models"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if model_name:
                query = '''
                    SELECT * FROM model_performance 
                    WHERE model_name = ? 
                    ORDER BY evaluation_date DESC 
                    LIMIT 10
                '''
                df = pd.read_sql_query(query, conn, params=[model_name])
            else:
                query = '''
                    SELECT * FROM model_performance 
                    ORDER BY evaluation_date DESC 
                    LIMIT 50
                '''
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df.to_dict('records') if not df.empty else []
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []

# Global instance
model_evaluator = AIModelEvaluator()