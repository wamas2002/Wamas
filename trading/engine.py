import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import threading
from .data_handler import DataHandler
from .risk_manager import RiskManager
from strategies.ensemble_strategy import EnsembleStrategy
from strategies.ml_strategy import MLStrategy
from ai.reinforcement import QLearningAgent
from ai.predictor import AIPredictor
from utils.logger import TradingLogger
from utils.performance import PerformanceTracker
from config import Config
from database.services import DatabaseService
import warnings
warnings.filterwarnings('ignore')

class TradingEngine:
    """Main trading engine coordinating all components"""
    
    def __init__(self):
        # Core components
        self.data_handler = DataHandler()
        self.risk_manager = RiskManager()
        self.logger = TradingLogger()
        self.performance_tracker = PerformanceTracker()
        
        # Database integration
        try:
            self.db_service = DatabaseService()
        except Exception as e:
            self.logger.log_error(f"Database connection failed: {e}")
            self.db_service = None
        
        # Strategies
        self.strategies = {
            'ensemble': EnsembleStrategy(),
            'ml': MLStrategy(),
        }
        
        # AI components
        self.ai_predictor = AIPredictor()
        self.rl_agent = QLearningAgent()
        
        # Trading state
        self.is_running = False
        self.current_positions = {}
        self.portfolio_value = 10000.0  # Starting portfolio value
        self.cash_balance = 10000.0
        self.paper_trading = True
        
        # Performance tracking with database persistence
        self.trade_history = []
        self.portfolio_history = []
        self.signals_history = []
        self.market_data = {}
        
        # Threading
        self.trading_thread = None
        self.data_update_thread = None
        
        # Current market data
        self.latest_prices = {}
        
    def initialize(self) -> bool:
        """Initialize the trading engine"""
        try:
            print("Initializing Trading Engine...")
            
            # Initialize data handler
            if not self.data_handler.initialize():
                print("Failed to initialize data handler")
                return False
            
            # Load initial market data for supported symbols
            for symbol in Config.SUPPORTED_SYMBOLS[:3]:  # Start with first 3 symbols
                try:
                    data = self.data_handler.get_historical_data(symbol, "1h", 500)
                    if not data.empty:
                        self.market_data[symbol] = data
                        self.latest_prices[symbol] = data['close'].iloc[-1]
                        print(f"Loaded data for {symbol}: {len(data)} candles")
                    else:
                        print(f"No data available for {symbol}")
                except Exception as e:
                    print(f"Error loading data for {symbol}: {e}")
            
            # Initialize risk manager
            self.risk_manager.initialize(self.portfolio_value)
            
            print("Trading Engine initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing trading engine: {e}")
            return False
    
    def start_trading(self, symbol: str, strategy_name: str, timeframe: str = "1h") -> bool:
        """Start automated trading"""
        try:
            if self.is_running:
                print("Trading already running")
                return False
            
            if symbol not in Config.SUPPORTED_SYMBOLS:
                print(f"Unsupported symbol: {symbol}")
                return False
            
            if strategy_name not in self.strategies:
                print(f"Unknown strategy: {strategy_name}")
                return False
            
            print(f"Starting trading for {symbol} using {strategy_name} strategy...")
            
            self.is_running = True
            
            # Start data update thread
            self.data_update_thread = threading.Thread(
                target=self._data_update_loop,
                args=(symbol, timeframe),
                daemon=True
            )
            self.data_update_thread.start()
            
            # Start trading thread
            self.trading_thread = threading.Thread(
                target=self._trading_loop,
                args=(symbol, strategy_name, timeframe),
                daemon=True
            )
            self.trading_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting trading: {e}")
            self.is_running = False
            return False
    
    def stop_trading(self):
        """Stop automated trading"""
        print("Stopping trading...")
        self.is_running = False
        
        # Wait for threads to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        if self.data_update_thread and self.data_update_thread.is_alive():
            self.data_update_thread.join(timeout=5)
        
        print("Trading stopped")
    
    def _data_update_loop(self, symbol: str, timeframe: str):
        """Background thread for updating market data with database persistence"""
        while self.is_running:
            try:
                # Update market data
                new_data = self.data_handler.get_historical_data(symbol, timeframe, 100)
                if not new_data.empty:
                    self.market_data[symbol] = new_data
                    self.latest_prices[symbol] = new_data['close'].iloc[-1]
                    
                    # Store market data in database
                    if self.db_service:
                        try:
                            self.db_service.store_market_data(symbol, timeframe, new_data.tail(10))
                        except Exception as db_error:
                            self.logger.log_error(f"Database storage failed: {db_error}")
                
                # Update portfolio value and store portfolio snapshot
                self._update_portfolio_value()
                if self.db_service:
                    try:
                        self.db_service.store_portfolio_snapshot(
                            total_value=self.portfolio_value,
                            cash_balance=self.cash_balance,
                            positions_value=self.portfolio_value - self.cash_balance
                        )
                    except Exception as db_error:
                        self.logger.log_error(f"Portfolio storage failed: {db_error}")
                
                time.sleep(Config.DATA_PARAMS['update_interval'])
                
            except Exception as e:
                self.logger.log_error(f"Error in data update loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _trading_loop(self, symbol: str, strategy_name: str, timeframe: str):
        """Main trading loop"""
        strategy = self.strategies[strategy_name]
        last_signal_time = None
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if we have sufficient data
                if symbol not in self.market_data or self.market_data[symbol].empty:
                    time.sleep(30)
                    continue
                
                data = self.market_data[symbol]
                
                # Train AI models periodically (every hour)
                if (last_signal_time is None or 
                    (current_time - last_signal_time).total_seconds() > 3600):
                    
                    print("Training AI models...")
                    self._train_ai_models(data)
                
                # Generate AI predictions
                ai_predictions = self.ai_predictor.predict(data)
                
                # Generate strategy signal
                if strategy_name == 'ensemble':
                    signal = strategy.generate_signal(data, ai_predictions)
                else:
                    signal = strategy.generate_signal(data)
                
                # Get RL agent action
                rl_state = self.rl_agent.get_state(data)
                rl_action = self.rl_agent.act(rl_state)
                rl_signal = self.rl_agent.get_action_name(rl_action)
                
                # Combine signals (ensemble approach)
                final_signal = self._combine_signals(signal, rl_signal, ai_predictions)
                
                # Execute trade if signal is strong enough
                if final_signal['confidence'] > 0.6 and final_signal['strength'] > 0.3:
                    self._execute_trade(symbol, final_signal)
                
                # Update RL agent
                if len(data) >= 2:
                    prev_price = data['close'].iloc[-2]
                    current_price = data['close'].iloc[-1]
                    reward = self.rl_agent.calculate_reward(
                        prev_price, current_price, rl_action,
                        self.current_positions.get(symbol, 0.0)
                    )
                    
                    # Learn from experience
                    if hasattr(self, '_prev_rl_state'):
                        self.rl_agent.learn(
                            self._prev_rl_state, self._prev_rl_action,
                            reward, rl_state
                        )
                    
                    self._prev_rl_state = rl_state
                    self._prev_rl_action = rl_action
                
                # Log signal and store in database
                self.logger.log_signal(symbol, final_signal)
                self.signals_history.append({
                    'timestamp': current_time,
                    'symbol': symbol,
                    'signal': final_signal,
                    'ai_predictions': ai_predictions,
                    'rl_action': rl_signal
                })
                
                # Store signal in database
                if self.db_service and symbol in self.latest_prices:
                    try:
                        self.db_service.store_trading_signal(
                            symbol=symbol,
                            signal_type=final_signal['action'],
                            strength=final_signal['strength'],
                            confidence=final_signal['confidence'],
                            strategy_name=strategy_name,
                            price=self.latest_prices[symbol],
                            market_regime=final_signal.get('market_regime', 'unknown'),
                            indicators=final_signal.get('indicators', {})
                        )
                    except Exception as db_error:
                        self.logger.log_error(f"Signal storage failed: {db_error}")
                
                last_signal_time = current_time
                
                # Wait before next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def _train_ai_models(self, data: pd.DataFrame):
        """Train AI models with latest data and store state in database"""
        try:
            # Train AI predictor
            training_result = self.ai_predictor.train_models(data)
            if 'error' not in training_result:
                print("AI models trained successfully")
                
                # Store AI model state in database
                if self.db_service:
                    try:
                        for model_name, result in training_result.items():
                            if isinstance(result, dict) and 'accuracy' in result:
                                self.db_service.store_ai_model_state(
                                    model_name=model_name,
                                    model_type='predictor',
                                    training_accuracy=result.get('accuracy', 0.0),
                                    model_parameters=result,
                                    data_size=len(data)
                                )
                    except Exception as db_error:
                        self.logger.log_error(f"AI model state storage failed: {db_error}")
            else:
                print(f"AI training error: {training_result['error']}")
            
            # Train ML strategy if available
            if 'ml' in self.strategies and len(data) > 200:
                ml_result = self.strategies['ml'].train_models(data)
                if 'error' not in ml_result:
                    print("ML strategy trained successfully")
                    
                    # Store ML strategy state in database
                    if self.db_service:
                        try:
                            self.db_service.store_ai_model_state(
                                model_name='ml_strategy',
                                model_type='strategy',
                                training_accuracy=ml_result.get('accuracy', 0.0),
                                model_parameters=ml_result,
                                data_size=len(data)
                            )
                        except Exception as db_error:
                            self.logger.log_error(f"ML strategy state storage failed: {db_error}")
                else:
                    print(f"ML training error: {ml_result['error']}")
            
        except Exception as e:
            print(f"Error training AI models: {e}")
            self.logger.log_error(f"AI training error: {e}")
    
    def _combine_signals(self, strategy_signal: Dict[str, Any], 
                        rl_signal: str, ai_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Combine different signals into final trading decision"""
        try:
            # Weight different signal sources
            weights = {
                'strategy': 0.5,
                'rl': 0.3,
                'ai_confidence': 0.2
            }
            
            # Convert signals to numeric scores
            signal_scores = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
            
            strategy_score = signal_scores.get(strategy_signal.get('signal', 'HOLD'), 0)
            rl_score = signal_scores.get(rl_signal, 0)
            
            # AI confidence boost
            ai_confidence = ai_predictions.get('confidence', 0.0) if ai_predictions else 0.0
            
            # Calculate weighted score
            combined_score = (
                strategy_score * strategy_signal.get('strength', 0) * weights['strategy'] +
                rl_score * weights['rl'] +
                ai_confidence * weights['ai_confidence']
            )
            
            # Determine final signal
            if combined_score > 0.3:
                final_signal = 'BUY'
            elif combined_score < -0.3:
                final_signal = 'SELL'
            else:
                final_signal = 'HOLD'
            
            # Calculate combined confidence
            base_confidence = strategy_signal.get('confidence', 0.5)
            combined_confidence = min(1.0, base_confidence + ai_confidence * 0.3)
            
            return {
                'signal': final_signal,
                'strength': abs(combined_score),
                'confidence': combined_confidence,
                'combined_score': combined_score,
                'components': {
                    'strategy': strategy_signal,
                    'rl': rl_signal,
                    'ai_confidence': ai_confidence
                }
            }
            
        except Exception as e:
            print(f"Error combining signals: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _execute_trade(self, symbol: str, signal: Dict[str, Any]):
        """Execute trade based on signal"""
        try:
            if symbol not in self.latest_prices:
                print(f"No price data for {symbol}")
                return
            
            current_price = self.latest_prices[symbol]
            signal_type = signal['signal']
            strength = signal['strength']
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                self.portfolio_value, current_price, strength
            )
            
            current_position = self.current_positions.get(symbol, 0.0)
            
            if signal_type == 'BUY' and current_position <= 0:
                # Buy signal
                trade_value = position_size * current_price
                
                if self.cash_balance >= trade_value:
                    if self.paper_trading:
                        # Paper trading execution
                        self.current_positions[symbol] = position_size
                        self.cash_balance -= trade_value
                        
                        trade_record = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': position_size,
                            'price': current_price,
                            'value': trade_value,
                            'signal_strength': strength,
                            'portfolio_value': self.portfolio_value
                        }
                        
                        self.trade_history.append(trade_record)
                        self.logger.log_trade(trade_record)
                        
                        # Store trade in database
                        if self.db_service:
                            try:
                                self.db_service.store_trade(
                                    symbol=symbol,
                                    trade_type='BUY',
                                    quantity=position_size,
                                    entry_price=current_price,
                                    strategy_used='paper_trading',
                                    is_paper_trade=True
                                )
                            except Exception as db_error:
                                self.logger.log_error(f"Trade storage failed: {db_error}")
                        
                        print(f"BUY {position_size:.6f} {symbol} at {current_price:.2f}")
                else:
                    print(f"Insufficient cash for BUY order: {self.cash_balance:.2f} < {trade_value:.2f}")
            
            elif signal_type == 'SELL' and current_position > 0:
                # Sell signal
                sell_quantity = min(current_position, position_size)
                trade_value = sell_quantity * current_price
                
                if self.paper_trading:
                    # Paper trading execution
                    self.current_positions[symbol] -= sell_quantity
                    self.cash_balance += trade_value
                    
                    trade_record = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': sell_quantity,
                        'price': current_price,
                        'value': trade_value,
                        'signal_strength': strength,
                        'portfolio_value': self.portfolio_value
                    }
                    
                    self.trade_history.append(trade_record)
                    self.logger.log_trade(trade_record)
                    
                    # Store trade in database
                    if self.db_service:
                        try:
                            self.db_service.store_trade(
                                symbol=symbol,
                                trade_type='SELL',
                                quantity=sell_quantity,
                                entry_price=current_price,
                                strategy_used='paper_trading',
                                is_paper_trade=True
                            )
                        except Exception as db_error:
                            self.logger.log_error(f"Trade storage failed: {db_error}")
                    
                    print(f"SELL {sell_quantity:.6f} {symbol} at {current_price:.2f}")
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            self.logger.log_error(f"Trade execution error: {e}")
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        try:
            total_value = self.cash_balance
            
            # Add value of current positions
            for symbol, quantity in self.current_positions.items():
                if symbol in self.latest_prices and quantity > 0:
                    total_value += quantity * self.latest_prices[symbol]
            
            self.portfolio_value = total_value
            
            # Record portfolio history
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': self.portfolio_value,
                'cash_balance': self.cash_balance,
                'positions': self.current_positions.copy()
            })
            
            # Keep only recent history
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
                
        except Exception as e:
            print(f"Error updating portfolio value: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            position_values = {}
            total_position_value = 0
            
            for symbol, quantity in self.current_positions.items():
                if symbol in self.latest_prices and quantity > 0:
                    value = quantity * self.latest_prices[symbol]
                    position_values[symbol] = {
                        'quantity': quantity,
                        'price': self.latest_prices[symbol],
                        'value': value
                    }
                    total_position_value += value
            
            return {
                'portfolio_value': self.portfolio_value,
                'cash_balance': self.cash_balance,
                'positions_value': total_position_value,
                'positions': position_values,
                'total_trades': len(self.trade_history),
                'is_running': self.is_running,
                'paper_trading': self.paper_trading
            }
            
        except Exception as e:
            print(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_recent_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades"""
        return self.trade_history[-count:] if self.trade_history else []
    
    def get_recent_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals"""
        return self.signals_history[-count:] if self.signals_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            if len(self.trade_history) < 2:
                return {'message': 'Insufficient trade history'}
            
            # Calculate returns
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1]['portfolio_value']
                curr_value = self.portfolio_history[i]['portfolio_value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if not returns:
                return {'message': 'No returns data available'}
            
            returns_array = np.array(returns)
            
            metrics = {
                'total_return': (self.portfolio_value - 10000) / 10000,
                'total_trades': len(self.trade_history),
                'avg_return': np.mean(returns_array),
                'volatility': np.std(returns_array),
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
            
            # Sharpe ratio
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['volatility'] * np.sqrt(252)
            
            # Win rate
            winning_trades = [t for t in self.trade_history 
                            if t.get('action') == 'SELL' and 
                            len([prev_t for prev_t in self.trade_history 
                                if prev_t.get('symbol') == t.get('symbol') and 
                                prev_t.get('action') == 'BUY' and 
                                prev_t.get('timestamp', datetime.min) < t.get('timestamp', datetime.max)]) > 0]
            
            if len(self.trade_history) > 0:
                metrics['win_rate'] = len(winning_trades) / len(self.trade_history)
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    def backtest_strategy(self, symbol: str, strategy_name: str, 
                         start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest for a strategy"""
        try:
            # Get historical data
            data = self.data_handler.get_historical_data(symbol, "1h", 1000)
            
            if data.empty:
                return {'error': 'No historical data available'}
            
            # Initialize backtest portfolio
            backtest_cash = 10000.0
            backtest_position = 0.0
            backtest_trades = []
            
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                return {'error': f'Strategy {strategy_name} not found'}
            
            # Train models on first 70% of data
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
            
            # Train AI models
            if hasattr(strategy, 'train_models'):
                strategy.train_models(train_data)
            
            # Run backtest on remaining data
            for i in range(50, len(test_data)):  # Start after minimum lookback
                current_data = data.iloc[:train_size + i]
                current_price = test_data.iloc[i]['close']
                
                # Generate signal
                signal = strategy.generate_signal(current_data)
                
                if signal['confidence'] > 0.6 and signal['strength'] > 0.3:
                    if signal['signal'] == 'BUY' and backtest_position == 0:
                        # Buy
                        shares = backtest_cash / current_price
                        backtest_position = shares
                        backtest_cash = 0
                        
                        backtest_trades.append({
                            'timestamp': test_data.index[i],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'portfolio_value': shares * current_price
                        })
                        
                    elif signal['signal'] == 'SELL' and backtest_position > 0:
                        # Sell
                        backtest_cash = backtest_position * current_price
                        
                        backtest_trades.append({
                            'timestamp': test_data.index[i],
                            'action': 'SELL',
                            'price': current_price,
                            'shares': backtest_position,
                            'portfolio_value': backtest_cash
                        })
                        
                        backtest_position = 0
            
            # Calculate final portfolio value
            final_value = backtest_cash + (backtest_position * test_data.iloc[-1]['close'])
            total_return = (final_value - 10000) / 10000
            
            return {
                'initial_value': 10000,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': len(backtest_trades),
                'trades': backtest_trades,
                'test_period': {
                    'start': test_data.index[0].strftime('%Y-%m-%d'),
                    'end': test_data.index[-1].strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            print(f"Error in backtest: {e}")
            return {'error': str(e)}
