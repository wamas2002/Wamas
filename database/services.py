from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_, func
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import hashlib
from .models import (
    MarketData, TradingSignal, Trade, Portfolio, Position, AIModel,
    Prediction, RiskMetrics, StrategyPerformance, BacktestResult,
    WalkForwardResult, AlertLog, SystemLog, DatabaseManager
)

class DatabaseService:
    def safe_extract_price(self, data):
        """Safely extract price from any data type"""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, dict):
            return float(data.get('price', data.get('last', data.get('close', 0.0))))
        elif hasattr(data, 'price'):
            return float(data.price)
        elif hasattr(data, 'last'):
            return float(data.last)
        elif hasattr(data, 'close'):
            return float(data.close)
        else:
            return 0.0

    """Comprehensive database service for the trading system"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.db_manager.create_tables()
    
    def get_session(self):
        """Get database session"""
        return self.db_manager.get_session()
    
    # Market Data Operations
    def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store market data in database"""
        session = self.get_session()
        try:
            for _, row in data.iterrows():
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=row.name if hasattr(row.name, 'to_pydatetime') else datetime.now(),
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close']),
                    volume=float(row['volume']),
                    timeframe=timeframe
                )
                session.merge(market_data)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       start_date: datetime = None, end_date: datetime = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Retrieve market data from database"""
        session = self.get_session()
        try:
            query = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe
            )
            
            if start_date:
                query = query.filter(MarketData.timestamp >= start_date)
            if end_date:
                query = query.filter(MarketData.timestamp <= end_date)
            
            query = query.order_by(MarketData.timestamp).limit(limit)
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = []
            for row in results:
                data.append({
                    'timestamp': row.timestamp,
                    'open': row.open_price,
                    'high': row.high_price,
                    'low': row.low_price,
                    'close': row.close_price,
                    'volume': row.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        finally:
            session.close()
    
    # Trading Signal Operations
    def store_trading_signal(self, symbol: str, signal_type: str, strength: float,
                           confidence: float, strategy_name: str, price: float,
                           market_regime: str = None, indicators: Dict = None):
        """Store trading signal"""
        session = self.get_session()
        try:
            signal = TradingSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                strategy_name=strategy_name,
                market_regime=market_regime,
                price_at_signal=price,
                indicators=indicators or {}
            )
            session.add(signal)
            session.commit()
            return signal.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_recent_signals(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get recent trading signals"""
        session = self.get_session()
        try:
            query = session.query(TradingSignal)
            
            if symbol:
                query = query.filter(TradingSignal.symbol == symbol)
            
            signals = query.order_by(desc(TradingSignal.timestamp)).limit(limit).all()
            
            return [{
                'id': s.id,
                'symbol': s.symbol,
                'timestamp': s.timestamp,
                'signal_type': s.signal_type,
                'strength': s.strength,
                'confidence': s.confidence,
                'strategy_name': s.strategy_name,
                'market_regime': s.market_regime,
                'price_at_signal': s.price_at_signal,
                'indicators': s.indicators
            } for s in signals]
            
        finally:
            session.close()
    
    # Trade Operations
    def store_trade(self, symbol: str, trade_type: str, quantity: float, 
                   entry_price: float, strategy_used: str, signal_id: int = None,
                   order_id: str = None, leverage: float = 1.0, 
                   is_paper_trade: bool = True):
        """Store new trade"""
        session = self.get_session()
        try:
            trade = Trade(
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=datetime.now(),
                strategy_used=strategy_used,
                signal_id=signal_id,
                order_id=order_id,
                leverage=leverage,
                is_paper_trade=is_paper_trade
            )
            session.add(trade)
            session.commit()
            return trade.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def update_trade_exit(self, trade_id: int, exit_price: float, 
                         commission: float = 0.0):
        """Update trade with exit information"""
        session = self.get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.exit_time = datetime.now()
                trade.commission = commission
                trade.status = 'CLOSED'
                
                # Calculate PnL
                if trade.trade_type == 'BUY':
                    pnl = (exit_price - trade.entry_price) * trade.quantity - commission
                else:  # SELL
                    pnl = (trade.entry_price - exit_price) * trade.quantity - commission
                
                trade.pnl = pnl
                trade.pnl_percentage = (pnl / (trade.entry_price * trade.quantity)) * 100
                
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_trades(self, symbol: str = None, status: str = None, 
                  start_date: datetime = None, limit: int = 100) -> List[Dict]:
        """Get trades with optional filters"""
        session = self.get_session()
        try:
            query = session.query(Trade)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if status:
                query = query.filter(Trade.status == status)
            if start_date:
                query = query.filter(Trade.entry_time >= start_date)
            
            trades = query.order_by(desc(Trade.entry_time)).limit(limit).all()
            
            return [{
                'id': t.id,
                'symbol': t.symbol,
                'trade_type': t.trade_type,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl': t.pnl,
                'pnl_percentage': t.pnl_percentage,
                'commission': t.commission,
                'strategy_used': t.strategy_used,
                'status': t.status,
                'leverage': t.leverage,
                'is_paper_trade': t.is_paper_trade
            } for t in trades]
            
        finally:
            session.close()
    
    # Portfolio Operations
    def store_portfolio_snapshot(self, total_value: float, cash_balance: float,
                               positions_value: float, daily_return: float = None,
                               total_return: float = None, drawdown: float = None,
                               benchmark_return: float = None, sharpe_ratio: float = None,
                               volatility: float = None, max_drawdown: float = None):
        """Store portfolio snapshot"""
        session = self.get_session()
        try:
            portfolio = Portfolio(
                timestamp=datetime.now(),
                total_value=total_value,
                cash_balance=cash_balance,
                positions_value=positions_value,
                daily_return=daily_return,
                total_return=total_return,
                drawdown=drawdown,
                benchmark_return=benchmark_return,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                max_drawdown=max_drawdown
            )
            session.add(portfolio)
            session.commit()
            return portfolio.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history for specified days"""
        session = self.get_session()
        try:
            start_date = datetime.now() - timedelta(days=days)
            portfolios = session.query(Portfolio).filter(
                Portfolio.timestamp >= start_date
            ).order_by(Portfolio.timestamp).all()
            
            if not portfolios:
                return pd.DataFrame()
            
            data = []
            for p in portfolios:
                data.append({
                    'timestamp': p.timestamp,
                    'total_value': p.total_value,
                    'cash_balance': p.cash_balance,
                    'positions_value': p.positions_value,
                    'daily_return': p.daily_return,
                    'total_return': p.total_return,
                    'drawdown': p.drawdown,
                    'sharpe_ratio': p.sharpe_ratio,
                    'volatility': p.volatility
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        finally:
            session.close()
    
    # AI Model Operations
    def store_ai_model(self, model_name: str, model_type: str, symbol: str,
                      model_state: str, training_data_hash: str,
                      performance_metrics: Dict):
        """Store AI model state"""
        session = self.get_session()
        try:
            # Deactivate previous models of same type for symbol
            session.query(AIModel).filter(
                AIModel.model_type == model_type,
                AIModel.symbol == symbol,
                AIModel.is_active == True
            ).update({'is_active': False})
            
            ai_model = AIModel(
                model_name=model_name,
                model_type=model_type,
                symbol=symbol,
                model_state=model_state,
                training_data_hash=training_data_hash,
                training_date=datetime.now(),
                performance_metrics=performance_metrics,
                is_active=True
            )
            session.add(ai_model)
            session.commit()
            return ai_model.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_active_ai_model(self, model_type: str, symbol: str) -> Dict:
        """Get active AI model for symbol and type"""
        session = self.get_session()
        try:
            model = session.query(AIModel).filter(
                AIModel.model_type == model_type,
                AIModel.symbol == symbol,
                AIModel.is_active == True
            ).first()
            
            if model:
                return {
                    'id': model.id,
                    'model_name': model.model_name,
                    'model_type': model.model_type,
                    'symbol': model.symbol,
                    'model_state': model.model_state,
                    'training_date': model.training_date,
                    'performance_metrics': model.performance_metrics
                }
            return None
            
        finally:
            session.close()
    
    # Prediction Operations
    def store_prediction(self, model_id: int, symbol: str, predicted_price: float,
                        prediction_horizon: int, confidence: float,
                        model_version: str = None, features_used: Dict = None):
        """Store AI prediction"""
        session = self.get_session()
        try:
            prediction = Prediction(
                model_id=model_id,
                symbol=symbol,
                prediction_timestamp=datetime.now(),
                predicted_price=predicted_price,
                prediction_horizon=prediction_horizon,
                confidence=confidence,
                model_version=model_version,
                features_used=features_used or {}
            )
            session.add(prediction)
            session.commit()
            return prediction.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def update_prediction_accuracy(self, prediction_id: int, actual_price: float):
        """Update prediction with actual price for accuracy calculation"""
        session = self.get_session()
        try:
            prediction = session.query(Prediction).filter(
                Prediction.id == prediction_id
            ).first()
            
            if prediction:
                prediction.actual_price = actual_price
                # Calculate accuracy score (percentage error)
                error = abs(prediction.predicted_price - actual_price) / actual_price
                prediction.accuracy_score = max(0, 1 - error)  # 1 = perfect, 0 = worst
                session.commit()
                return True
            return False
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # Risk Metrics Operations
    def store_risk_metrics(self, portfolio_value: float, var_95: float = None,
                          var_99: float = None, expected_shortfall_95: float = None,
                          expected_shortfall_99: float = None, beta: float = None,
                          correlation_with_benchmark: float = None,
                          max_individual_position: float = None,
                          leverage_ratio: float = None,
                          concentration_risk: float = None,
                          liquidity_score: float = None):
        """Store risk metrics"""
        session = self.get_session()
        try:
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=expected_shortfall_95,
                expected_shortfall_99=expected_shortfall_99,
                beta=beta,
                correlation_with_benchmark=correlation_with_benchmark,
                max_individual_position=max_individual_position,
                leverage_ratio=leverage_ratio,
                concentration_risk=concentration_risk,
                liquidity_score=liquidity_score
            )
            session.add(risk_metrics)
            session.commit()
            return risk_metrics.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # Backtest Operations
    def store_backtest_result(self, strategy_name: str, symbol: str, timeframe: str,
                             start_date: datetime, end_date: datetime,
                             initial_capital: float, final_value: float,
                             total_return: float, annual_return: float = None,
                             volatility: float = None, sharpe_ratio: float = None,
                             max_drawdown: float = None, win_rate: float = None,
                             total_trades: int = None, commission_paid: float = None,
                             parameters: Dict = None, equity_curve: List = None,
                             trade_details: List = None):
        """Store backtest results"""
        session = self.get_session()
        try:
            backtest = BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_value=final_value,
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                commission_paid=commission_paid,
                parameters=parameters or {},
                equity_curve=equity_curve or [],
                trade_details=trade_details or []
            )
            session.add(backtest)
            session.commit()
            return backtest.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_backtest_results(self, strategy_name: str = None, symbol: str = None,
                           limit: int = 50) -> List[Dict]:
        """Get backtest results with optional filters"""
        session = self.get_session()
        try:
            query = session.query(BacktestResult)
            
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            
            results = query.order_by(desc(BacktestResult.created_at)).limit(limit).all()
            
            return [{
                'id': r.id,
                'strategy_name': r.strategy_name,
                'symbol': r.symbol,
                'timeframe': r.timeframe,
                'start_date': r.start_date,
                'end_date': r.end_date,
                'total_return': r.total_return,
                'annual_return': r.annual_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'total_trades': r.total_trades,
                'parameters': r.parameters,
                'created_at': r.created_at
            } for r in results]
            
        finally:
            session.close()
    
    # Utility Methods
    def get_trading_statistics(self, days: int = 30) -> Dict:
        """Get comprehensive trading statistics"""
        session = self.get_session()
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Total trades
            total_trades = session.query(Trade).filter(
                Trade.entry_time >= start_date
            ).count()
            
            # Winning trades
            winning_trades = session.query(Trade).filter(
                Trade.entry_time >= start_date,
                Trade.pnl > 0
            ).count()
            
            # Total PnL
            total_pnl = session.query(func.sum(Trade.pnl)).filter(
                Trade.entry_time >= start_date,
                Trade.pnl.isnot(None)
            ).scalar() or 0
            
            # Average PnL
            avg_pnl = session.query(func.avg(Trade.pnl)).filter(
                Trade.entry_time >= start_date,
                Trade.pnl.isnot(None)
            ).scalar() or 0
            
            # Total commission
            total_commission = session.query(func.sum(Trade.commission)).filter(
                Trade.entry_time >= start_date
            ).scalar() or 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': float(total_pnl),
                'average_pnl': float(avg_pnl),
                'total_commission': float(total_commission),
                'net_pnl': float(total_pnl - total_commission)
            }
            
        finally:
            session.close()
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to maintain database performance"""
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean old market data
            deleted_market_data = session.query(MarketData).filter(
                MarketData.created_at < cutoff_date
            ).delete()
            
            # Clean old system logs
            deleted_logs = session.query(SystemLog).filter(
                SystemLog.created_at < cutoff_date
            ).delete()
            
            # Clean old predictions (keep model performance data)
            deleted_predictions = session.query(Prediction).filter(
                Prediction.created_at < cutoff_date
            ).delete()
            
            session.commit()
            
            return {
                'deleted_market_data': deleted_market_data,
                'deleted_logs': deleted_logs,
                'deleted_predictions': deleted_predictions
            }
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for data to detect changes"""
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log_system_event(self, log_level: str, component: str, message: str,
                        details: Dict = None, session_id: str = None,
                        user_action: str = None):
        """Log system events"""
        session = self.get_session()
        try:
            log_entry = SystemLog(
                timestamp=datetime.now(),
                log_level=log_level,
                component=component,
                message=message,
                details=details or {},
                session_id=session_id,
                user_action=user_action
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()