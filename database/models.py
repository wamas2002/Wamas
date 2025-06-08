from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class TradingSignal(Base):
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD
    strength = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    strategy_name = Column(String(50), nullable=False)
    market_regime = Column(String(20))
    price_at_signal = Column(Float, nullable=False)
    indicators = Column(JSON)  # Store technical indicators
    created_at = Column(DateTime, default=datetime.utcnow)

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    commission = Column(Float, default=0.0)
    strategy_used = Column(String(50), nullable=False)
    signal_id = Column(Integer)  # Reference to trading_signals
    order_id = Column(String(100))  # External order ID from exchange
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED
    leverage = Column(Float, default=1.0)
    is_paper_trade = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Portfolio(Base):
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    daily_return = Column(Float)
    total_return = Column(Float)
    drawdown = Column(Float)
    benchmark_return = Column(Float)
    sharpe_ratio = Column(Float)
    volatility = Column(Float)
    max_drawdown = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float, default=0.0)
    position_type = Column(String(10), default='LONG')  # LONG, SHORT
    leverage = Column(Float, default=1.0)
    margin_used = Column(Float, default=0.0)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AIModel(Base):
    __tablename__ = 'ai_models'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    model_type = Column(String(30), nullable=False)  # LSTM, PROPHET, QLEARNING
    symbol = Column(String(20), nullable=False)
    model_state = Column(Text)  # Serialized model state
    training_data_hash = Column(String(64))  # Hash of training data
    training_date = Column(DateTime, nullable=False)
    performance_metrics = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    prediction_timestamp = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float)
    prediction_horizon = Column(Integer, nullable=False)  # Hours ahead
    confidence = Column(Float, nullable=False)
    model_version = Column(String(20))
    features_used = Column(JSON)
    accuracy_score = Column(Float)  # Updated when actual price is known
    created_at = Column(DateTime, default=datetime.utcnow)

class RiskMetrics(Base):
    __tablename__ = 'risk_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    portfolio_value = Column(Float, nullable=False)
    var_95 = Column(Float)  # Value at Risk 95%
    var_99 = Column(Float)  # Value at Risk 99%
    expected_shortfall_95 = Column(Float)
    expected_shortfall_99 = Column(Float)
    beta = Column(Float)
    correlation_with_benchmark = Column(Float)
    max_individual_position = Column(Float)
    leverage_ratio = Column(Float)
    concentration_risk = Column(Float)
    liquidity_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class StrategyPerformance(Base):
    __tablename__ = 'strategy_performance'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_return = Column(Float, nullable=False)
    annual_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    calmar_ratio = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
    avg_trade_return = Column(Float)
    best_trade = Column(Float)
    worst_trade = Column(Float)
    parameters_used = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_value = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    annual_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    commission_paid = Column(Float)
    parameters = Column(JSON)
    equity_curve = Column(JSON)  # Time series of portfolio values
    trade_details = Column(JSON)  # Detailed trade log
    created_at = Column(DateTime, default=datetime.utcnow)

class WalkForwardResult(Base):
    __tablename__ = 'walk_forward_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    in_sample_start = Column(DateTime, nullable=False)
    in_sample_end = Column(DateTime, nullable=False)
    out_sample_start = Column(DateTime, nullable=False)
    out_sample_end = Column(DateTime, nullable=False)
    optimal_parameters = Column(JSON, nullable=False)
    in_sample_return = Column(Float)
    out_sample_return = Column(Float)
    in_sample_sharpe = Column(Float)
    out_sample_sharpe = Column(Float)
    parameter_stability_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class AlertLog(Base):
    __tablename__ = 'alert_log'
    
    id = Column(Integer, primary_key=True)
    alert_type = Column(String(30), nullable=False)  # RISK_LIMIT, SIGNAL, TRADE_COMPLETE
    severity = Column(String(10), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    message = Column(Text, nullable=False)
    symbol = Column(String(20))
    details = Column(JSON)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemLog(Base):
    __tablename__ = 'system_log'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    log_level = Column(String(10), nullable=False)  # DEBUG, INFO, WARNING, ERROR
    component = Column(String(50), nullable=False)  # TRADING_ENGINE, AI_PREDICTOR, etc.
    message = Column(Text, nullable=False)
    details = Column(JSON)
    session_id = Column(String(50))
    user_action = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection and session management
class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Add connection pooling and error handling
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
        
    def drop_tables(self):
        """Drop all tables (use with caution)"""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_table_info(self):
        """Get information about all tables"""
        inspector = create_engine(self.database_url).connect()
        tables = Base.metadata.tables.keys()
        return list(tables)