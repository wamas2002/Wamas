import os
from typing import List, Dict, Any

class Config:
    """Configuration settings for the trading system"""
    
    # API Configuration
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
    
    # OKX API Configuration (if available)
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
    OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
    OKX_SANDBOX = os.getenv("OKX_SANDBOX", "true").lower() == "true"
    
    # Supported trading pairs
    SUPPORTED_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", 
        "DOTUSDT", "LINKUSDT", "LTCUSDT", "XRPUSDT"
    ]
    
    # Technical Analysis Parameters
    TA_PARAMS = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "sma_short": 20,
        "sma_long": 50,
        "ema_short": 12,
        "ema_long": 26
    }
    
    # AI Model Parameters
    AI_PARAMS = {
        "lstm_lookback": 60,
        "lstm_units": 50,
        "lstm_dropout": 0.2,
        "prophet_periods": 30,
        "ensemble_weights": {
            "lstm": 0.4,
            "prophet": 0.3,
            "transformer": 0.3
        }
    }
    
    # Q-Learning Parameters
    RL_PARAMS = {
        "epsilon": 0.1,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "learning_rate": 0.01,
        "discount_factor": 0.95,
        "memory_size": 10000
    }
    
    # Risk Management
    RISK_PARAMS = {
        "max_position_size": 0.1,  # 10% of portfolio
        "max_leverage": 1.0,       # No leverage by default
        "stop_loss_pct": 0.05,     # 5% stop loss
        "take_profit_pct": 0.10,   # 10% take profit
        "max_drawdown": 0.20       # 20% max drawdown
    }
    
    # Data collection settings
    DATA_PARAMS = {
        "max_candles": 1000,
        "update_interval": 60,     # seconds
        "cache_duration": 300      # 5 minutes
    }
    
    # Performance tracking
    PERFORMANCE_PARAMS = {
        "benchmark_symbol": "BTCUSDT",
        "risk_free_rate": 0.02,    # 2% annual
        "trading_fees": 0.001      # 0.1% per trade
    }

    @classmethod
    def get_api_headers(cls) -> Dict[str, str]:
        """Get API headers for requests"""
        return {
            "User-Agent": "AI-Crypto-Trading-System/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Check if required parameters are present
            required_params = [
                cls.SUPPORTED_SYMBOLS,
                cls.TA_PARAMS,
                cls.AI_PARAMS,
                cls.RL_PARAMS,
                cls.RISK_PARAMS
            ]
            
            for param in required_params:
                if not param:
                    return False
            
            return True
        except Exception:
            return False
