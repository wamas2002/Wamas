import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
from trading.okx_connector import OKXConnector
from config import Config
import os

warnings.filterwarnings('ignore')

class OKXDataService:
    """Unified data service using OKX API as primary source"""
    
    def __init__(self):
        # Initialize OKX connector with environment credentials
        self.okx_connector = OKXConnector(
            api_key=os.getenv('OKX_API_KEY', ''),
            secret_key=os.getenv('OKX_SECRET_KEY', ''),
            passphrase=os.getenv('OKX_PASSPHRASE', ''),
            sandbox=False
        )
        
        # Disable caching to force live data
        self.data_cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 0  # No caching - always fetch fresh data
        
        # Symbol mapping (Binance format -> OKX format)
        self.symbol_map = {
            'BTCUSDT': 'BTC-USDT-SWAP',
            'ETHUSDT': 'ETH-USDT-SWAP',
            'ADAUSDT': 'ADA-USDT-SWAP',
            'BNBUSDT': 'BNB-USDT-SWAP',
            'DOTUSDT': 'DOT-USDT-SWAP',
            'LINKUSDT': 'LINK-USDT-SWAP',
            'LTCUSDT': 'LTC-USDT-SWAP',
            'XRPUSDT': 'XRP-USDT-SWAP'
        }
        
        # Interval mapping (Binance -> OKX)
        self.interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol format to OKX format"""
        return self.symbol_map.get(symbol, symbol)
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval format to OKX format"""
        return self.interval_map.get(interval, interval)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = time.time() - self.cache_timestamps[cache_key]
        return cache_age < self.cache_duration
    
    def get_historical_data(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Get historical OHLCV data from OKX"""
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            
            # Check cache first
            if cache_key in self.data_cache and self._is_cache_valid(cache_key):
                return self.data_cache[cache_key].copy()
            
            # Convert symbol and interval to OKX format
            okx_symbol = self._convert_symbol(symbol)
            okx_interval = self._convert_interval(interval)
            
            # Get data from OKX
            result = self.okx_connector.get_historical_data(okx_symbol, okx_interval, limit)
            
            if 'error' in result:
                print(f"OKX API error for {symbol}: {result['error']}")
                return pd.DataFrame()
            
            if 'data' not in result:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Extract the DataFrame from the result
            df = result['data']
            
            # Validate that we got a proper DataFrame
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"Invalid or empty data for {symbol}")
                return pd.DataFrame()
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Cache the data
            self.data_cache[cache_key] = df.copy()
            self.cache_timestamps[cache_key] = time.time()
            
            return df
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            okx_symbol = self._convert_symbol(symbol)
            result = self.okx_connector.get_ticker(okx_symbol)
            
            if 'error' in result:
                print(f"Error getting price for {symbol}: {result['error']}")
                return 0.0
            
            return result.get('price', 0.0)
            
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics"""
        try:
            okx_symbol = self._convert_symbol(symbol)
            result = self.okx_connector.get_ticker(okx_symbol)
            
            if 'error' in result:
                return {}
            
            # Convert to expected format
            return {
                'symbol': symbol,
                'price_change': result.get('change_24h', 0),
                'price_change_percent': result.get('change_24h', 0),
                'weighted_avg_price': result.get('price', 0),
                'prev_close_price': result.get('price', 0),
                'last_price': result.get('price', 0),
                'bid_price': result.get('bid', 0),
                'ask_price': result.get('ask', 0),
                'open_price': result.get('price', 0),
                'high_price': result.get('price', 0),
                'low_price': result.get('price', 0),
                'volume': result.get('volume', 0),
                'count': 0
            }
            
        except Exception as e:
            print(f"Error getting 24hr ticker for {symbol}: {e}")
            return {}
    
    def get_market_data_summary(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data summary for multiple symbols"""
        try:
            summary = {}
            
            for symbol in symbols:
                try:
                    # Get current price and 24hr stats
                    ticker_data = self.get_24hr_ticker(symbol)
                    current_price = self.get_current_price(symbol)
                    
                    if ticker_data and current_price:
                        summary[symbol] = {
                            'current_price': current_price,
                            'price_change_24h': ticker_data.get('price_change', 0),
                            'price_change_percent_24h': ticker_data.get('price_change_percent', 0),
                            'high_24h': ticker_data.get('high_price', current_price),
                            'low_24h': ticker_data.get('low_price', current_price),
                            'volume_24h': ticker_data.get('volume', 0),
                            'last_updated': datetime.now().isoformat()
                        }
                    
                    # Small delay between requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error getting summary for {symbol}: {e}")
                    continue
            
            return summary
            
        except Exception as e:
            print(f"Error getting market data summary: {e}")
            return {}
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book data"""
        try:
            okx_symbol = self._convert_symbol(symbol)
            # OKX doesn't have direct orderbook in our connector, use ticker for now
            ticker = self.get_24hr_ticker(symbol)
            
            if not ticker:
                return {}
            
            # Simplified orderbook structure
            return {
                'symbol': symbol,
                'bids': [[ticker.get('bid_price', 0), 1.0]],
                'asks': [[ticker.get('ask_price', 0), 1.0]]
            }
            
        except Exception as e:
            print(f"Error getting orderbook for {symbol}: {e}")
            return {}
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return list(self.symbol_map.keys())
    
    def test_connection(self) -> bool:
        """Test connection to OKX API"""
        try:
            result = self.okx_connector.test_connection()
            return result.get('success', False)
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance from OKX"""
        try:
            return self.okx_connector.get_account_balance()
        except Exception as e:
            print(f"Error getting account balance: {e}")
            return {'error': str(e)}
    
    def place_order(self, symbol: str, side: str, amount: float, 
                   price: float = None, order_type: str = 'market', 
                   leverage: int = 1) -> Dict[str, Any]:
        """Place trading order via OKX"""
        try:
            okx_symbol = self._convert_symbol(symbol)
            return self.okx_connector.place_futures_order(
                symbol=okx_symbol,
                side=side,
                size=amount,
                order_type=order_type,
                price=price,
                leverage=leverage
            )
        except Exception as e:
            print(f"Error placing order: {e}")
            return {'error': str(e)}
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        try:
            return self.okx_connector.get_positions()
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {'error': str(e)}
    
    def close_position(self, symbol: str, size: float = None) -> Dict[str, Any]:
        """Close position"""
        try:
            okx_symbol = self._convert_symbol(symbol)
            return self.okx_connector.close_position(okx_symbol, size)
        except Exception as e:
            print(f"Error closing position: {e}")
            return {'error': str(e)}