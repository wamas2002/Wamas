import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import websocket
import threading
from config import Config
import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    """Handles market data retrieval from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(Config.get_api_headers())
        
        # Data cache
        self.data_cache = {}
        self.cache_timestamps = {}
        self.cache_duration = Config.DATA_PARAMS['cache_duration']
        
        # WebSocket connections
        self.ws_connections = {}
        self.real_time_data = {}
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
    def initialize(self) -> bool:
        """Initialize data handler and test connections"""
        try:
            print("Initializing data handler...")
            
            # Test Binance API connection
            test_url = f"{Config.BINANCE_API_URL}/ping"
            response = self.session.get(test_url, timeout=10)
            
            if response.status_code == 200:
                print("Binance API connection successful")
            else:
                print(f"Binance API connection failed: {response.status_code}")
                return False
            
            # Test CoinGecko API connection
            test_url = f"{Config.COINGECKO_API_URL}/ping"
            response = self.session.get(test_url, timeout=10)
            
            if response.status_code == 200:
                print("CoinGecko API connection successful")
            else:
                print("CoinGecko API connection failed, continuing without it")
            
            return True
            
        except Exception as e:
            print(f"Error initializing data handler: {e}")
            return False
    
    def _rate_limit_check(self, source: str):
        """Check and enforce rate limiting"""
        current_time = time.time()
        if source in self.last_request_time:
            time_diff = current_time - self.last_request_time[source]
            if time_diff < self.min_request_interval:
                time.sleep(self.min_request_interval - time_diff)
        
        self.last_request_time[source] = time.time()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = time.time() - self.cache_timestamps[cache_key]
        return cache_age < self.cache_duration
    
    def get_historical_data(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Get historical OHLCV data from Binance"""
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            
            # Check cache first
            if cache_key in self.data_cache and self._is_cache_valid(cache_key):
                return self.data_cache[cache_key].copy()
            
            # Rate limiting
            self._rate_limit_check('binance')
            
            # Convert interval format if needed
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h',
                '8h': '8h', '12h': '12h', '1d': '1d', '3d': '3d',
                '1w': '1w', '1M': '1M'
            }
            
            binance_interval = interval_map.get(interval, '1h')
            
            # Prepare request
            url = f"{Config.BINANCE_API_URL}/klines"
            params = {
                'symbol': symbol,
                'interval': binance_interval,
                'limit': min(limit, 1000)  # Binance limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"Binance API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if df.empty:
                print(f"No valid data after processing for {symbol}")
                return pd.DataFrame()
            
            # Cache the data
            self.data_cache[cache_key] = df.copy()
            self.cache_timestamps[cache_key] = time.time()
            
            print(f"Retrieved {len(df)} candles for {symbol} ({interval})")
            return df
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            # Rate limiting
            self._rate_limit_check('binance_price')
            
            url = f"{Config.BINANCE_API_URL}/ticker/price"
            params = {'symbol': symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                print(f"Error getting current price for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics"""
        try:
            # Rate limiting
            self._rate_limit_check('binance_ticker')
            
            url = f"{Config.BINANCE_API_URL}/ticker/24hr"
            params = {'symbol': symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': data['symbol'],
                    'price_change': float(data['priceChange']),
                    'price_change_percent': float(data['priceChangePercent']),
                    'high_price': float(data['highPrice']),
                    'low_price': float(data['lowPrice']),
                    'volume': float(data['volume']),
                    'count': int(data['count'])
                }
            else:
                print(f"Error getting 24hr ticker for {symbol}: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error getting 24hr ticker for {symbol}: {e}")
            return {}
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book data"""
        try:
            # Rate limiting
            self._rate_limit_check('binance_orderbook')
            
            url = f"{Config.BINANCE_API_URL}/depth"
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Convert to more usable format
                bids = [[float(price), float(qty)] for price, qty in data['bids']]
                asks = [[float(price), float(qty)] for price, qty in data['asks']]
                
                return {
                    'bids': bids,
                    'asks': asks,
                    'last_update_id': data['lastUpdateId']
                }
            else:
                print(f"Error getting order book for {symbol}: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"Error getting order book for {symbol}: {e}")
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
    
    def start_websocket_stream(self, symbol: str, callback=None):
        """Start WebSocket stream for real-time data"""
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    
                    # Process ticker data
                    if 'c' in data:  # Current price
                        self.real_time_data[symbol] = {
                            'price': float(data['c']),
                            'volume': float(data['v']),
                            'change': float(data['P']),
                            'timestamp': datetime.now()
                        }
                        
                        if callback:
                            callback(symbol, self.real_time_data[symbol])
                            
                except Exception as e:
                    print(f"Error processing WebSocket message: {e}")
            
            def on_error(ws, error):
                print(f"WebSocket error for {symbol}: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                print(f"WebSocket closed for {symbol}")
            
            def on_open(ws):
                print(f"WebSocket opened for {symbol}")
            
            # Create WebSocket URL
            stream_name = f"{symbol.lower()}@ticker"
            ws_url = f"{Config.BINANCE_WS_URL}/{stream_name}"
            
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Store connection
            self.ws_connections[symbol] = ws
            
            # Start in separate thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting WebSocket stream for {symbol}: {e}")
            return False
    
    def stop_websocket_stream(self, symbol: str):
        """Stop WebSocket stream for a symbol"""
        try:
            if symbol in self.ws_connections:
                self.ws_connections[symbol].close()
                del self.ws_connections[symbol]
                print(f"WebSocket stream stopped for {symbol}")
            
            if symbol in self.real_time_data:
                del self.real_time_data[symbol]
                
        except Exception as e:
            print(f"Error stopping WebSocket stream for {symbol}: {e}")
    
    def get_real_time_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest real-time data for a symbol"""
        return self.real_time_data.get(symbol)
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available for trading"""
        try:
            # Rate limiting
            self._rate_limit_check('binance_validate')
            
            url = f"{Config.BINANCE_API_URL}/exchangeInfo"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
                return symbol in symbols
            
            return False
            
        except Exception as e:
            print(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol (returns default fees as we can't access account)"""
        return {
            'maker_fee': 0.001,  # 0.1%
            'taker_fee': 0.001   # 0.1%
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Close all WebSocket connections
            for symbol in list(self.ws_connections.keys()):
                self.stop_websocket_stream(symbol)
            
            # Close session
            self.session.close()
            
            print("Data handler cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
