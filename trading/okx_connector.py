import hmac
import hashlib
import base64
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from urllib.parse import urlencode

class OKXConnector:
    """OKX API connector for spot and futures trading"""
    
    def __init__(self, api_key: str = "", secret_key: str = "", passphrase: str = "", 
                 sandbox: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.sandbox = sandbox
        
        # API endpoints
        if sandbox:
            self.base_url = "https://www.okx.com"  # Sandbox URL
        else:
            self.base_url = "https://www.okx.com"  # Production URL
            
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate API signature"""
        try:
            if not self.secret_key:
                return ""
                
            # Create the prehash string
            prehash = timestamp + method.upper() + request_path + body
            
            # Create signature
            signature = base64.b64encode(
                hmac.new(
                    self.secret_key.encode('utf-8'),
                    prehash.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            return signature
            
        except Exception as e:
            print(f"Error generating signature: {e}")
            return ""
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict[str, Any]:
        """Make authenticated API request"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            # Prepare request
            url = f"{self.base_url}{endpoint}"
            timestamp = str(int(time.time() * 1000))
            
            # Handle query parameters
            if params:
                query_string = urlencode(params)
                endpoint_with_params = f"{endpoint}?{query_string}"
                url = f"{self.base_url}{endpoint_with_params}"
            else:
                endpoint_with_params = endpoint
            
            # Prepare body
            body = ""
            if data:
                body = json.dumps(data)
            
            # Generate signature
            signature = self._generate_signature(timestamp, method, endpoint_with_params, body)
            
            # Set headers
            headers = {
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': signature,
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            }
            
            # Make request
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, data=body, timeout=30)
            else:
                return {'error': f'Unsupported method: {method}'}
            
            self.last_request_time = time.time()
            
            # Parse response
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f'HTTP {response.status_code}',
                    'message': response.text
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
                
            result = self._make_request('GET', '/api/v5/account/balance')
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            balances = {}
            for balance_info in result.get('data', []):
                for detail in balance_info.get('details', []):
                    currency = detail.get('ccy', '')
                    available = float(detail.get('availBal', '0'))
                    total = float(detail.get('bal', '0'))
                    
                    if total > 0:
                        balances[currency] = {
                            'available': available,
                            'total': total,
                            'frozen': total - available
                        }
            
            return {'success': True, 'balances': balances}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a symbol"""
        try:
            # Convert symbol format (BTCUSDT -> BTC-USDT)
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    symbol = f"{base}-USDT"
                elif symbol.endswith('USD'):
                    base = symbol[:-3]
                    symbol = f"{base}-USD"
            
            params = {'instId': symbol}
            result = self._make_request('GET', '/api/v5/market/ticker', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            data = result.get('data', [])
            if not data:
                return {'error': 'No ticker data found'}
            
            ticker = data[0]
            return {
                'success': True,
                'symbol': ticker.get('instId'),
                'price': float(ticker.get('last', '0')),
                'bid': float(ticker.get('bidPx', '0')),
                'ask': float(ticker.get('askPx', '0')),
                'volume': float(ticker.get('vol24h', '0')),
                'change_24h': float(ticker.get('chg24h', '0')),
                'timestamp': int(ticker.get('ts', '0'))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_historical_data(self, symbol: str, interval: str = '1H', limit: int = 100) -> Dict[str, Any]:
        """Get historical OHLCV data"""
        try:
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    symbol = f"{base}-USDT"
            
            # Convert interval format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1H', '1H': '1H', '4h': '4H', '1d': '1D', '1D': '1D'
            }
            okx_interval = interval_map.get(interval, '1H')
            
            params = {
                'instId': symbol,
                'bar': okx_interval,
                'limit': str(limit)
            }
            
            result = self._make_request('GET', '/api/v5/market/candles', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            data = result.get('data', [])
            if not data:
                return {'error': 'No historical data found'}
            
            # Convert to DataFrame format
            df_data = []
            for candle in reversed(data):  # OKX returns newest first
                df_data.append({
                    'timestamp': pd.to_datetime(int(candle[0]), unit='ms'),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            df = pd.DataFrame(df_data)
            
            return {
                'success': True,
                'symbol': symbol,
                'data': df,
                'count': len(df)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def place_order(self, symbol: str, side: str, amount: float, price: float = None, 
                   order_type: str = 'market', leverage: int = 1) -> Dict[str, Any]:
        """Place a trading order"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    symbol = f"{base}-USDT"
            
            # Prepare order data
            order_data = {
                'instId': symbol,
                'tdMode': 'cross' if leverage > 1 else 'cash',
                'side': side.lower(),
                'ordType': order_type.lower(),
                'sz': str(amount)
            }
            
            if price and order_type.lower() == 'limit':
                order_data['px'] = str(price)
            
            if leverage > 1:
                order_data['lever'] = str(leverage)
            
            result = self._make_request('POST', '/api/v5/trade/order', data=order_data)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            order_result = result.get('data', [])
            if order_result:
                order = order_result[0]
                return {
                    'success': True,
                    'order_id': order.get('ordId'),
                    'client_order_id': order.get('clOrdId'),
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'order_type': order_type,
                    'timestamp': datetime.now()
                }
            else:
                return {'error': 'No order data returned'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    symbol = f"{base}-USDT"
            
            params = {
                'ordId': order_id,
                'instId': symbol
            }
            
            result = self._make_request('GET', '/api/v5/trade/order', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            orders = result.get('data', [])
            if orders:
                order = orders[0]
                return {
                    'success': True,
                    'order_id': order.get('ordId'),
                    'status': order.get('state'),
                    'filled_amount': float(order.get('fillSz', '0')),
                    'remaining_amount': float(order.get('sz', '0')) - float(order.get('fillSz', '0')),
                    'average_price': float(order.get('avgPx', '0')),
                    'timestamp': int(order.get('uTime', '0'))
                }
            else:
                return {'error': 'Order not found'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    symbol = f"{base}-USDT"
            
            cancel_data = {
                'ordId': order_id,
                'instId': symbol
            }
            
            result = self._make_request('POST', '/api/v5/trade/cancel-order', data=cancel_data)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            return {'success': True, 'order_id': order_id, 'cancelled': True}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            result = self._make_request('GET', '/api/v5/account/positions')
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            positions = {}
            for pos_data in result.get('data', []):
                symbol = pos_data.get('instId', '')
                if symbol and float(pos_data.get('pos', '0')) != 0:
                    positions[symbol] = {
                        'symbol': symbol,
                        'side': pos_data.get('posSide', ''),
                        'size': float(pos_data.get('pos', '0')),
                        'entry_price': float(pos_data.get('avgPx', '0')),
                        'mark_price': float(pos_data.get('markPx', '0')),
                        'unrealized_pnl': float(pos_data.get('upl', '0')),
                        'leverage': float(pos_data.get('lever', '1')),
                        'margin': float(pos_data.get('margin', '0'))
                    }
            
            return {'success': True, 'positions': positions}
            
        except Exception as e:
            return {'error': str(e)}
    
    def set_leverage(self, symbol: str, leverage: int, margin_mode: str = 'cross') -> Dict[str, Any]:
        """Set leverage for a symbol"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    symbol = f"{base}-USDT"
            
            leverage_data = {
                'instId': symbol,
                'lever': str(leverage),
                'mgnMode': margin_mode
            }
            
            result = self._make_request('POST', '/api/v5/account/set-leverage', data=leverage_data)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            return {
                'success': True,
                'symbol': symbol,
                'leverage': leverage,
                'margin_mode': margin_mode
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_trading_fees(self, symbol: str) -> Dict[str, Any]:
        """Get trading fees for a symbol"""
        try:
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    symbol = f"{base}-USDT"
            
            params = {'instId': symbol}
            result = self._make_request('GET', '/api/v5/account/trade-fee', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Unknown error')}
            
            fee_data = result.get('data', [])
            if fee_data:
                fee = fee_data[0]
                return {
                    'success': True,
                    'symbol': symbol,
                    'maker_fee': float(fee.get('maker', '0.001')),
                    'taker_fee': float(fee.get('taker', '0.001'))
                }
            else:
                # Default fees if not available
                return {
                    'success': True,
                    'symbol': symbol,
                    'maker_fee': 0.001,  # 0.1%
                    'taker_fee': 0.001   # 0.1%
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and authentication"""
        try:
            if not self.api_key:
                return {
                    'success': False,
                    'error': 'API credentials not configured',
                    'authenticated': False
                }
            
            # Test public endpoint first
            result = self.get_ticker('BTC-USDT')
            if 'error' in result:
                return {
                    'success': False,
                    'error': f'Public API failed: {result["error"]}',
                    'authenticated': False
                }
            
            # Test private endpoint
            balance_result = self.get_account_balance()
            if 'error' in balance_result:
                return {
                    'success': False,
                    'error': f'Private API failed: {balance_result["error"]}',
                    'authenticated': False
                }
            
            return {
                'success': True,
                'message': 'OKX API connection successful',
                'authenticated': True,
                'sandbox': self.sandbox
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'authenticated': False
            }