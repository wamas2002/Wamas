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
    """Advanced OKX API connector for spot and futures trading with leverage support"""
    
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
        
        # Trading configuration
        self.trading_mode = "cross"  # cross or isolated margin
        self.position_side = "net"   # net, long, short for hedge mode
        
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate API signature"""
        try:
            if not self.secret_key:
                return ""
                
            # Create the prehash string (timestamp + method + requestPath + body)
            prehash = timestamp + method.upper() + request_path + body
            
            # Create signature using HMAC-SHA256
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
            print(f"Prehash string: {timestamp + method.upper() + request_path + body}")
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
            timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
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
    
    def set_leverage(self, symbol: str, leverage: int, margin_mode: str = "cross") -> Dict[str, Any]:
        """Set leverage for futures trading (1x-100x)"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            if not (1 <= leverage <= 100):
                return {'error': 'Leverage must be between 1x and 100x'}
            
            # Convert symbol format for futures
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    futures_symbol = f"{base}-USDT-SWAP"
                else:
                    futures_symbol = f"{symbol}-SWAP"
            else:
                futures_symbol = f"{symbol}-SWAP"
            
            body = {
                "instId": futures_symbol,
                "lever": str(leverage),
                "mgnMode": margin_mode,  # cross or isolated
                "posSide": "net"  # net for one-way mode
            }
            
            result = self._make_request('POST', '/api/v5/account/set-leverage', data=body)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to set leverage')}
            
            return {
                'success': True,
                'symbol': futures_symbol,
                'leverage': leverage,
                'margin_mode': margin_mode,
                'message': f'Leverage set to {leverage}x for {futures_symbol}'
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
    
    # ==================== ADVANCED FUTURES TRADING METHODS ====================
    
    def place_futures_order(self, symbol: str, side: str, size: float, 
                           order_type: str = "market", price: float = None,
                           leverage: int = 1, reduce_only: bool = False,
                           stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """Place futures order with leverage support"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Set leverage first
            leverage_result = self.set_leverage(symbol, leverage)
            if 'error' in leverage_result:
                return leverage_result
            
            # Convert symbol format for futures
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    futures_symbol = f"{base}-USDT-SWAP"
                else:
                    futures_symbol = f"{symbol}-SWAP"
            else:
                futures_symbol = f"{symbol}-SWAP"
            
            # Build order payload
            order_data = {
                "instId": futures_symbol,
                "tdMode": self.trading_mode,  # cross or isolated
                "side": side.lower(),  # buy or sell
                "ordType": order_type.lower(),  # market, limit, post_only, fok, ioc
                "sz": str(size),
                "reduceOnly": reduce_only
            }
            
            # Add price for limit orders
            if order_type.lower() in ['limit', 'post_only'] and price:
                order_data["px"] = str(price)
            
            # Place main order
            result = self._make_request('POST', '/api/v5/trade/order', data=order_data)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to place order')}
            
            order_info = result.get('data', [{}])[0]
            order_id = order_info.get('ordId', '')
            
            response = {
                'success': True,
                'order_id': order_id,
                'symbol': futures_symbol,
                'side': side,
                'size': size,
                'leverage': leverage,
                'order_type': order_type,
                'status': 'submitted'
            }
            
            # Place stop loss order if specified
            if stop_loss and order_id:
                sl_result = self._place_stop_order(futures_symbol, side, size, stop_loss, "stop_loss")
                if 'success' in sl_result:
                    response['stop_loss_order_id'] = sl_result.get('order_id')
            
            # Place take profit order if specified
            if take_profit and order_id:
                tp_result = self._place_stop_order(futures_symbol, side, size, take_profit, "take_profit")
                if 'success' in tp_result:
                    response['take_profit_order_id'] = tp_result.get('order_id')
            
            return response
            
        except Exception as e:
            return {'error': str(e)}
    
    def _place_stop_order(self, symbol: str, side: str, size: float, 
                         trigger_price: float, order_type: str) -> Dict[str, Any]:
        """Place stop loss or take profit order"""
        try:
            # Determine opposite side for closing position
            close_side = "sell" if side.lower() == "buy" else "buy"
            
            stop_data = {
                "instId": symbol,
                "tdMode": self.trading_mode,
                "side": close_side,
                "ordType": "conditional",
                "sz": str(size),
                "stopPx": str(trigger_price),
                "triggerPx": str(trigger_price),
                "reduceOnly": True
            }
            
            result = self._make_request('POST', '/api/v5/trade/order-algo', data=stop_data)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', f'Failed to place {order_type} order')}
            
            order_info = result.get('data', [{}])[0]
            return {
                'success': True,
                'order_id': order_info.get('algoId', ''),
                'type': order_type,
                'trigger_price': trigger_price
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def close_position(self, symbol: str, size: float = None) -> Dict[str, Any]:
        """Close position (partial or full)"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Get current position
            positions_result = self.get_positions()
            if 'error' in positions_result:
                return positions_result
            
            # Find the position
            position = None
            for pos in positions_result.get('positions', {}).values():
                if pos['symbol'] == symbol or pos['symbol'].replace('-SWAP', '') == symbol:
                    position = pos
                    break
            
            if not position:
                return {'error': f'No open position found for {symbol}'}
            
            # Determine close parameters
            close_size = size if size else abs(position['size'])
            close_side = "sell" if position['size'] > 0 else "buy"
            
            # Convert symbol format
            futures_symbol = position['symbol']
            
            # Place closing order
            close_data = {
                "instId": futures_symbol,
                "tdMode": self.trading_mode,
                "side": close_side,
                "ordType": "market",
                "sz": str(close_size),
                "reduceOnly": True
            }
            
            result = self._make_request('POST', '/api/v5/trade/order', data=close_data)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to close position')}
            
            order_info = result.get('data', [{}])[0]
            return {
                'success': True,
                'order_id': order_info.get('ordId', ''),
                'symbol': futures_symbol,
                'side': close_side,
                'size': close_size,
                'message': f'Position closing order placed for {close_size} {futures_symbol}'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_trading_instruments(self, inst_type: str = "SWAP") -> Dict[str, Any]:
        """Get available trading instruments"""
        try:
            params = {'instType': inst_type}  # SPOT, MARGIN, SWAP, FUTURES, OPTION
            result = self._make_request('GET', '/api/v5/public/instruments', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to get instruments')}
            
            instruments = []
            for inst_data in result.get('data', []):
                instrument = {
                    'symbol': inst_data.get('instId', ''),
                    'base_currency': inst_data.get('baseCcy', ''),
                    'quote_currency': inst_data.get('quoteCcy', ''),
                    'contract_value': float(inst_data.get('ctVal', '0')),
                    'min_size': float(inst_data.get('minSz', '0')),
                    'max_leverage': float(inst_data.get('maxLever', '1')),
                    'tick_size': float(inst_data.get('tickSz', '0')),
                    'lot_size': float(inst_data.get('lotSz', '0')),
                    'status': inst_data.get('state', '')
                }
                instruments.append(instrument)
            
            return {'success': True, 'instruments': instruments}
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get funding rate for perpetual swaps"""
        try:
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    futures_symbol = f"{base}-USDT-SWAP"
                else:
                    futures_symbol = f"{symbol}-SWAP"
            else:
                futures_symbol = f"{symbol}-SWAP"
            
            params = {'instId': futures_symbol}
            result = self._make_request('GET', '/api/v5/public/funding-rate', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to get funding rate')}
            
            data = result.get('data', [])
            if not data:
                return {'error': 'No funding rate data found'}
            
            funding_data = data[0]
            return {
                'success': True,
                'symbol': futures_symbol,
                'funding_rate': float(funding_data.get('fundingRate', '0')),
                'next_funding_time': int(funding_data.get('nextFundingTime', '0')),
                'funding_time': int(funding_data.get('fundingTime', '0'))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> Dict[str, Any]:
        """Get order history"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            params = {
                'limit': str(limit)
            }
            
            if symbol:
                # Convert symbol format
                if '-' not in symbol:
                    if symbol.endswith('USDT'):
                        base = symbol[:-4]
                        futures_symbol = f"{base}-USDT-SWAP"
                    else:
                        futures_symbol = f"{symbol}-SWAP"
                else:
                    futures_symbol = f"{symbol}-SWAP"
                params['instId'] = futures_symbol
            
            result = self._make_request('GET', '/api/v5/trade/orders-history-archive', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to get order history')}
            
            orders = []
            for order_data in result.get('data', []):
                order = {
                    'order_id': order_data.get('ordId', ''),
                    'symbol': order_data.get('instId', ''),
                    'side': order_data.get('side', ''),
                    'order_type': order_data.get('ordType', ''),
                    'size': float(order_data.get('sz', '0')),
                    'filled_size': float(order_data.get('fillSz', '0')),
                    'price': float(order_data.get('px', '0')),
                    'avg_price': float(order_data.get('avgPx', '0')),
                    'status': order_data.get('state', ''),
                    'fee': float(order_data.get('fee', '0')),
                    'timestamp': int(order_data.get('cTime', '0')),
                    'update_time': int(order_data.get('uTime', '0'))
                }
                orders.append(order)
            
            return {'success': True, 'orders': orders}
            
        except Exception as e:
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an open order"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Convert symbol format
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    futures_symbol = f"{base}-USDT-SWAP"
                else:
                    futures_symbol = f"{symbol}-SWAP"
            else:
                futures_symbol = f"{symbol}-SWAP"
            
            body = {
                "instId": futures_symbol,
                "ordId": order_id
            }
            
            result = self._make_request('POST', '/api/v5/trade/cancel-order', data=body)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to cancel order')}
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': futures_symbol,
                'message': 'Order cancelled successfully'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_account_configuration(self) -> Dict[str, Any]:
        """Get account configuration for trading mode and position mode"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            result = self._make_request('GET', '/api/v5/account/config')
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to get account configuration')}
            
            config_data = result.get('data', [{}])[0]
            return {
                'success': True,
                'account_level': config_data.get('acctLv', ''),
                'position_mode': config_data.get('posMode', ''),
                'auto_loan': config_data.get('autoLoan', ''),
                'margin_mode': config_data.get('mgnIsoMode', ''),
                'spot_offset_type': config_data.get('spotOffsetType', '')
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def set_position_mode(self, position_mode: str) -> Dict[str, Any]:
        """Set position mode (long_short_mode or net_mode)"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            body = {
                "posMode": position_mode  # long_short_mode or net_mode
            }
            
            result = self._make_request('POST', '/api/v5/account/set-position-mode', data=body)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to set position mode')}
            
            return {
                'success': True,
                'position_mode': position_mode,
                'message': f'Position mode set to {position_mode}'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_maximum_tradable_size(self, symbol: str, side: str, leverage: int = 1) -> Dict[str, Any]:
        """Get maximum tradable size for a symbol"""
        try:
            if not self.api_key:
                return {'error': 'API key not configured'}
            
            # Convert symbol format for futures
            if '-' not in symbol:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    futures_symbol = f"{base}-USDT-SWAP"
                else:
                    futures_symbol = f"{symbol}-SWAP"
            else:
                futures_symbol = f"{symbol}-SWAP"
            
            params = {
                'instId': futures_symbol,
                'tdMode': self.trading_mode,
                'px': '',  # Empty for market price
                'leverage': str(leverage)
            }
            
            result = self._make_request('GET', '/api/v5/account/max-size', params=params)
            
            if 'error' in result:
                return result
            
            if result.get('code') != '0':
                return {'error': result.get('msg', 'Failed to get maximum size')}
            
            data = result.get('data', [{}])[0]
            return {
                'success': True,
                'symbol': futures_symbol,
                'max_buy_size': float(data.get('maxBuy', '0')),
                'max_sell_size': float(data.get('maxSell', '0')),
                'leverage': leverage
            }
            
        except Exception as e:
            return {'error': str(e)}