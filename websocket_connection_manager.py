#!/usr/bin/env python3
"""
WebSocket Connection Manager
Advanced real-time data streaming with connection resilience
"""

import websocket
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional
import queue

class WebSocketConnectionManager:
    def __init__(self):
        self.connections = {}
        self.message_queues = {}
        self.reconnect_intervals = {}
        self.active = True
        self.max_reconnect_attempts = 5
        
    def create_connection(self, name: str, url: str, on_message: Callable = None, on_error: Callable = None):
        """Create a new WebSocket connection"""
        if name in self.connections:
            self.close_connection(name)
        
        self.message_queues[name] = queue.Queue()
        self.reconnect_intervals[name] = 1
        
        def on_message_wrapper(ws, message):
            try:
                data = json.loads(message)
                self.message_queues[name].put(data)
                if on_message:
                    on_message(data)
            except json.JSONDecodeError:
                if on_message:
                    on_message(message)
        
        def on_error_wrapper(ws, error):
            print(f"WebSocket {name} error: {error}")
            if on_error:
                on_error(error)
            self.schedule_reconnect(name, url, on_message, on_error)
        
        def on_close_wrapper(ws, close_status_code, close_msg):
            print(f"WebSocket {name} closed: {close_status_code} - {close_msg}")
            if self.active:
                self.schedule_reconnect(name, url, on_message, on_error)
        
        def on_open_wrapper(ws):
            print(f"WebSocket {name} connected")
            self.reconnect_intervals[name] = 1
        
        try:
            ws = websocket.WebSocketApp(
                url,
                on_message=on_message_wrapper,
                on_error=on_error_wrapper,
                on_close=on_close_wrapper,
                on_open=on_open_wrapper
            )
            
            self.connections[name] = ws
            
            # Start connection in thread
            thread = threading.Thread(target=ws.run_forever, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to create WebSocket {name}: {e}")
            return False
    
    def schedule_reconnect(self, name: str, url: str, on_message: Callable, on_error: Callable):
        """Schedule automatic reconnection"""
        if not self.active:
            return
        
        interval = min(self.reconnect_intervals[name], 30)
        self.reconnect_intervals[name] = min(interval * 2, 60)
        
        def reconnect():
            time.sleep(interval)
            if self.active and name not in self.connections:
                print(f"Attempting to reconnect WebSocket {name}")
                self.create_connection(name, url, on_message, on_error)
        
        threading.Thread(target=reconnect, daemon=True).start()
    
    def send_message(self, name: str, message: dict) -> bool:
        """Send message through WebSocket connection"""
        if name not in self.connections:
            return False
        
        try:
            ws = self.connections[name]
            ws.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"Failed to send message to {name}: {e}")
            return False
    
    def get_messages(self, name: str, timeout: float = 0.1) -> List[dict]:
        """Get queued messages from connection"""
        if name not in self.message_queues:
            return []
        
        messages = []
        q = self.message_queues[name]
        
        try:
            while True:
                message = q.get(timeout=timeout)
                messages.append(message)
                q.task_done()
        except queue.Empty:
            pass
        
        return messages
    
    def close_connection(self, name: str):
        """Close specific WebSocket connection"""
        if name in self.connections:
            try:
                self.connections[name].close()
                del self.connections[name]
            except:
                pass
        
        if name in self.message_queues:
            del self.message_queues[name]
        
        if name in self.reconnect_intervals:
            del self.reconnect_intervals[name]
    
    def close_all_connections(self):
        """Close all WebSocket connections"""
        self.active = False
        
        for name in list(self.connections.keys()):
            self.close_connection(name)
        
        print("All WebSocket connections closed")
    
    def get_connection_status(self) -> Dict:
        """Get status of all connections"""
        status = {}
        for name, ws in self.connections.items():
            try:
                status[name] = {
                    'connected': ws.sock and ws.sock.connected,
                    'url': ws.url,
                    'messages_queued': self.message_queues[name].qsize()
                }
            except:
                status[name] = {
                    'connected': False,
                    'url': 'unknown',
                    'messages_queued': 0
                }
        
        return status


class RealTimeDataStreamer:
    def __init__(self):
        self.ws_manager = WebSocketConnectionManager()
        self.price_data = {}
        self.subscribers = {}
        
    def start_okx_price_stream(self, symbols: List[str]):
        """Start OKX price streaming"""
        def on_message(data):
            try:
                if 'data' in data:
                    for item in data['data']:
                        symbol = item.get('instId', '').replace('-', '/')
                        if symbol in symbols:
                            self.price_data[symbol] = {
                                'price': float(item.get('last', 0)),
                                'timestamp': datetime.now().isoformat(),
                                'volume': float(item.get('vol24h', 0)),
                                'change_24h': float(item.get('sodUtc0', 0))
                            }
                            
                            # Notify subscribers
                            if symbol in self.subscribers:
                                for callback in self.subscribers[symbol]:
                                    callback(self.price_data[symbol])
            except Exception as e:
                print(f"Error processing OKX data: {e}")
        
        def on_error(error):
            print(f"OKX WebSocket error: {error}")
        
        # OKX WebSocket URL (public market data)
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        if self.ws_manager.create_connection("okx_prices", url, on_message, on_error):
            # Subscribe to tickers
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": symbol.replace('/', '-')} for symbol in symbols]
            }
            
            time.sleep(1)  # Wait for connection
            return self.ws_manager.send_message("okx_prices", subscribe_msg)
        
        return False
    
    def subscribe_to_price_updates(self, symbol: str, callback: Callable):
        """Subscribe to price updates for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        
        self.subscribers[symbol].append(callback)
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price data for symbol"""
        return self.price_data.get(symbol)
    
    def get_all_prices(self) -> Dict:
        """Get all current price data"""
        return self.price_data.copy()
    
    def stop_streaming(self):
        """Stop all data streaming"""
        self.ws_manager.close_all_connections()


def create_trading_websocket_server():
    """Create WebSocket server for trading dashboard"""
    import asyncio
    import websockets
    import json
    
    connected_clients = set()
    data_streamer = RealTimeDataStreamer()
    
    # Start price streaming for major pairs
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'ADA/USDT']
    data_streamer.start_okx_price_stream(symbols)
    
    async def handle_client(websocket, path):
        """Handle individual client connections"""
        connected_clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            # Send initial data
            initial_data = {
                'type': 'initial_prices',
                'data': data_streamer.get_all_prices()
            }
            await websocket.send(json.dumps(initial_data))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'subscribe':
                        # Handle subscription requests
                        symbol = data.get('symbol')
                        if symbol:
                            def price_callback(price_data):
                                # Send price update to this client
                                update_msg = {
                                    'type': 'price_update',
                                    'symbol': symbol,
                                    'data': price_data
                                }
                                asyncio.create_task(websocket.send(json.dumps(update_msg)))
                            
                            data_streamer.subscribe_to_price_updates(symbol, price_callback)
                            
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            connected_clients.discard(websocket)
            print(f"Client disconnected: {websocket.remote_address}")
    
    async def broadcast_updates():
        """Broadcast regular updates to all clients"""
        while True:
            if connected_clients:
                try:
                    # Get current prices
                    prices = data_streamer.get_all_prices()
                    
                    if prices:
                        broadcast_data = {
                            'type': 'price_broadcast',
                            'data': prices,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Send to all connected clients
                        disconnected = set()
                        for client in connected_clients:
                            try:
                                await client.send(json.dumps(broadcast_data))
                            except:
                                disconnected.add(client)
                        
                        # Remove disconnected clients
                        connected_clients -= disconnected
                        
                except Exception as e:
                    print(f"Broadcast error: {e}")
            
            await asyncio.sleep(2)  # Broadcast every 2 seconds
    
    # Start WebSocket server
    async def start_server():
        print("Starting WebSocket server on port 8765...")
        
        # Start background broadcasting
        asyncio.create_task(broadcast_updates())
        
        # Start WebSocket server
        server = await websockets.serve(handle_client, "0.0.0.0", 8765)
        print("WebSocket server started successfully")
        
        return server
    
    return start_server


def main():
    """Test WebSocket functionality"""
    print("WEBSOCKET CONNECTION MANAGER TEST")
    print("=" * 40)
    
    # Test real-time data streaming
    streamer = RealTimeDataStreamer()
    
    # Start streaming major cryptocurrency pairs
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    if streamer.start_okx_price_stream(symbols):
        print("✅ Price streaming started")
        
        # Subscribe to BTC price updates
        def btc_callback(price_data):
            print(f"BTC Price Update: ${price_data['price']:,.2f}")
        
        streamer.subscribe_to_price_updates('BTC/USDT', btc_callback)
        
        # Let it run for 30 seconds
        time.sleep(30)
        
        # Display final prices
        prices = streamer.get_all_prices()
        print("\nFinal Prices:")
        for symbol, data in prices.items():
            print(f"  {symbol}: ${data['price']:,.2f}")
        
        streamer.stop_streaming()
    else:
        print("❌ Failed to start price streaming")
    
    print("=" * 40)

if __name__ == '__main__':
    main()