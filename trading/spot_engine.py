"""
Spot Trading Engine - Handles all spot market trading logic
Supports dynamic symbol discovery and execution for all USDT spot pairs
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .dynamic_symbol_manager import DynamicSymbolManager, MarketType

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class SpotOrder:
    """Represents a spot trading order"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"
    
    def to_okx_params(self) -> Dict[str, Any]:
        """Convert to OKX API parameters"""
        params = {
            'instId': self.symbol,
            'tdMode': 'cash',  # Spot trading mode
            'side': self.side.value,
            'ordType': self.order_type.value,
            'sz': str(self.quantity)
        }
        
        if self.price:
            params['px'] = str(self.price)
        
        if self.stop_price:
            params['slTriggerPx'] = str(self.stop_price)
        
        if self.client_order_id:
            params['clOrdId'] = self.client_order_id
            
        return params

class SpotTradingEngine:
    """Manages spot trading operations across all USDT pairs"""
    
    def __init__(self, okx_connector, symbol_manager: DynamicSymbolManager):
        self.okx_connector = okx_connector
        self.symbol_manager = symbol_manager
        self.logger = logging.getLogger(__name__)
        
        # Trading state
        self.active_orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}
        self.portfolio_balance: Dict[str, float] = {}
        
        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_concurrent_positions = 20
        
        # Initialize portfolio tracking
        self._update_portfolio_balance()
        
    def _update_portfolio_balance(self):
        """Update current portfolio balance"""
        try:
            if self.okx_connector:
                balance = self.okx_connector.get_account_balance()
                if balance and 'data' in balance:
                    for item in balance['data']:
                        if item.get('ccy') == 'USDT':
                            self.portfolio_balance['USDT'] = float(item.get('availBal', 0))
                            break
            else:
                # Default balance for testing
                self.portfolio_balance['USDT'] = 10000.0
                
            self.logger.info(f"Portfolio balance updated: {self.portfolio_balance}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio balance: {e}")
            self.portfolio_balance['USDT'] = 10000.0
    
    def get_tradeable_symbols(self, min_volume: float = 100000) -> List[str]:
        """Get list of tradeable spot symbols with minimum volume"""
        return self.symbol_manager.filter_symbols(
            market_type=MarketType.SPOT,
            min_volume=min_volume,
            active_only=True
        )
    
    def calculate_position_size(self, symbol: str, signal_strength: float = 1.0) -> float:
        """Calculate appropriate position size for symbol"""
        try:
            symbol_info = self.symbol_manager.get_symbol_info(symbol)
            if not symbol_info:
                return 0.0
            
            # Get current price
            ticker = self.okx_connector.get_ticker(symbol) if self.okx_connector else None
            current_price = float(ticker.get('last', 1000)) if ticker else 1000.0
            
            # Calculate base position size
            available_balance = self.portfolio_balance.get('USDT', 0)
            max_position_value = available_balance * self.max_position_size * signal_strength
            
            # Calculate quantity
            quantity = max_position_value / current_price
            
            # Apply symbol-specific constraints
            min_quantity = symbol_info.min_notional / current_price
            quantity = max(quantity, min_quantity)
            
            # Round to step size
            step_size = symbol_info.step_size
            quantity = round(quantity / step_size) * step_size
            
            self.logger.info(f"Calculated position size for {symbol}: {quantity}")
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def place_market_buy(self, symbol: str, quantity: Optional[float] = None, 
                        signal_strength: float = 1.0) -> Optional[Dict]:
        """Place market buy order"""
        try:
            # Validate symbol
            if not self.symbol_manager.is_symbol_active(symbol):
                self.logger.error(f"Symbol {symbol} is not active for trading")
                return None
            
            # Calculate quantity if not provided
            if quantity is None:
                quantity = self.calculate_position_size(symbol, signal_strength)
            
            if quantity <= 0:
                self.logger.error(f"Invalid quantity for {symbol}: {quantity}")
                return None
            
            # Create order
            order = SpotOrder(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                client_order_id=f"spot_buy_{symbol}_{int(time.time())}"
            )
            
            # Execute order
            if self.okx_connector:
                result = self.okx_connector.place_order(**order.to_okx_params())
                if result and result.get('code') == '0':
                    order_data = result['data'][0]
                    self.active_orders[order_data['ordId']] = {
                        'symbol': symbol,
                        'side': 'buy',
                        'type': 'market',
                        'quantity': quantity,
                        'timestamp': datetime.now(),
                        'market_type': 'spot'
                    }
                    
                    self.logger.info(f"Market buy order placed for {symbol}: {quantity}")
                    return order_data
            else:
                # Simulate order for testing
                fake_order = {
                    'ordId': f"fake_order_{int(time.time())}",
                    'clOrdId': order.client_order_id,
                    'symbol': symbol,
                    'state': 'filled'
                }
                self.active_orders[fake_order['ordId']] = {
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'quantity': quantity,
                    'timestamp': datetime.now(),
                    'market_type': 'spot'
                }
                return fake_order
                
        except Exception as e:
            self.logger.error(f"Error placing market buy for {symbol}: {e}")
            return None
    
    def place_market_sell(self, symbol: str, quantity: Optional[float] = None) -> Optional[Dict]:
        """Place market sell order"""
        try:
            # Validate symbol
            if not self.symbol_manager.is_symbol_active(symbol):
                self.logger.error(f"Symbol {symbol} is not active for trading")
                return None
            
            # Get current position if quantity not provided
            if quantity is None:
                position = self.get_position(symbol)
                if not position:
                    self.logger.error(f"No position found for {symbol}")
                    return None
                quantity = position.get('size', 0)
            
            if quantity <= 0:
                self.logger.error(f"Invalid quantity for {symbol}: {quantity}")
                return None
            
            # Create order
            order = SpotOrder(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                client_order_id=f"spot_sell_{symbol}_{int(time.time())}"
            )
            
            # Execute order
            if self.okx_connector:
                result = self.okx_connector.place_order(**order.to_okx_params())
                if result and result.get('code') == '0':
                    order_data = result['data'][0]
                    self.active_orders[order_data['ordId']] = {
                        'symbol': symbol,
                        'side': 'sell',
                        'type': 'market',
                        'quantity': quantity,
                        'timestamp': datetime.now(),
                        'market_type': 'spot'
                    }
                    
                    self.logger.info(f"Market sell order placed for {symbol}: {quantity}")
                    return order_data
            else:
                # Simulate order for testing
                fake_order = {
                    'ordId': f"fake_order_{int(time.time())}",
                    'clOrdId': order.client_order_id,
                    'symbol': symbol,
                    'state': 'filled'
                }
                self.active_orders[fake_order['ordId']] = {
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': quantity,
                    'timestamp': datetime.now(),
                    'market_type': 'spot'
                }
                return fake_order
                
        except Exception as e:
            self.logger.error(f"Error placing market sell for {symbol}: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: OrderSide, quantity: float, 
                         price: float) -> Optional[Dict]:
        """Place limit order"""
        try:
            # Validate symbol
            if not self.symbol_manager.is_symbol_active(symbol):
                self.logger.error(f"Symbol {symbol} is not active for trading")
                return None
            
            # Create order
            order = SpotOrder(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                client_order_id=f"spot_limit_{symbol}_{int(time.time())}"
            )
            
            # Execute order
            if self.okx_connector:
                result = self.okx_connector.place_order(**order.to_okx_params())
                if result and result.get('code') == '0':
                    order_data = result['data'][0]
                    self.active_orders[order_data['ordId']] = {
                        'symbol': symbol,
                        'side': side.value,
                        'type': 'limit',
                        'quantity': quantity,
                        'price': price,
                        'timestamp': datetime.now(),
                        'market_type': 'spot'
                    }
                    
                    self.logger.info(f"Limit {side.value} order placed for {symbol}: {quantity} @ {price}")
                    return order_data
                    
        except Exception as e:
            self.logger.error(f"Error placing limit order for {symbol}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                self.logger.error(f"Order {order_id} not found in active orders")
                return False
            
            order_info = self.active_orders[order_id]
            symbol = order_info['symbol']
            
            if self.okx_connector:
                result = self.okx_connector.cancel_order(order_id, symbol)
                if result and result.get('code') == '0':
                    del self.active_orders[order_id]
                    self.logger.info(f"Order {order_id} cancelled successfully")
                    return True
            else:
                # Simulate cancellation
                del self.active_orders[order_id]
                return True
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol"""
        try:
            if self.okx_connector:
                positions = self.okx_connector.get_positions()
                if positions and 'data' in positions:
                    for pos in positions['data']:
                        if pos.get('instId') == symbol:
                            return {
                                'symbol': symbol,
                                'size': float(pos.get('pos', 0)),
                                'avg_price': float(pos.get('avgPx', 0)),
                                'unrealized_pnl': float(pos.get('upl', 0)),
                                'market_type': 'spot'
                            }
            
            # Return cached position if available
            return self.positions.get(symbol)
            
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return None
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions"""
        positions = {}
        
        try:
            if self.okx_connector:
                result = self.okx_connector.get_positions()
                if result and 'data' in result:
                    for pos in result['data']:
                        symbol = pos.get('instId')
                        if symbol and float(pos.get('pos', 0)) != 0:
                            positions[symbol] = {
                                'symbol': symbol,
                                'size': float(pos.get('pos', 0)),
                                'avg_price': float(pos.get('avgPx', 0)),
                                'unrealized_pnl': float(pos.get('upl', 0)),
                                'market_type': 'spot'
                            }
            
            self.positions = positions
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting all positions: {e}")
            return self.positions
    
    def set_stop_loss(self, symbol: str, stop_price: float) -> Optional[Dict]:
        """Set stop loss for existing position"""
        try:
            position = self.get_position(symbol)
            if not position:
                self.logger.error(f"No position found for {symbol}")
                return None
            
            quantity = abs(position['size'])
            side = OrderSide.SELL if position['size'] > 0 else OrderSide.BUY
            
            order = SpotOrder(
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP_LOSS,
                quantity=quantity,
                stop_price=stop_price,
                client_order_id=f"spot_sl_{symbol}_{int(time.time())}"
            )
            
            if self.okx_connector:
                result = self.okx_connector.place_order(**order.to_okx_params())
                if result and result.get('code') == '0':
                    self.logger.info(f"Stop loss set for {symbol} at {stop_price}")
                    return result['data'][0]
                    
        except Exception as e:
            self.logger.error(f"Error setting stop loss for {symbol}: {e}")
            return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        self._update_portfolio_balance()
        positions = self.get_all_positions()
        
        total_value = self.portfolio_balance.get('USDT', 0)
        total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
        
        active_symbols = list(positions.keys())
        tradeable_symbols = self.get_tradeable_symbols()
        
        return {
            'cash_balance': self.portfolio_balance.get('USDT', 0),
            'total_value': total_value + total_unrealized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'active_positions': len(positions),
            'active_symbols': active_symbols,
            'tradeable_symbols_count': len(tradeable_symbols),
            'max_concurrent_positions': self.max_concurrent_positions,
            'market_type': 'spot',
            'last_updated': datetime.now().isoformat()
        }
    
    def execute_ai_decision(self, symbol: str, decision: Dict[str, Any]) -> Optional[Dict]:
        """Execute trading decision from AI system"""
        try:
            action = decision.get('action', 'hold').lower()
            confidence = decision.get('confidence', 0.5)
            
            if action == 'buy':
                return self.place_market_buy(symbol, signal_strength=confidence)
            elif action == 'sell':
                return self.place_market_sell(symbol)
            elif action == 'hold':
                self.logger.info(f"AI decision for {symbol}: HOLD (confidence: {confidence:.2f})")
                return None
            else:
                self.logger.warning(f"Unknown AI action for {symbol}: {action}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing AI decision for {symbol}: {e}")
            return None