"""
Futures Trading Engine - Handles all futures/swap market trading logic
Supports dynamic symbol discovery and execution for all USDT futures pairs
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .dynamic_symbol_manager import DynamicSymbolManager, MarketType

class FuturesOrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"

class FuturesOrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class MarginMode(Enum):
    ISOLATED = "isolated"
    CROSS = "cross"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NET = "net"

@dataclass
class FuturesOrder:
    """Represents a futures trading order"""
    symbol: str
    side: FuturesOrderSide
    order_type: FuturesOrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_side: PositionSide = PositionSide.NET
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    
    def to_okx_params(self) -> Dict[str, Any]:
        """Convert to OKX API parameters"""
        params = {
            'instId': self.symbol,
            'tdMode': self.margin_mode.value,
            'side': self.side.value,
            'ordType': self.order_type.value,
            'sz': str(self.quantity),
            'posSide': self.position_side.value
        }
        
        if self.price:
            params['px'] = str(self.price)
        
        if self.stop_price:
            params['slTriggerPx'] = str(self.stop_price)
        
        if self.client_order_id:
            params['clOrdId'] = self.client_order_id
            
        if self.reduce_only:
            params['reduceOnly'] = 'true'
            
        return params

class FuturesTradingEngine:
    """Manages futures trading operations across all USDT pairs"""
    
    def __init__(self, okx_connector, symbol_manager: DynamicSymbolManager):
        self.okx_connector = okx_connector
        self.symbol_manager = symbol_manager
        self.logger = logging.getLogger(__name__)
        
        # Trading state
        self.active_orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}
        self.leverage_settings: Dict[str, int] = {}
        self.margin_balances: Dict[str, float] = {}
        
        # Risk management for futures
        self.max_leverage = 10
        self.max_position_size = 0.2  # 20% of margin per position
        self.max_concurrent_positions = 15
        self.funding_rate_threshold = 0.01  # 1% funding rate warning
        
        # Initialize futures-specific data
        self._update_margin_balance()
        self._initialize_leverage_settings()
        
    def _update_margin_balance(self):
        """Update current margin balance"""
        try:
            if self.okx_connector:
                balance = self.okx_connector.get_account_balance()
                if balance and 'data' in balance:
                    for item in balance['data']:
                        if item.get('ccy') == 'USDT':
                            self.margin_balances['USDT'] = float(item.get('availBal', 0))
                            break
            else:
                # Default balance for testing
                self.margin_balances['USDT'] = 5000.0
                
            self.logger.info(f"Margin balance updated: {self.margin_balances}")
            
        except Exception as e:
            self.logger.error(f"Error updating margin balance: {e}")
            self.margin_balances['USDT'] = 5000.0
    
    def _initialize_leverage_settings(self):
        """Initialize leverage settings for all symbols"""
        futures_symbols = self.symbol_manager.get_all_symbols(MarketType.FUTURES)
        
        for symbol in futures_symbols[:20]:  # Initialize top 20 symbols
            try:
                # Set default leverage based on symbol volatility
                symbol_info = self.symbol_manager.get_symbol_info(symbol)
                if symbol_info:
                    # Lower leverage for higher volume (more stable) pairs
                    if symbol_info.volume_24h > 1000000:
                        default_leverage = 5
                    elif symbol_info.volume_24h > 100000:
                        default_leverage = 3
                    else:
                        default_leverage = 2
                    
                    self.leverage_settings[symbol] = default_leverage
                    
                    # Set leverage on exchange if connected
                    if self.okx_connector:
                        self.set_leverage(symbol, default_leverage)
                        
            except Exception as e:
                self.logger.warning(f"Could not initialize leverage for {symbol}: {e}")
    
    def get_tradeable_symbols(self, min_volume: float = 50000) -> List[str]:
        """Get list of tradeable futures symbols with minimum volume"""
        return self.symbol_manager.filter_symbols(
            market_type=MarketType.FUTURES,
            min_volume=min_volume,
            active_only=True
        )
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            if leverage > self.max_leverage:
                self.logger.warning(f"Leverage {leverage} exceeds maximum {self.max_leverage}")
                leverage = self.max_leverage
            
            if self.okx_connector:
                result = self.okx_connector.set_leverage(symbol, leverage, 'cross')
                if result and result.get('code') == '0':
                    self.leverage_settings[symbol] = leverage
                    self.logger.info(f"Leverage set for {symbol}: {leverage}x")
                    return True
            else:
                # Simulate leverage setting
                self.leverage_settings[symbol] = leverage
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting leverage for {symbol}: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, signal_strength: float = 1.0, 
                              leverage: Optional[int] = None) -> float:
        """Calculate appropriate position size for futures symbol"""
        try:
            symbol_info = self.symbol_manager.get_symbol_info(symbol)
            if not symbol_info:
                return 0.0
            
            # Get current price
            ticker = self.okx_connector.get_ticker(symbol) if self.okx_connector else None
            current_price = float(ticker.get('last', 50000)) if ticker else 50000.0
            
            # Get leverage
            if leverage is None:
                leverage = self.leverage_settings.get(symbol, 1)
            
            # Calculate base position size
            available_margin = self.margin_balances.get('USDT', 0)
            max_position_value = available_margin * self.max_position_size * signal_strength
            
            # Apply leverage
            notional_value = max_position_value * leverage
            
            # Calculate contracts/quantity
            if 'SWAP' in symbol:
                # For swaps, quantity is in base currency
                quantity = notional_value / current_price
            else:
                # For futures, quantity is in contracts
                contract_size = 1  # Assume 1 unit contracts
                quantity = notional_value / (current_price * contract_size)
            
            # Apply symbol-specific constraints
            min_quantity = symbol_info.min_notional / current_price
            quantity = max(quantity, min_quantity)
            
            # Round to step size
            step_size = symbol_info.step_size
            quantity = round(quantity / step_size) * step_size
            
            self.logger.info(f"Calculated futures position size for {symbol}: {quantity} (leverage: {leverage}x)")
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def place_market_long(self, symbol: str, quantity: Optional[float] = None, 
                         signal_strength: float = 1.0, leverage: Optional[int] = None) -> Optional[Dict]:
        """Place market long order (buy)"""
        try:
            # Validate symbol
            if not self.symbol_manager.is_symbol_active(symbol):
                self.logger.error(f"Symbol {symbol} is not active for trading")
                return None
            
            # Set leverage if provided
            if leverage and leverage != self.leverage_settings.get(symbol):
                self.set_leverage(symbol, leverage)
            
            # Calculate quantity if not provided
            if quantity is None:
                quantity = self.calculate_position_size(symbol, signal_strength, leverage)
            
            if quantity <= 0:
                self.logger.error(f"Invalid quantity for {symbol}: {quantity}")
                return None
            
            # Create order
            order = FuturesOrder(
                symbol=symbol,
                side=FuturesOrderSide.BUY,
                order_type=FuturesOrderType.MARKET,
                quantity=quantity,
                leverage=self.leverage_settings.get(symbol, 1),
                position_side=PositionSide.LONG,
                client_order_id=f"futures_long_{symbol}_{int(time.time())}"
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
                        'leverage': order.leverage,
                        'position_side': 'long',
                        'timestamp': datetime.now(),
                        'market_type': 'futures'
                    }
                    
                    self.logger.info(f"Market long order placed for {symbol}: {quantity}")
                    return order_data
            else:
                # Simulate order for testing
                fake_order = {
                    'ordId': f"fake_futures_order_{int(time.time())}",
                    'clOrdId': order.client_order_id,
                    'symbol': symbol,
                    'state': 'filled',
                    'leverage': order.leverage
                }
                self.active_orders[fake_order['ordId']] = {
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'quantity': quantity,
                    'leverage': order.leverage,
                    'position_side': 'long',
                    'timestamp': datetime.now(),
                    'market_type': 'futures'
                }
                return fake_order
                
        except Exception as e:
            self.logger.error(f"Error placing market long for {symbol}: {e}")
            return None
    
    def place_market_short(self, symbol: str, quantity: Optional[float] = None, 
                          signal_strength: float = 1.0, leverage: Optional[int] = None) -> Optional[Dict]:
        """Place market short order (sell)"""
        try:
            # Validate symbol
            if not self.symbol_manager.is_symbol_active(symbol):
                self.logger.error(f"Symbol {symbol} is not active for trading")
                return None
            
            # Set leverage if provided
            if leverage and leverage != self.leverage_settings.get(symbol):
                self.set_leverage(symbol, leverage)
            
            # Calculate quantity if not provided
            if quantity is None:
                quantity = self.calculate_position_size(symbol, signal_strength, leverage)
            
            if quantity <= 0:
                self.logger.error(f"Invalid quantity for {symbol}: {quantity}")
                return None
            
            # Create order
            order = FuturesOrder(
                symbol=symbol,
                side=FuturesOrderSide.SELL,
                order_type=FuturesOrderType.MARKET,
                quantity=quantity,
                leverage=self.leverage_settings.get(symbol, 1),
                position_side=PositionSide.SHORT,
                client_order_id=f"futures_short_{symbol}_{int(time.time())}"
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
                        'leverage': order.leverage,
                        'position_side': 'short',
                        'timestamp': datetime.now(),
                        'market_type': 'futures'
                    }
                    
                    self.logger.info(f"Market short order placed for {symbol}: {quantity}")
                    return order_data
            else:
                # Simulate order for testing
                fake_order = {
                    'ordId': f"fake_futures_order_{int(time.time())}",
                    'clOrdId': order.client_order_id,
                    'symbol': symbol,
                    'state': 'filled',
                    'leverage': order.leverage
                }
                self.active_orders[fake_order['ordId']] = {
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': quantity,
                    'leverage': order.leverage,
                    'position_side': 'short',
                    'timestamp': datetime.now(),
                    'market_type': 'futures'
                }
                return fake_order
                
        except Exception as e:
            self.logger.error(f"Error placing market short for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, position_side: Optional[str] = None) -> Optional[Dict]:
        """Close existing position"""
        try:
            position = self.get_position(symbol, position_side)
            if not position:
                self.logger.error(f"No position found for {symbol}")
                return None
            
            quantity = abs(position['size'])
            # Opposite side to close position
            side = FuturesOrderSide.SELL if position['size'] > 0 else FuturesOrderSide.BUY
            pos_side = PositionSide.LONG if position['size'] > 0 else PositionSide.SHORT
            
            # Create closing order
            order = FuturesOrder(
                symbol=symbol,
                side=side,
                order_type=FuturesOrderType.MARKET,
                quantity=quantity,
                position_side=pos_side,
                reduce_only=True,
                client_order_id=f"futures_close_{symbol}_{int(time.time())}"
            )
            
            # Execute order
            if self.okx_connector:
                result = self.okx_connector.place_order(**order.to_okx_params())
                if result and result.get('code') == '0':
                    self.logger.info(f"Position closed for {symbol}: {quantity}")
                    return result['data'][0]
            else:
                # Simulate close
                fake_order = {
                    'ordId': f"fake_close_order_{int(time.time())}",
                    'symbol': symbol,
                    'state': 'filled'
                }
                return fake_order
                
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return None
    
    def get_position(self, symbol: str, position_side: Optional[str] = None) -> Optional[Dict]:
        """Get current position for symbol"""
        try:
            if self.okx_connector:
                positions = self.okx_connector.get_positions(symbol)
                if positions and 'data' in positions:
                    for pos in positions['data']:
                        if pos.get('instId') == symbol:
                            pos_size = float(pos.get('pos', 0))
                            if pos_size != 0:
                                return {
                                    'symbol': symbol,
                                    'size': pos_size,
                                    'side': 'long' if pos_size > 0 else 'short',
                                    'avg_price': float(pos.get('avgPx', 0)),
                                    'unrealized_pnl': float(pos.get('upl', 0)),
                                    'leverage': int(pos.get('lever', 1)),
                                    'margin': float(pos.get('margin', 0)),
                                    'market_type': 'futures'
                                }
            
            # Return cached position if available
            return self.positions.get(symbol)
            
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return None
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current futures positions"""
        positions = {}
        
        try:
            if self.okx_connector:
                result = self.okx_connector.get_positions()
                if result and 'data' in result:
                    for pos in result['data']:
                        symbol = pos.get('instId')
                        pos_size = float(pos.get('pos', 0))
                        
                        if symbol and pos_size != 0:
                            positions[symbol] = {
                                'symbol': symbol,
                                'size': pos_size,
                                'side': 'long' if pos_size > 0 else 'short',
                                'avg_price': float(pos.get('avgPx', 0)),
                                'unrealized_pnl': float(pos.get('upl', 0)),
                                'leverage': int(pos.get('lever', 1)),
                                'margin': float(pos.get('margin', 0)),
                                'market_type': 'futures'
                            }
            
            self.positions = positions
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting all positions: {e}")
            return self.positions
    
    def monitor_funding_rates(self) -> Dict[str, float]:
        """Monitor funding rates for all active positions"""
        funding_rates = {}
        active_symbols = list(self.positions.keys())
        
        for symbol in active_symbols:
            try:
                symbol_info = self.symbol_manager.get_symbol_info(symbol)
                if symbol_info and symbol_info.funding_rate is not None:
                    funding_rates[symbol] = symbol_info.funding_rate
                    
                    # Warn about high funding rates
                    if abs(symbol_info.funding_rate) > self.funding_rate_threshold:
                        self.logger.warning(
                            f"High funding rate for {symbol}: {symbol_info.funding_rate:.4f}"
                        )
                        
            except Exception as e:
                self.logger.warning(f"Could not get funding rate for {symbol}: {e}")
        
        return funding_rates
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive futures portfolio summary"""
        self._update_margin_balance()
        positions = self.get_all_positions()
        funding_rates = self.monitor_funding_rates()
        
        total_margin = sum(pos.get('margin', 0) for pos in positions.values())
        total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
        
        # Calculate total exposure
        total_exposure = 0
        for pos in positions.values():
            exposure = abs(pos.get('size', 0)) * pos.get('avg_price', 0)
            total_exposure += exposure
        
        active_symbols = list(positions.keys())
        tradeable_symbols = self.get_tradeable_symbols()
        
        return {
            'margin_balance': self.margin_balances.get('USDT', 0),
            'total_margin_used': total_margin,
            'unrealized_pnl': total_unrealized_pnl,
            'total_exposure': total_exposure,
            'active_positions': len(positions),
            'active_symbols': active_symbols,
            'tradeable_symbols_count': len(tradeable_symbols),
            'funding_rates': funding_rates,
            'max_leverage': self.max_leverage,
            'leverage_settings': dict(list(self.leverage_settings.items())[:10]),
            'market_type': 'futures',
            'last_updated': datetime.now().isoformat()
        }
    
    def execute_ai_decision(self, symbol: str, decision: Dict[str, Any]) -> Optional[Dict]:
        """Execute trading decision from AI system for futures"""
        try:
            action = decision.get('action', 'hold').lower()
            confidence = decision.get('confidence', 0.5)
            leverage = decision.get('leverage', self.leverage_settings.get(symbol, 2))
            
            if action == 'buy' or action == 'long':
                return self.place_market_long(symbol, signal_strength=confidence, leverage=leverage)
            elif action == 'sell' or action == 'short':
                return self.place_market_short(symbol, signal_strength=confidence, leverage=leverage)
            elif action == 'close':
                return self.close_position(symbol)
            elif action == 'hold':
                self.logger.info(f"AI decision for {symbol}: HOLD (confidence: {confidence:.2f})")
                return None
            else:
                self.logger.warning(f"Unknown AI action for {symbol}: {action}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing AI decision for {symbol}: {e}")
            return None