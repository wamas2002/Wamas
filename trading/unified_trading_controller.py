"""
Unified Trading Controller - Manages all trading operations across spot and futures markets
Provides centralized interface for dynamic symbol trading with market-specific logic
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .dynamic_symbol_manager import DynamicSymbolManager, MarketType, TradingPair
from .spot_engine import SpotTradingEngine
from .futures_engine import FuturesTradingEngine
from config import Config

class TradeExecutionMode(Enum):
    MANUAL = "manual"
    AUTOMATED = "automated"
    HYBRID = "hybrid"

@dataclass
class UnifiedTradeRequest:
    """Unified trade request supporting both spot and futures"""
    symbol: str
    action: str  # buy, sell, long, short, close
    market_type: MarketType
    quantity: Optional[float] = None
    price: Optional[float] = None
    leverage: Optional[int] = None
    signal_strength: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict] = None

class UnifiedTradingController:
    """Central controller for all trading operations across markets"""
    
    def __init__(self, okx_connector=None):
        self.okx_connector = okx_connector
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.symbol_manager = DynamicSymbolManager(okx_connector)
        
        # Initialize trading engines
        self.spot_engine = None
        self.futures_engine = None
        
        if Config.ENABLE_SPOT_TRADING:
            self.spot_engine = SpotTradingEngine(okx_connector, self.symbol_manager)
            self.logger.info("Spot trading engine initialized")
            
        if Config.ENABLE_FUTURES_TRADING:
            self.futures_engine = FuturesTradingEngine(okx_connector, self.symbol_manager)
            self.logger.info("Futures trading engine initialized")
        
        # Trading state
        self.execution_mode = TradeExecutionMode.AUTOMATED
        self.active_symbols: Dict[str, MarketType] = {}
        self.trade_history: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize with dynamic symbols
        self._populate_active_symbols()
        
    def _populate_active_symbols(self):
        """Populate active symbols from both markets"""
        try:
            # Get top volume symbols from each market
            if self.spot_engine:
                spot_symbols = self.symbol_manager.get_symbols_by_volume(
                    MarketType.SPOT, limit=Config.MAX_SYMBOLS_PER_MARKET
                )
                for symbol in spot_symbols:
                    self.active_symbols[symbol] = MarketType.SPOT
            
            if self.futures_engine:
                futures_symbols = self.symbol_manager.get_symbols_by_volume(
                    MarketType.FUTURES, limit=Config.MAX_SYMBOLS_PER_MARKET
                )
                for symbol in futures_symbols:
                    self.active_symbols[symbol] = MarketType.FUTURES
            
            # Update config with dynamic symbols
            Config.SUPPORTED_SYMBOLS = list(self.active_symbols.keys())
            
            self.logger.info(f"Populated {len(self.active_symbols)} active symbols across markets")
            
        except Exception as e:
            self.logger.error(f"Error populating active symbols: {e}")
    
    def get_all_tradeable_symbols(self, market_type: Optional[MarketType] = None) -> Dict[str, MarketType]:
        """Get all tradeable symbols with their market types"""
        if market_type:
            return {k: v for k, v in self.active_symbols.items() if v == market_type}
        return self.active_symbols.copy()
    
    def get_symbol_info(self, symbol: str) -> Optional[TradingPair]:
        """Get comprehensive symbol information"""
        return self.symbol_manager.get_symbol_info(symbol)
    
    def get_market_type(self, symbol: str) -> Optional[MarketType]:
        """Determine market type for symbol"""
        return self.active_symbols.get(symbol) or self.symbol_manager.get_market_type(symbol)
    
    def execute_trade(self, trade_request: UnifiedTradeRequest) -> Optional[Dict]:
        """Execute trade across appropriate market"""
        try:
            symbol = trade_request.symbol
            action = trade_request.action.lower()
            market_type = trade_request.market_type
            
            # Validate symbol is active
            if not self.symbol_manager.is_symbol_active(symbol):
                self.logger.error(f"Symbol {symbol} is not active for trading")
                return None
            
            # Route to appropriate engine
            if market_type == MarketType.SPOT and self.spot_engine:
                return self._execute_spot_trade(trade_request)
            elif market_type == MarketType.FUTURES and self.futures_engine:
                return self._execute_futures_trade(trade_request)
            else:
                self.logger.error(f"No engine available for {market_type} trading")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {trade_request.symbol}: {e}")
            return None
    
    def _execute_spot_trade(self, trade_request: UnifiedTradeRequest) -> Optional[Dict]:
        """Execute spot market trade"""
        action = trade_request.action.lower()
        
        if action in ['buy', 'long']:
            result = self.spot_engine.place_market_buy(
                trade_request.symbol,
                trade_request.quantity,
                trade_request.signal_strength
            )
        elif action in ['sell', 'short']:
            result = self.spot_engine.place_market_sell(
                trade_request.symbol,
                trade_request.quantity
            )
        else:
            self.logger.error(f"Unknown spot action: {action}")
            return None
        
        # Set stop loss and take profit if specified
        if result and trade_request.stop_loss:
            self.spot_engine.set_stop_loss(trade_request.symbol, trade_request.stop_loss)
        
        if result:
            self._record_trade(trade_request, result, MarketType.SPOT)
        
        return result
    
    def _execute_futures_trade(self, trade_request: UnifiedTradeRequest) -> Optional[Dict]:
        """Execute futures market trade"""
        action = trade_request.action.lower()
        
        if action in ['buy', 'long']:
            result = self.futures_engine.place_market_long(
                trade_request.symbol,
                trade_request.quantity,
                trade_request.signal_strength,
                trade_request.leverage
            )
        elif action in ['sell', 'short']:
            result = self.futures_engine.place_market_short(
                trade_request.symbol,
                trade_request.quantity,
                trade_request.signal_strength,
                trade_request.leverage
            )
        elif action == 'close':
            result = self.futures_engine.close_position(trade_request.symbol)
        else:
            self.logger.error(f"Unknown futures action: {action}")
            return None
        
        if result:
            self._record_trade(trade_request, result, MarketType.FUTURES)
        
        return result
    
    def _record_trade(self, trade_request: UnifiedTradeRequest, execution_result: Dict, market_type: MarketType):
        """Record trade execution in history"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_request.symbol,
            'action': trade_request.action,
            'market_type': market_type.value,
            'quantity': trade_request.quantity,
            'price': trade_request.price,
            'leverage': trade_request.leverage,
            'signal_strength': trade_request.signal_strength,
            'order_id': execution_result.get('ordId'),
            'execution_result': execution_result,
            'metadata': trade_request.metadata
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def execute_ai_decision(self, symbol: str, decision: Dict[str, Any]) -> Optional[Dict]:
        """Execute AI-generated trading decision"""
        try:
            # Determine market type
            market_type = self.get_market_type(symbol)
            if not market_type:
                self.logger.error(f"Unknown market type for symbol {symbol}")
                return None
            
            # Create unified trade request
            trade_request = UnifiedTradeRequest(
                symbol=symbol,
                action=decision.get('action', 'hold'),
                market_type=market_type,
                signal_strength=decision.get('confidence', 0.5),
                leverage=decision.get('leverage'),
                metadata={
                    'source': 'ai_decision',
                    'model': decision.get('model', 'unknown'),
                    'confidence': decision.get('confidence', 0.5),
                    'reasoning': decision.get('reasoning', '')
                }
            )
            
            return self.execute_trade(trade_request)
            
        except Exception as e:
            self.logger.error(f"Error executing AI decision for {symbol}: {e}")
            return None
    
    def get_portfolio_overview(self) -> Dict[str, Any]:
        """Get comprehensive portfolio overview across all markets"""
        overview = {
            'timestamp': datetime.now().isoformat(),
            'total_value': 0,
            'unrealized_pnl': 0,
            'markets': {},
            'active_positions': 0,
            'active_symbols': list(self.active_symbols.keys())[:20],  # Top 20
            'symbol_statistics': self.symbol_manager.get_symbol_statistics()
        }
        
        # Get spot portfolio summary
        if self.spot_engine:
            spot_summary = self.spot_engine.get_portfolio_summary()
            overview['markets']['spot'] = spot_summary
            overview['total_value'] += spot_summary.get('total_value', 0)
            overview['unrealized_pnl'] += spot_summary.get('unrealized_pnl', 0)
            overview['active_positions'] += spot_summary.get('active_positions', 0)
        
        # Get futures portfolio summary
        if self.futures_engine:
            futures_summary = self.futures_engine.get_portfolio_summary()
            overview['markets']['futures'] = futures_summary
            overview['total_value'] += futures_summary.get('total_exposure', 0)
            overview['unrealized_pnl'] += futures_summary.get('unrealized_pnl', 0)
            overview['active_positions'] += futures_summary.get('active_positions', 0)
        
        return overview
    
    def get_positions_by_market(self, market_type: MarketType) -> Dict[str, Dict]:
        """Get positions for specific market type"""
        if market_type == MarketType.SPOT and self.spot_engine:
            return self.spot_engine.get_all_positions()
        elif market_type == MarketType.FUTURES and self.futures_engine:
            return self.futures_engine.get_all_positions()
        else:
            return {}
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions across markets"""
        all_positions = {}
        
        # Spot positions
        if self.spot_engine:
            spot_positions = self.spot_engine.get_all_positions()
            for symbol, position in spot_positions.items():
                position['market_type'] = 'spot'
                all_positions[f"{symbol}_spot"] = position
        
        # Futures positions
        if self.futures_engine:
            futures_positions = self.futures_engine.get_all_positions()
            for symbol, position in futures_positions.items():
                position['market_type'] = 'futures'
                all_positions[f"{symbol}_futures"] = position
        
        return all_positions
    
    def close_all_positions(self, market_type: Optional[MarketType] = None) -> List[Dict]:
        """Close all positions in specified market or all markets"""
        results = []
        
        if market_type is None or market_type == MarketType.SPOT:
            if self.spot_engine:
                spot_positions = self.spot_engine.get_all_positions()
                for symbol in spot_positions:
                    result = self.spot_engine.place_market_sell(symbol)
                    if result:
                        results.append(result)
        
        if market_type is None or market_type == MarketType.FUTURES:
            if self.futures_engine:
                futures_positions = self.futures_engine.get_all_positions()
                for symbol in futures_positions:
                    result = self.futures_engine.close_position(symbol)
                    if result:
                        results.append(result)
        
        return results
    
    def update_symbol_discovery(self) -> bool:
        """Force update of symbol discovery"""
        try:
            success = self.symbol_manager.force_update()
            if success:
                self._populate_active_symbols()
                self.logger.info("Symbol discovery updated successfully")
            return success
        except Exception as e:
            self.logger.error(f"Error updating symbol discovery: {e}")
            return False
    
    def get_market_statistics(self) -> Dict[str, Any]:
        """Get comprehensive market statistics"""
        return {
            'symbol_stats': self.symbol_manager.get_symbol_statistics(),
            'active_symbols_count': len(self.active_symbols),
            'spot_symbols': len([s for s, m in self.active_symbols.items() if m == MarketType.SPOT]),
            'futures_symbols': len([s for s, m in self.active_symbols.items() if m == MarketType.FUTURES]),
            'trade_history_count': len(self.trade_history),
            'execution_mode': self.execution_mode.value,
            'last_update': datetime.now().isoformat()
        }
    
    def export_trading_config(self, file_path: str):
        """Export complete trading configuration"""
        config = {
            'active_symbols': {k: v.value for k, v in self.active_symbols.items()},
            'execution_mode': self.execution_mode.value,
            'spot_enabled': Config.ENABLE_SPOT_TRADING,
            'futures_enabled': Config.ENABLE_FUTURES_TRADING,
            'trade_history_count': len(self.trade_history),
            'symbol_statistics': self.symbol_manager.get_symbol_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Trading configuration exported to {file_path}")
    
    def set_execution_mode(self, mode: TradeExecutionMode):
        """Set trade execution mode"""
        self.execution_mode = mode
        self.logger.info(f"Execution mode set to: {mode.value}")
    
    def get_trade_history(self, limit: int = 100, market_type: Optional[MarketType] = None) -> List[Dict]:
        """Get recent trade history with optional filtering"""
        trades = self.trade_history[-limit:]
        
        if market_type:
            trades = [t for t in trades if t.get('market_type') == market_type.value]
        
        return trades
    
    def calculate_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Calculate performance metrics across all markets"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            t for t in self.trade_history 
            if datetime.fromisoformat(t['timestamp']) > cutoff_date
        ]
        
        if not recent_trades:
            return {'message': 'No recent trades for analysis'}
        
        # Basic metrics
        total_trades = len(recent_trades)
        spot_trades = len([t for t in recent_trades if t['market_type'] == 'spot'])
        futures_trades = len([t for t in recent_trades if t['market_type'] == 'futures'])
        
        return {
            'period_days': days,
            'total_trades': total_trades,
            'spot_trades': spot_trades,
            'futures_trades': futures_trades,
            'avg_signal_strength': sum(t.get('signal_strength', 0) for t in recent_trades) / total_trades,
            'markets_traded': len(set(t['market_type'] for t in recent_trades)),
            'symbols_traded': len(set(t['symbol'] for t in recent_trades)),
            'calculation_timestamp': datetime.now().isoformat()
        }