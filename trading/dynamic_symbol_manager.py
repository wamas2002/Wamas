"""
Dynamic Symbol Manager - Automatically discovers and manages all USDT trading pairs
Supports both Spot and Futures markets with real-time symbol discovery
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
import json
from dataclasses import dataclass
from enum import Enum

class MarketType(Enum):
    SPOT = "spot"
    FUTURES = "futures"

@dataclass
class TradingPair:
    """Represents a trading pair with market-specific metadata"""
    symbol: str
    base_asset: str
    quote_asset: str
    market_type: MarketType
    is_active: bool
    min_notional: float
    tick_size: float
    step_size: float
    volume_24h: float
    price_precision: int
    quantity_precision: int
    margin_enabled: bool = False
    leverage_bracket: Optional[Dict] = None
    funding_rate: Optional[float] = None
    
    @property
    def symbol_id(self) -> str:
        """Unique identifier for the symbol across markets"""
        return f"{self.symbol}_{self.market_type.value}"

class DynamicSymbolManager:
    """Manages dynamic discovery and tracking of all USDT trading pairs"""
    
    def __init__(self, okx_connector=None):
        self.okx_connector = okx_connector
        self.logger = logging.getLogger(__name__)
        
        # Symbol storage
        self.spot_symbols: Dict[str, TradingPair] = {}
        self.futures_symbols: Dict[str, TradingPair] = {}
        self.all_symbols: Dict[str, TradingPair] = {}
        
        # Update tracking
        self.last_update = None
        self.update_interval = 3600  # 1 hour
        self.auto_update_enabled = True
        
        # Threading for background updates
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Initialize symbol discovery
        self.discover_all_symbols()
        self.start_auto_update()
    
    def discover_all_symbols(self) -> bool:
        """Discover all available USDT trading pairs from OKX"""
        try:
            self.logger.info("Starting symbol discovery for all USDT pairs...")
            
            # Discover spot symbols
            spot_count = self._discover_spot_symbols()
            
            # Discover futures symbols
            futures_count = self._discover_futures_symbols()
            
            # Update combined symbol dictionary
            self._update_combined_symbols()
            
            self.last_update = datetime.now()
            
            self.logger.info(f"Symbol discovery completed: {spot_count} spot, {futures_count} futures pairs")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during symbol discovery: {e}")
            return False
    
    def _discover_spot_symbols(self) -> int:
        """Discover all spot USDT pairs"""
        try:
            if not self.okx_connector:
                # Use default symbols if no connector available
                return self._load_default_spot_symbols()
            
            # Get all spot instruments
            instruments = self.okx_connector.get_instruments(inst_type="SPOT")
            
            spot_count = 0
            for instrument in instruments:
                if instrument.get('quoteCcy') == 'USDT' and instrument.get('state') == 'live':
                    symbol = instrument['instId']
                    base_asset = instrument['baseCcy']
                    
                    # Get 24h ticker data for volume
                    ticker = self.okx_connector.get_ticker(symbol)
                    volume_24h = float(ticker.get('vol24h', 0)) if ticker else 0
                    
                    trading_pair = TradingPair(
                        symbol=symbol,
                        base_asset=base_asset,
                        quote_asset='USDT',
                        market_type=MarketType.SPOT,
                        is_active=True,
                        min_notional=float(instrument.get('minSz', 0)),
                        tick_size=float(instrument.get('tickSz', 0)),
                        step_size=float(instrument.get('lotSz', 0)),
                        volume_24h=volume_24h,
                        price_precision=len(instrument.get('tickSz', '0').split('.')[-1]),
                        quantity_precision=len(instrument.get('lotSz', '0').split('.')[-1])
                    )
                    
                    self.spot_symbols[symbol] = trading_pair
                    spot_count += 1
            
            self.logger.info(f"Discovered {spot_count} spot USDT pairs")
            return spot_count
            
        except Exception as e:
            self.logger.error(f"Error discovering spot symbols: {e}")
            return self._load_default_spot_symbols()
    
    def _discover_futures_symbols(self) -> int:
        """Discover all futures USDT pairs"""
        try:
            if not self.okx_connector:
                # Use default symbols if no connector available
                return self._load_default_futures_symbols()
            
            # Get all swap instruments (perpetual futures)
            instruments = self.okx_connector.get_instruments(inst_type="SWAP")
            
            futures_count = 0
            for instrument in instruments:
                if instrument.get('settleCcy') == 'USDT' and instrument.get('state') == 'live':
                    symbol = instrument['instId']
                    base_asset = instrument['ctValCcy']
                    
                    # Get 24h ticker data
                    ticker = self.okx_connector.get_ticker(symbol)
                    volume_24h = float(ticker.get('vol24h', 0)) if ticker else 0
                    
                    # Get funding rate
                    funding_rate = self._get_funding_rate(symbol)
                    
                    trading_pair = TradingPair(
                        symbol=symbol,
                        base_asset=base_asset,
                        quote_asset='USDT',
                        market_type=MarketType.FUTURES,
                        is_active=True,
                        min_notional=float(instrument.get('minSz', 0)),
                        tick_size=float(instrument.get('tickSz', 0)),
                        step_size=float(instrument.get('lotSz', 0)),
                        volume_24h=volume_24h,
                        price_precision=len(instrument.get('tickSz', '0').split('.')[-1]),
                        quantity_precision=len(instrument.get('lotSz', '0').split('.')[-1]),
                        margin_enabled=True,
                        funding_rate=funding_rate
                    )
                    
                    self.futures_symbols[symbol] = trading_pair
                    futures_count += 1
            
            self.logger.info(f"Discovered {futures_count} futures USDT pairs")
            return futures_count
            
        except Exception as e:
            self.logger.error(f"Error discovering futures symbols: {e}")
            return self._load_default_futures_symbols()
    
    def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for futures symbol"""
        try:
            if self.okx_connector:
                funding_data = self.okx_connector.get_funding_rate(symbol)
                if funding_data:
                    return float(funding_data.get('fundingRate', 0))
        except Exception as e:
            self.logger.warning(f"Could not get funding rate for {symbol}: {e}")
        return None
    
    def _load_default_spot_symbols(self) -> int:
        """Load default spot symbols when API is unavailable"""
        default_spots = [
            "BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT", "SOL-USDT",
            "XRP-USDT", "DOT-USDT", "AVAX-USDT", "LINK-USDT", "LTC-USDT",
            "UNI-USDT", "ATOM-USDT", "NEAR-USDT", "FTM-USDT", "SAND-USDT",
            "MANA-USDT", "CRV-USDT", "SUSHI-USDT", "COMP-USDT", "YFI-USDT"
        ]
        
        for symbol in default_spots:
            base_asset = symbol.split('-')[0]
            trading_pair = TradingPair(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset='USDT',
                market_type=MarketType.SPOT,
                is_active=True,
                min_notional=1.0,
                tick_size=0.01,
                step_size=0.001,
                volume_24h=1000000,
                price_precision=2,
                quantity_precision=3
            )
            self.spot_symbols[symbol] = trading_pair
        
        return len(default_spots)
    
    def _load_default_futures_symbols(self) -> int:
        """Load default futures symbols when API is unavailable"""
        default_futures = [
            "BTC-USDT-SWAP", "ETH-USDT-SWAP", "BNB-USDT-SWAP", "ADA-USDT-SWAP",
            "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOT-USDT-SWAP", "AVAX-USDT-SWAP",
            "LINK-USDT-SWAP", "LTC-USDT-SWAP", "UNI-USDT-SWAP", "ATOM-USDT-SWAP"
        ]
        
        for symbol in default_futures:
            base_asset = symbol.split('-')[0]
            trading_pair = TradingPair(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset='USDT',
                market_type=MarketType.FUTURES,
                is_active=True,
                min_notional=1.0,
                tick_size=0.01,
                step_size=0.001,
                volume_24h=1000000,
                price_precision=2,
                quantity_precision=3,
                margin_enabled=True,
                funding_rate=0.0001
            )
            self.futures_symbols[symbol] = trading_pair
        
        return len(default_futures)
    
    def _update_combined_symbols(self):
        """Update the combined symbols dictionary"""
        self.all_symbols.clear()
        self.all_symbols.update(self.spot_symbols)
        self.all_symbols.update(self.futures_symbols)
    
    def start_auto_update(self):
        """Start automatic symbol discovery updates"""
        if self.auto_update_enabled:
            self.update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
            self.update_thread.start()
            self.logger.info("Started automatic symbol discovery updates")
    
    def _auto_update_loop(self):
        """Background loop for automatic updates"""
        while not self.stop_event.wait(self.update_interval):
            try:
                self.discover_all_symbols()
            except Exception as e:
                self.logger.error(f"Error in auto-update loop: {e}")
    
    def stop_auto_update(self):
        """Stop automatic symbol discovery updates"""
        self.stop_event.set()
        if self.update_thread:
            self.update_thread.join()
        self.logger.info("Stopped automatic symbol discovery updates")
    
    # Public API methods
    
    def get_all_symbols(self, market_type: Optional[MarketType] = None) -> List[str]:
        """Get list of all symbols, optionally filtered by market type"""
        if market_type == MarketType.SPOT:
            return list(self.spot_symbols.keys())
        elif market_type == MarketType.FUTURES:
            return list(self.futures_symbols.keys())
        else:
            return list(self.all_symbols.keys())
    
    def get_symbol_info(self, symbol: str) -> Optional[TradingPair]:
        """Get detailed information for a specific symbol"""
        return self.all_symbols.get(symbol)
    
    def get_symbols_by_volume(self, market_type: Optional[MarketType] = None, limit: int = 50) -> List[str]:
        """Get symbols sorted by 24h volume"""
        if market_type == MarketType.SPOT:
            symbols = self.spot_symbols
        elif market_type == MarketType.FUTURES:
            symbols = self.futures_symbols
        else:
            symbols = self.all_symbols
        
        sorted_symbols = sorted(symbols.items(), key=lambda x: x[1].volume_24h, reverse=True)
        return [symbol for symbol, _ in sorted_symbols[:limit]]
    
    def get_symbols_by_base_asset(self, base_asset: str) -> List[str]:
        """Get all symbols for a specific base asset"""
        return [symbol for symbol, pair in self.all_symbols.items() 
                if pair.base_asset == base_asset]
    
    def is_symbol_active(self, symbol: str) -> bool:
        """Check if a symbol is active for trading"""
        pair = self.all_symbols.get(symbol)
        return pair is not None and pair.is_active
    
    def get_market_type(self, symbol: str) -> Optional[MarketType]:
        """Get market type for a symbol"""
        pair = self.all_symbols.get(symbol)
        return pair.market_type if pair else None
    
    def filter_symbols(self, 
                      market_type: Optional[MarketType] = None,
                      min_volume: Optional[float] = None,
                      base_assets: Optional[List[str]] = None,
                      active_only: bool = True) -> List[str]:
        """Filter symbols based on criteria"""
        symbols = self.all_symbols
        
        if market_type:
            symbols = {k: v for k, v in symbols.items() if v.market_type == market_type}
        
        if active_only:
            symbols = {k: v for k, v in symbols.items() if v.is_active}
        
        if min_volume:
            symbols = {k: v for k, v in symbols.items() if v.volume_24h >= min_volume}
        
        if base_assets:
            symbols = {k: v for k, v in symbols.items() if v.base_asset in base_assets}
        
        return list(symbols.keys())
    
    def get_symbol_statistics(self) -> Dict[str, Any]:
        """Get comprehensive symbol statistics"""
        total_symbols = len(self.all_symbols)
        spot_count = len(self.spot_symbols)
        futures_count = len(self.futures_symbols)
        
        # Calculate volume statistics
        total_volume = sum(pair.volume_24h for pair in self.all_symbols.values())
        spot_volume = sum(pair.volume_24h for pair in self.spot_symbols.values())
        futures_volume = sum(pair.volume_24h for pair in self.futures_symbols.values())
        
        # Most active pairs
        top_volume_pairs = self.get_symbols_by_volume(limit=10)
        
        return {
            'total_symbols': total_symbols,
            'spot_symbols': spot_count,
            'futures_symbols': futures_count,
            'total_volume_24h': total_volume,
            'spot_volume_24h': spot_volume,
            'futures_volume_24h': futures_volume,
            'top_volume_pairs': top_volume_pairs,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_interval_hours': self.update_interval / 3600
        }
    
    def export_symbols_config(self, file_path: str):
        """Export symbol configuration to JSON file"""
        config = {
            'spot_symbols': {k: {
                'symbol': v.symbol,
                'base_asset': v.base_asset,
                'volume_24h': v.volume_24h,
                'min_notional': v.min_notional,
                'tick_size': v.tick_size
            } for k, v in self.spot_symbols.items()},
            'futures_symbols': {k: {
                'symbol': v.symbol,
                'base_asset': v.base_asset,
                'volume_24h': v.volume_24h,
                'min_notional': v.min_notional,
                'tick_size': v.tick_size,
                'funding_rate': v.funding_rate
            } for k, v in self.futures_symbols.items()},
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Exported symbol configuration to {file_path}")

    def force_update(self) -> bool:
        """Force immediate symbol discovery update"""
        return self.discover_all_symbols()