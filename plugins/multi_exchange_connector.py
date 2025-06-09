"""
Multi-Exchange Connector Plugin
Integrates Binance Spot, Binance Futures, and DEX data alongside OKX
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
import requests

logger = logging.getLogger(__name__)

class MultiExchangeConnector:
    def __init__(self):
        self.exchanges = self._initialize_exchanges()
        self.supported_symbols = {}
        self.price_cache = {}
        
    def _initialize_exchanges(self) -> Dict:
        """Initialize all supported exchanges"""
        exchanges = {}
        
        try:
            # OKX Exchange
            exchanges['okx'] = ccxt.okx({
                'sandbox': False,
                'rateLimit': 100,
                'enableRateLimit': True,
            })
            
            # Binance Spot
            exchanges['binance'] = ccxt.binance({
                'sandbox': False,
                'rateLimit': 100,
                'enableRateLimit': True,
            })
            
            # Binance Futures
            exchanges['binance_futures'] = ccxt.binance({
                'sandbox': False,
                'rateLimit': 100,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            
            logger.info("Multi-exchange connector initialized successfully")
            return exchanges
            
        except Exception as e:
            logger.error(f"Exchange initialization error: {e}")
            return {}
    
    def get_exchange_info(self, exchange_name: str) -> Dict:
        """Get exchange information and available markets"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return {'error': f'Exchange {exchange_name} not supported'}
            
            markets = exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if '/USDT' in symbol]
            
            return {
                'name': exchange_name,
                'markets_count': len(markets),
                'usdt_pairs': len(usdt_pairs),
                'top_pairs': usdt_pairs[:20],
                'status': 'active',
                'features': {
                    'spot': True,
                    'futures': 'futures' in exchange_name,
                    'margin': exchange.has.get('margin', False),
                    'options': exchange.has.get('option', False)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting {exchange_name} info: {e}")
            return {'error': str(e)}
    
    def get_multi_exchange_prices(self, symbol: str) -> Dict:
        """Get current prices from all exchanges for comparison"""
        prices = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                ticker = exchange.fetch_ticker(symbol)
                prices[exchange_name] = {
                    'price': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'volume': ticker['baseVolume'],
                    'change_24h': ticker['percentage'],
                    'timestamp': ticker['timestamp']
                }
            except Exception as e:
                logger.error(f"Failed to fetch authentic data for {symbol} from {exchange_name}: {e}")
                # Do not add mock data - skip this exchange
                continue
        
        # Calculate arbitrage opportunities
        if len(prices) > 1:
            price_values = [p['price'] for p in prices.values()]
            min_price = min(price_values)
            max_price = max(price_values)
            arbitrage_pct = ((max_price - min_price) / min_price) * 100
            
            prices['arbitrage'] = {
                'opportunity': arbitrage_pct > 0.1,
                'percentage': arbitrage_pct,
                'buy_exchange': [k for k, v in prices.items() if v['price'] == min_price][0],
                'sell_exchange': [k for k, v in prices.items() if v['price'] == max_price][0],
                'profit_potential': arbitrage_pct
            }
        
        return {
            'symbol': symbol,
            'prices': prices,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_exchange_portfolio(self, exchange_name: str) -> Dict:
        """Get authentic portfolio breakdown for specific exchange"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise Exception(f"Exchange {exchange_name} not supported")
            
            # Fetch authentic balance data (requires API keys in production)
            # For now, we'll raise an error to force authentic API key configuration
            raise Exception(f"Authentic portfolio access requires API keys for {exchange_name}. Please configure authentication.")
            
        except Exception as e:
            logger.error(f"Error getting authentic {exchange_name} portfolio: {e}")
            raise Exception(f"Unable to fetch authentic portfolio from {exchange_name}: {e}")
    
    def get_aggregated_portfolio(self) -> Dict:
        """Get authentic aggregated portfolio across all exchanges"""
        try:
            # Require authentic portfolio data from each exchange
            # Cannot aggregate without real portfolio access
            raise Exception("Authentic aggregated portfolio requires API keys for all exchanges. Please configure authentication for OKX, Binance, and other exchanges.")
            
        except Exception as e:
            logger.error(f"Error getting authentic aggregated portfolio: {e}")
            raise Exception(f"Unable to fetch authentic aggregated portfolio: {e}")
    
    def get_exchange_orderbook(self, symbol: str, exchange_name: str, limit: int = 20) -> Dict:
        """Get orderbook data from specific exchange"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return {'error': f'Exchange {exchange_name} not supported'}
            
            orderbook = exchange.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'bids': orderbook['bids'][:limit],
                'asks': orderbook['asks'][:limit],
                'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0] if orderbook['asks'] and orderbook['bids'] else 0,
                'timestamp': orderbook['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch authentic orderbook for {symbol} from {exchange_name}: {e}")
            raise Exception(f"Unable to fetch authentic orderbook data for {symbol} from {exchange_name}: {e}")
    
    def compare_exchanges(self, symbol: str) -> Dict:
        """Compare trading conditions across exchanges"""
        comparison = {}
        
        for exchange_name in self.exchanges.keys():
            try:
                # Get orderbook for spread analysis
                orderbook = self.get_exchange_orderbook(symbol, exchange_name, 5)
                
                # Get recent trades for liquidity analysis
                price_data = self.get_multi_exchange_prices(symbol)
                exchange_price = price_data['prices'].get(exchange_name, {})
                
                comparison[exchange_name] = {
                    'spread': orderbook.get('spread', 0),
                    'spread_pct': (orderbook.get('spread', 0) / exchange_price.get('price', 1)) * 100,
                    'volume_24h': exchange_price.get('volume', 0),
                    'price': exchange_price.get('price', 0),
                    'liquidity_score': self._calculate_liquidity_score(orderbook),
                    'trading_fees': self._get_trading_fees(exchange_name),
                    'rating': 0  # Will calculate overall rating
                }
                
                # Calculate overall rating (0-100)
                rating = 100
                rating -= min(comparison[exchange_name]['spread_pct'] * 10, 20)  # Penalize high spreads
                rating += min(np.log10(comparison[exchange_name]['volume_24h'] + 1) * 5, 30)  # Reward high volume
                rating -= comparison[exchange_name]['trading_fees'] * 100  # Penalize high fees
                
                comparison[exchange_name]['rating'] = max(0, min(100, rating))
                
            except Exception as e:
                logger.error(f"Error comparing {exchange_name}: {e}")
                comparison[exchange_name] = {'error': str(e)}
        
        # Find best exchange
        best_exchange = max(comparison.keys(), 
                          key=lambda x: comparison[x].get('rating', 0))
        
        return {
            'symbol': symbol,
            'comparison': comparison,
            'best_exchange': best_exchange,
            'recommendation': f"Trade on {best_exchange} for best conditions",
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_liquidity_score(self, orderbook: Dict) -> float:
        """Calculate liquidity score from orderbook depth"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            # Calculate depth within 1% of best price
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            bid_depth = sum([vol for price, vol in bids if price >= best_bid * 0.99])
            ask_depth = sum([vol for price, vol in asks if price <= best_ask * 1.01])
            
            return min((bid_depth + ask_depth) / 2, 100)  # Cap at 100
            
        except:
            return 50.0  # Default score
    
    def _get_trading_fees(self, exchange_name: str) -> float:
        """Get trading fees for exchange (as percentage)"""
        fee_structure = {
            'okx': 0.0008,      # 0.08%
            'binance': 0.001,   # 0.10%
            'binance_futures': 0.0004  # 0.04%
        }
        return fee_structure.get(exchange_name, 0.001)

# Plugin instance
multi_exchange_connector = MultiExchangeConnector()

def get_exchange_prices(symbol: str) -> Dict:
    """Get prices across all exchanges"""
    return multi_exchange_connector.get_multi_exchange_prices(symbol)

def get_portfolio_by_exchange(exchange_name: str) -> Dict:
    """Get portfolio for specific exchange"""
    return multi_exchange_connector.get_exchange_portfolio(exchange_name)

def get_aggregated_portfolio() -> Dict:
    """Get aggregated portfolio across all exchanges"""
    return multi_exchange_connector.get_aggregated_portfolio()

def compare_trading_conditions(symbol: str) -> Dict:
    """Compare trading conditions across exchanges"""
    return multi_exchange_connector.compare_exchanges(symbol)

def get_supported_exchanges() -> List[str]:
    """Get list of supported exchanges"""
    return list(multi_exchange_connector.exchanges.keys())

def get_exchange_info(exchange_name: str) -> Dict:
    """Get exchange information"""
    return multi_exchange_connector.get_exchange_info(exchange_name)