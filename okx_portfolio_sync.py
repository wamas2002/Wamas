"""
OKX Portfolio Synchronization
Connects to real OKX account to fetch authentic wallet balances and positions
"""

import sqlite3
import os
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OKXPortfolioSync:
    def __init__(self):
        self.okx_service = None
        self.init_okx_connection()
    
    def init_okx_connection(self):
        """Initialize OKX connection with proper credentials"""
        try:
            from trading.okx_data_service import OKXDataService
            self.okx_service = OKXDataService()
            logger.info("OKX service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OKX service: {e}")
    
    def fetch_real_account_balance(self):
        """Fetch real account balance from OKX"""
        if not self.okx_service:
            logger.error("OKX service not available")
            return None
        
        try:
            # Require proper API credentials for live data access
            if not hasattr(self.okx_service, 'api_key') or not self.okx_service.api_key:
                raise Exception("OKX API credentials required for live portfolio access. Please configure authentication.")
            
            # Fetch live account data only
            balance_data = self.okx_service.get_account_balance()
            if balance_data:
                return self.process_real_balance_data(balance_data)
            
        except Exception as e:
            logger.warning(f"Could not fetch real OKX account data: {e}")
        
        # Require authentic API access only
        raise Exception("Authentic OKX API credentials required for portfolio access. Please configure API keys.")
    
    def process_real_balance_data(self, balance_data):
        """Process real OKX balance data"""
        try:
            # Parse OKX API response format
            portfolio = {
                'total_balance': 0,
                'available_balance': 0,
                'positions': []
            }
            
            # Process balance data according to OKX API format
            if isinstance(balance_data, dict) and 'data' in balance_data:
                for account in balance_data['data']:
                    for detail in account.get('details', []):
                        currency = detail.get('ccy', '')
                        available = float(detail.get('availBal', 0))
                        frozen = float(detail.get('frozenBal', 0))
                        
                        if currency == 'USDT':
                            portfolio['available_balance'] = available
                        
                        if available > 0 and currency != 'USDT':
                            # Get current price for valuation
                            symbol = f"{currency}USDT"
                            try:
                                current_price = self.okx_service.get_current_price(symbol)
                                if current_price:
                                    current_value = available * float(current_price)
                                    portfolio['positions'].append({
                                        'symbol': currency,
                                        'quantity': available,
                                        'current_price': float(current_price),
                                        'current_value': current_value,
                                        'unrealized_pnl': 0  # Would need position history for accurate PnL
                                    })
                                    portfolio['total_balance'] += current_value
                            except:
                                continue
            
            portfolio['total_balance'] += portfolio['available_balance']
            return portfolio
            
        except Exception as e:
            logger.error(f"Error processing real balance data: {e}")
            return None
    
    def update_portfolio_database(self, portfolio_data):
        """Update portfolio database with real OKX data"""
        if not portfolio_data:
            return False
        
        try:
            conn = sqlite3.connect('data/portfolio_tracking.db')
            cursor = conn.cursor()
            
            # Clear existing positions
            cursor.execute("DELETE FROM positions")
            
            # Insert current positions
            for position in portfolio_data['positions']:
                cursor.execute("""
                    INSERT INTO positions 
                    (symbol, quantity, current_price, current_value, unrealized_pnl, avg_price, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    position['symbol'],
                    position['quantity'],
                    position['current_price'],
                    position['current_value'],
                    position.get('unrealized_pnl', 0),
                    position.get('avg_price', position['current_price']),
                    datetime.now().isoformat()
                ))
            
            # Update portfolio metrics
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in portfolio_data['positions'])
            total_invested = portfolio_data['total_balance'] - portfolio_data['available_balance']
            
            cursor.execute("""
                INSERT INTO portfolio_metrics 
                (total_value, cash_balance, invested_value, daily_pnl, positions_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                portfolio_data['total_balance'],
                portfolio_data['available_balance'],
                total_invested,
                total_pnl,
                len(portfolio_data['positions']),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Portfolio database updated with OKX data: ${portfolio_data['total_balance']:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating portfolio database: {e}")
            return False
    
    def sync_portfolio(self):
        """Main synchronization function"""
        logger.info("Starting OKX portfolio synchronization...")
        
        # Fetch real account data
        portfolio_data = self.fetch_real_account_balance()
        
        if portfolio_data:
            # Update database
            success = self.update_portfolio_database(portfolio_data)
            
            if success:
                return {
                    'status': 'success',
                    'total_balance': portfolio_data['total_balance'],
                    'positions_count': len(portfolio_data['positions']),
                    'data_source': 'okx_api' if hasattr(self.okx_service, 'api_key') and self.okx_service.api_key else 'demo_realistic',
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'status': 'error',
            'message': 'Failed to sync portfolio data'
        }

def run_portfolio_sync():
    """Run portfolio synchronization"""
    syncer = OKXPortfolioSync()
    result = syncer.sync_portfolio()
    
    print("=" * 60)
    print("OKX PORTFOLIO SYNCHRONIZATION")
    print("=" * 60)
    
    if result['status'] == 'success':
        print(f"✅ Portfolio synchronized successfully")
        print(f"📊 Total Balance: ${result['total_balance']:.2f}")
        print(f"🏦 Positions: {result['positions_count']}")
        print(f"📡 Data Source: {result['data_source']}")
        print(f"🕒 Updated: {result['timestamp']}")
    else:
        print(f"❌ Synchronization failed: {result.get('message', 'Unknown error')}")
    
    print("=" * 60)
    return result

if __name__ == "__main__":
    run_portfolio_sync()