"""
Enhanced Binance Historical Data Collector using CCXT
Fetches up to 1 year of OHLCV data for comprehensive model training
"""

import ccxt
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

class BinanceDataCollector:
    """Comprehensive Binance data collector for AI model training"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.exchange = ccxt.binance({
            'apiKey': '',  # Public data only
            'secret': '',
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        self.db_path = db_path
        self.setup_database()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize SQLite database for historical data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create OHLCV table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                datetime TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                timeframe TEXT NOT NULL,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON ohlcv_data(symbol, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_historical_data(self, symbol: str, timeframe: str = '1m', 
                            days_back: int = 365) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe ('1m', '5m', '1h', '1d')
            days_back: Number of days to fetch (max 365)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            since = int(start_time.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            self.logger.info(f"Fetching {symbol} {timeframe} data from {start_time} to {end_time}")
            
            while current_since < int(end_time.timestamp() * 1000):
                try:
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=current_since,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    
                    # Update since for next batch
                    current_since = ohlcv[-1][0] + 1
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                    self.logger.info(f"Fetched {len(ohlcv)} candles, total: {len(all_data)}")
                    
                except Exception as e:
                    self.logger.error(f"Error fetching batch: {e}")
                    time.sleep(2)
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Add datetime column
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            self.logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame):
        """Save OHLCV data to SQLite database"""
        if df.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Use INSERT OR IGNORE to handle duplicates
            df.to_sql('ohlcv_data', conn, if_exists='append', index=False, method='multi')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved {len(df)} records to database")
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
    
    def load_from_database(self, symbol: str, timeframe: str = '1m', 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Load OHLCV data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM ohlcv_data 
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND datetime >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND datetime <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading from database: {e}")
            return pd.DataFrame()
    
    def collect_multiple_symbols(self, symbols: List[str], timeframe: str = '1m', 
                                days_back: int = 365):
        """Collect data for multiple symbols"""
        for symbol in symbols:
            try:
                self.logger.info(f"Collecting data for {symbol}")
                
                # Fetch data
                df = self.fetch_historical_data(symbol, timeframe, days_back)
                
                if not df.empty:
                    # Save to database
                    self.save_to_database(df)
                
                # Brief pause between symbols
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {e}")
                continue
    
    def get_data_summary(self) -> Dict:
        """Get summary of collected data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT 
                    symbol,
                    timeframe,
                    COUNT(*) as record_count,
                    MIN(datetime) as earliest_date,
                    MAX(datetime) as latest_date
                FROM ohlcv_data
                GROUP BY symbol, timeframe
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {}

if __name__ == "__main__":
    # Initialize collector
    collector = BinanceDataCollector()
    
    # Major trading pairs
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT',
        'SOL/USDT', 'XRP/USDT', 'DOT/USDT', 'AVAX/USDT'
    ]
    
    # Collect 1 year of 1-minute data
    collector.collect_multiple_symbols(symbols, '1m', 365)
    
    # Print summary
    summary = collector.get_data_summary()
    print("Data Collection Summary:")
    for item in summary:
        print(f"{item['symbol']} ({item['timeframe']}): {item['record_count']} records")
        print(f"  Range: {item['earliest_date']} to {item['latest_date']}")