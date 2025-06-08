"""
CryptoPanic RSS News Sentiment Collector
Scrapes headlines every 10 minutes and adds NLP sentiment scores
"""

import feedparser
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import re
from bs4 import BeautifulSoup

class NewsSentimentCollector:
    """Collect and analyze cryptocurrency news sentiment"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # News sources (RSS feeds)
        self.news_sources = {
            'cryptopanic': 'https://cryptopanic.com/api/v1/posts/?auth_token=&public=true&kind=news&format=rss',
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'decrypt': 'https://decrypt.co/feed',
            'bitcoinist': 'https://bitcoinist.com/feed/',
        }
        
        self.setup_database()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize database for news sentiment storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create news sentiment table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                datetime TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                url TEXT,
                symbols TEXT,
                textblob_polarity REAL,
                textblob_subjectivity REAL,
                vader_compound REAL,
                vader_positive REAL,
                vader_negative REAL,
                vader_neutral REAL,
                sentiment_score REAL,
                sentiment_label TEXT,
                UNIQUE(url, timestamp)
            )
        ''')
        
        # Create aggregated sentiment table (per minute intervals)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_aggregated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                datetime TEXT NOT NULL,
                symbol TEXT,
                avg_sentiment REAL,
                sentiment_count INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                sentiment_momentum REAL,
                UNIQUE(timestamp, symbol)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_crypto_symbols(self, text: str) -> List[str]:
        """Extract cryptocurrency symbols from text"""
        # Common crypto symbols and names
        crypto_patterns = {
            'BTC': ['bitcoin', 'btc', 'btcusd', 'btcusdt'],
            'ETH': ['ethereum', 'eth', 'ethusd', 'ethusdt'],
            'BNB': ['binance', 'bnb', 'bnbusd', 'bnbusdt'],
            'ADA': ['cardano', 'ada', 'adausd', 'adausdt'],
            'SOL': ['solana', 'sol', 'solusd', 'solusdt'],
            'XRP': ['ripple', 'xrp', 'xrpusd', 'xrpusdt'],
            'DOT': ['polkadot', 'dot', 'dotusd', 'dotusdt'],
            'AVAX': ['avalanche', 'avax', 'avaxusd', 'avaxusdt'],
            'MATIC': ['polygon', 'matic', 'maticusd', 'maticusdt'],
            'LINK': ['chainlink', 'link', 'linkusd', 'linkusdt'],
        }
        
        text_lower = text.lower()
        found_symbols = []
        
        for symbol, patterns in crypto_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    found_symbols.append(symbol)
                    break
        
        return list(set(found_symbols))  # Remove duplicates
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using multiple NLP methods"""
        if not text:
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral'
            }
        
        # TextBlob analysis
        blob = TextBlob(text)
        tb_polarity = blob.sentiment.polarity
        tb_subjectivity = blob.sentiment.subjectivity
        
        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # Combined sentiment score
        # Weight VADER compound (better for social media/news) more heavily
        combined_score = (0.3 * tb_polarity + 0.7 * vader_scores['compound'])
        
        # Determine sentiment label
        if combined_score >= 0.1:
            sentiment_label = 'positive'
        elif combined_score <= -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'textblob_polarity': tb_polarity,
            'textblob_subjectivity': tb_subjectivity,
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'sentiment_score': combined_score,
            'sentiment_label': sentiment_label
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    def fetch_news_from_source(self, source_name: str, url: str) -> List[Dict]:
        """Fetch news from a specific RSS source"""
        try:
            self.logger.info(f"Fetching news from {source_name}")
            
            # Parse RSS feed
            feed = feedparser.parse(url)
            
            news_items = []
            current_time = datetime.now()
            
            for entry in feed.entries:
                try:
                    # Extract published time
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_time = datetime(*entry.published_parsed[:6])
                    else:
                        pub_time = current_time
                    
                    # Only process recent news (last 24 hours)
                    if (current_time - pub_time).days > 1:
                        continue
                    
                    # Extract content
                    title = self.clean_text(entry.get('title', ''))
                    content = self.clean_text(entry.get('summary', '') or entry.get('description', ''))
                    full_text = f"{title} {content}"
                    
                    # Skip if no meaningful content
                    if len(full_text.strip()) < 10:
                        continue
                    
                    # Extract crypto symbols
                    symbols = self.extract_crypto_symbols(full_text)
                    
                    # Analyze sentiment
                    sentiment = self.analyze_sentiment(full_text)
                    
                    news_item = {
                        'timestamp': int(pub_time.timestamp() * 1000),
                        'datetime': pub_time.isoformat(),
                        'source': source_name,
                        'title': title,
                        'content': content,
                        'url': entry.get('link', ''),
                        'symbols': ','.join(symbols),
                        **sentiment
                    }
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    self.logger.error(f"Error processing entry from {source_name}: {e}")
                    continue
            
            self.logger.info(f"Collected {len(news_items)} news items from {source_name}")
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching news from {source_name}: {e}")
            return []
    
    def collect_all_news(self) -> List[Dict]:
        """Collect news from all sources"""
        all_news = []
        
        for source_name, url in self.news_sources.items():
            try:
                news_items = self.fetch_news_from_source(source_name, url)
                all_news.extend(news_items)
                
                # Brief pause between sources
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting from {source_name}: {e}")
                continue
        
        return all_news
    
    def save_news_to_database(self, news_items: List[Dict]):
        """Save news sentiment data to database"""
        if not news_items:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(news_items)
            
            # Save to database (use REPLACE to handle duplicates)
            df.to_sql('news_sentiment', conn, if_exists='append', index=False, method='multi')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved {len(news_items)} news items to database")
            
        except Exception as e:
            self.logger.error(f"Error saving news to database: {e}")
    
    def aggregate_sentiment_by_minute(self, symbol: str = None):
        """Aggregate sentiment scores by minute intervals"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query for aggregation
            if symbol:
                query = """
                    SELECT 
                        (timestamp / 60000) * 60000 as minute_timestamp,
                        AVG(sentiment_score) as avg_sentiment,
                        COUNT(*) as sentiment_count,
                        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                    FROM news_sentiment 
                    WHERE symbols LIKE ?
                    AND datetime >= datetime('now', '-24 hours')
                    GROUP BY minute_timestamp
                    ORDER BY minute_timestamp
                """
                params = [f'%{symbol}%']
            else:
                query = """
                    SELECT 
                        (timestamp / 60000) * 60000 as minute_timestamp,
                        AVG(sentiment_score) as avg_sentiment,
                        COUNT(*) as sentiment_count,
                        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                    FROM news_sentiment 
                    WHERE datetime >= datetime('now', '-24 hours')
                    GROUP BY minute_timestamp
                    ORDER BY minute_timestamp
                """
                params = []
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                # Calculate sentiment momentum (rate of change)
                df['sentiment_momentum'] = df['avg_sentiment'].pct_change().fillna(0)
                
                # Add datetime and symbol columns
                df['datetime'] = pd.to_datetime(df['minute_timestamp'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
                df['symbol'] = symbol or 'ALL'
                
                # Save aggregated data
                df[['minute_timestamp', 'datetime', 'symbol', 'avg_sentiment', 'sentiment_count',
                    'positive_count', 'negative_count', 'neutral_count', 'sentiment_momentum']].to_sql(
                    'sentiment_aggregated', conn, if_exists='append', index=False, method='multi'
                )
                
                conn.commit()
            
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error aggregating sentiment: {e}")
            return pd.DataFrame()
    
    def get_sentiment_for_timestamp(self, timestamp: int, symbol: str = None) -> float:
        """Get sentiment score for a specific timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Look for sentiment within 5 minutes of the timestamp
            if symbol:
                query = """
                    SELECT AVG(sentiment_score) as avg_sentiment
                    FROM news_sentiment
                    WHERE timestamp BETWEEN ? AND ?
                    AND symbols LIKE ?
                """
                params = [timestamp - 300000, timestamp + 300000, f'%{symbol}%']
            else:
                query = """
                    SELECT AVG(sentiment_score) as avg_sentiment
                    FROM news_sentiment
                    WHERE timestamp BETWEEN ? AND ?
                """
                params = [timestamp - 300000, timestamp + 300000]
            
            result = conn.execute(query, params).fetchone()
            conn.close()
            
            return result[0] if result[0] is not None else 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment for timestamp: {e}")
            return 0.0
    
    def run_continuous_collection(self, interval_minutes: int = 10):
        """Run continuous news collection every N minutes"""
        self.logger.info(f"Starting continuous news collection every {interval_minutes} minutes")
        
        while True:
            try:
                # Collect news
                news_items = self.collect_all_news()
                
                # Save to database
                if news_items:
                    self.save_news_to_database(news_items)
                    
                    # Aggregate sentiment for major symbols
                    major_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP']
                    for symbol in major_symbols:
                        self.aggregate_sentiment_by_minute(symbol)
                
                # Wait for next collection
                self.logger.info(f"Waiting {interval_minutes} minutes for next collection...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Stopping news collection")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous collection: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    # Initialize collector
    collector = NewsSentimentCollector()
    
    # Run single collection for testing
    news_items = collector.collect_all_news()
    if news_items:
        collector.save_news_to_database(news_items)
        
        # Aggregate sentiment
        for symbol in ['BTC', 'ETH', 'BNB']:
            collector.aggregate_sentiment_by_minute(symbol)
        
        print(f"Collected {len(news_items)} news items")
        print("Sample sentiment scores:")
        for item in news_items[:5]:
            print(f"  {item['title'][:50]}... | Sentiment: {item['sentiment_score']:.3f} ({item['sentiment_label']})")