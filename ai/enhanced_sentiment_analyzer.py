"""
Enhanced Sentiment Analyzer Integration
Integrates multiple sentiment data sources for comprehensive market sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import sqlite3
import time
from trafilatura import fetch_url, extract
import warnings
warnings.filterwarnings('ignore')

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with multiple data sources"""
    
    def __init__(self, db_path: str = "data/sentiment_data.db"):
        self.db_path = db_path
        self.setup_database()
        
        # API endpoints
        self.cryptopanic_url = "https://cryptopanic.com/api/v1/posts/"
        self.fear_greed_url = "https://api.alternative.me/fng/"
        
        # News sources
        self.news_sources = {
            'coindesk': 'https://www.coindesk.com/',
            'cointelegraph': 'https://cointelegraph.com/',
            'decrypt': 'https://decrypt.co/',
            'bitcoinist': 'https://bitcoinist.com/'
        }
        
    def setup_database(self):
        """Setup sentiment database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sentiment scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                source TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                volume_mentions INTEGER DEFAULT 0,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Fear & Greed index
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fear_greed_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL UNIQUE,
                value INTEGER NOT NULL,
                classification TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # News sentiment
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT UNIQUE,
                source TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                published_at INTEGER NOT NULL,
                symbols_mentioned TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis for a symbol"""
        
        # Get recent sentiment data
        recent_sentiment = self._get_recent_sentiment(symbol)
        
        # Get fear & greed index
        fear_greed = self._get_fear_greed_index()
        
        # Get news sentiment
        news_sentiment = self._get_news_sentiment(symbol)
        
        # Combine all sentiment sources
        combined_sentiment = self._combine_sentiment_sources(
            recent_sentiment, fear_greed, news_sentiment
        )
        
        return {
            'symbol': symbol,
            'overall_sentiment': combined_sentiment['overall'],
            'confidence': combined_sentiment['confidence'],
            'sentiment_breakdown': {
                'social': recent_sentiment,
                'fear_greed': fear_greed,
                'news': news_sentiment
            },
            'sentiment_classification': self._classify_sentiment(combined_sentiment['overall']),
            'timestamp': datetime.now()
        }
    
    def _get_recent_sentiment(self, symbol: str, hours: int = 24) -> float:
        """Get recent sentiment data from database"""
        conn = sqlite3.connect(self.db_path)
        
        since_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        query = '''
            SELECT sentiment_score, confidence, volume_mentions
            FROM sentiment_scores 
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, since_timestamp))
        conn.close()
        
        if df.empty:
            return 0.5  # Neutral sentiment
        
        # Weight by confidence and volume
        weights = df['confidence'] * np.log1p(df['volume_mentions'])
        if weights.sum() > 0:
            weighted_sentiment = (df['sentiment_score'] * weights).sum() / weights.sum()
        else:
            weighted_sentiment = df['sentiment_score'].mean()
        
        return float(weighted_sentiment)
    
    def _get_fear_greed_index(self) -> float:
        """Get current Fear & Greed index"""
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                current_value = int(data['data'][0]['value'])
                
                # Store in database
                self._store_fear_greed(current_value, data['data'][0]['value_classification'])
                
                # Convert to 0-1 scale (0 = extreme fear, 1 = extreme greed)
                return current_value / 100.0
            
        except Exception as e:
            print(f"Error fetching Fear & Greed index: {e}")
        
        # Get latest from database as fallback
        return self._get_latest_fear_greed()
    
    def _get_latest_fear_greed(self) -> float:
        """Get latest Fear & Greed index from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT value FROM fear_greed_index 
            ORDER BY timestamp DESC LIMIT 1
        '''
        
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0] / 100.0
        return 0.5  # Neutral if no data
    
    def _store_fear_greed(self, value: int, classification: str):
        """Store Fear & Greed index in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = int(datetime.now().timestamp())
        
        cursor.execute('''
            INSERT OR REPLACE INTO fear_greed_index 
            (timestamp, value, classification) VALUES (?, ?, ?)
        ''', (timestamp, value, classification))
        
        conn.commit()
        conn.close()
    
    def _get_news_sentiment(self, symbol: str, hours: int = 24) -> float:
        """Get news sentiment for symbol"""
        conn = sqlite3.connect(self.db_path)
        
        since_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        query = '''
            SELECT sentiment_score FROM news_sentiment 
            WHERE symbols_mentioned LIKE ? AND published_at >= ?
            ORDER BY published_at DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(f'%{symbol}%', since_timestamp))
        conn.close()
        
        if df.empty:
            return 0.5  # Neutral sentiment
        
        return float(df['sentiment_score'].mean())
    
    def collect_cryptopanic_sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """Collect sentiment from CryptoPanic API"""
        sentiment_scores = {}
        
        for symbol in symbols:
            try:
                # Clean symbol for API call
                clean_symbol = symbol.replace('USDT', '').replace('USD', '')
                
                params = {
                    'auth_token': 'free',  # Using free tier
                    'public': 'true',
                    'currencies': clean_symbol,
                    'filter': 'hot'
                }
                
                response = requests.get(self.cryptopanic_url, params=params, timeout=10)
                data = response.json()
                
                if 'results' in data:
                    posts = data['results']
                    sentiment_score = self._calculate_cryptopanic_sentiment(posts)
                    sentiment_scores[symbol] = sentiment_score
                    
                    # Store in database
                    self._store_sentiment_score(
                        symbol, 'cryptopanic', sentiment_score, 0.7, len(posts)
                    )
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error collecting CryptoPanic sentiment for {symbol}: {e}")
                sentiment_scores[symbol] = 0.5
        
        return sentiment_scores
    
    def _calculate_cryptopanic_sentiment(self, posts: List[Dict]) -> float:
        """Calculate sentiment from CryptoPanic posts"""
        if not posts:
            return 0.5
        
        positive_keywords = ['bullish', 'moon', 'pump', 'rally', 'surge', 'breakout', 'gain']
        negative_keywords = ['bearish', 'dump', 'crash', 'drop', 'fall', 'decline', 'sell']
        
        sentiment_scores = []
        
        for post in posts:
            title = post.get('title', '').lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in title)
            negative_count = sum(1 for keyword in negative_keywords if keyword in title)
            
            if positive_count > negative_count:
                sentiment_scores.append(0.7)
            elif negative_count > positive_count:
                sentiment_scores.append(0.3)
            else:
                sentiment_scores.append(0.5)
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.5
    
    def collect_news_sentiment(self):
        """Collect sentiment from news sources"""
        for source_name, source_url in self.news_sources.items():
            try:
                self._scrape_news_source(source_name, source_url)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"Error scraping {source_name}: {e}")
    
    def _scrape_news_source(self, source_name: str, source_url: str):
        """Scrape news from a specific source"""
        try:
            # Fetch the webpage
            downloaded = fetch_url(source_url)
            if not downloaded:
                return
            
            # Extract text content
            text = extract(downloaded)
            if not text:
                return
            
            # Simple sentiment analysis based on keywords
            sentiment_score = self._analyze_text_sentiment(text)
            
            # Store news sentiment
            self._store_news_sentiment(
                title=f"{source_name} daily sentiment",
                url=source_url,
                source=source_name,
                sentiment_score=sentiment_score,
                symbols_mentioned="BTC,ETH,crypto"
            )
            
        except Exception as e:
            print(f"Error scraping {source_name}: {e}")
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_keywords = [
            'bullish', 'positive', 'optimistic', 'growth', 'increase', 'rise',
            'surge', 'rally', 'boom', 'success', 'profit', 'gain', 'moon'
        ]
        
        negative_keywords = [
            'bearish', 'negative', 'pessimistic', 'decline', 'decrease', 'fall',
            'crash', 'dump', 'loss', 'fear', 'sell', 'drop', 'bear'
        ]
        
        positive_count = sum(text_lower.count(keyword) for keyword in positive_keywords)
        negative_count = sum(text_lower.count(keyword) for keyword in negative_keywords)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.5  # Neutral
        
        sentiment_score = positive_count / total_sentiment_words
        return min(max(sentiment_score, 0.0), 1.0)
    
    def _store_sentiment_score(self, symbol: str, source: str, sentiment_score: float, 
                             confidence: float, volume_mentions: int):
        """Store sentiment score in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = int(datetime.now().timestamp())
        
        cursor.execute('''
            INSERT INTO sentiment_scores 
            (symbol, timestamp, source, sentiment_score, confidence, volume_mentions)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, timestamp, source, sentiment_score, confidence, volume_mentions))
        
        conn.commit()
        conn.close()
    
    def _store_news_sentiment(self, title: str, url: str, source: str, 
                            sentiment_score: float, symbols_mentioned: str):
        """Store news sentiment in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        published_at = int(datetime.now().timestamp())
        
        cursor.execute('''
            INSERT OR IGNORE INTO news_sentiment 
            (title, url, source, sentiment_score, published_at, symbols_mentioned)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, url, source, sentiment_score, published_at, symbols_mentioned))
        
        conn.commit()
        conn.close()
    
    def _combine_sentiment_sources(self, social_sentiment: float, fear_greed: float, 
                                 news_sentiment: float) -> Dict[str, float]:
        """Combine sentiment from multiple sources"""
        
        # Weights for different sources
        weights = {
            'social': 0.4,
            'fear_greed': 0.3,
            'news': 0.3
        }
        
        # Calculate weighted sentiment
        overall_sentiment = (
            social_sentiment * weights['social'] +
            fear_greed * weights['fear_greed'] +
            news_sentiment * weights['news']
        )
        
        # Calculate confidence based on source agreement
        sentiments = [social_sentiment, fear_greed, news_sentiment]
        confidence = 1.0 - np.std(sentiments)  # Higher std = lower confidence
        confidence = max(0.3, min(1.0, confidence))  # Clamp between 0.3 and 1.0
        
        return {
            'overall': overall_sentiment,
            'confidence': confidence
        }
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment score into categories"""
        if sentiment_score >= 0.7:
            return "Very Positive"
        elif sentiment_score >= 0.6:
            return "Positive"
        elif sentiment_score >= 0.4:
            return "Neutral"
        elif sentiment_score >= 0.3:
            return "Negative"
        else:
            return "Very Negative"
    
    def get_market_sentiment_overview(self, symbols: List[str]) -> Dict[str, Any]:
        """Get overall market sentiment overview"""
        sentiment_data = {}
        
        for symbol in symbols:
            sentiment_data[symbol] = self.get_symbol_sentiment(symbol)
        
        # Calculate market-wide metrics
        overall_sentiments = [data['overall_sentiment'] for data in sentiment_data.values()]
        
        return {
            'market_sentiment': np.mean(overall_sentiments),
            'sentiment_distribution': {
                'very_positive': len([s for s in overall_sentiments if s >= 0.7]),
                'positive': len([s for s in overall_sentiments if 0.6 <= s < 0.7]),
                'neutral': len([s for s in overall_sentiments if 0.4 <= s < 0.6]),
                'negative': len([s for s in overall_sentiments if 0.3 <= s < 0.4]),
                'very_negative': len([s for s in overall_sentiments if s < 0.3])
            },
            'fear_greed_index': self._get_fear_greed_index(),
            'symbol_sentiments': sentiment_data,
            'timestamp': datetime.now()
        }
    
    def update_all_sentiment_data(self, symbols: List[str]):
        """Update sentiment data from all sources"""
        print("Updating CryptoPanic sentiment...")
        self.collect_cryptopanic_sentiment(symbols)
        
        print("Updating Fear & Greed index...")
        self._get_fear_greed_index()
        
        print("Updating news sentiment...")
        self.collect_news_sentiment()
        
        print("Sentiment data update completed.")