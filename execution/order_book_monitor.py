"""
Order Book Monitor - Real-time Market Microstructure Analysis
Monitors bid/ask imbalance, spread, and spoofing detection for smart execution
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
import threading
import sqlite3
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class OrderBookMonitor:
    """Monitor order book depth and detect market microstructure anomalies"""
    
    def __init__(self):
        self.exchange = None
        self.is_monitoring = False
        self.order_book_data = {}
        self.spread_threshold = 0.003  # 0.3% spread threshold
        self.imbalance_threshold = 0.7  # 70% bid/ask imbalance threshold
        self.spoofing_detection_window = 60  # seconds
        
        # Market quality metrics
        self.market_quality = {}
        self.quality_history = []
        
        self.setup_database()
        self.initialize_exchange()
    
    def setup_database(self):
        """Initialize order book monitoring database"""
        try:
            conn = sqlite3.connect('order_book_monitor.db')
            cursor = conn.cursor()
            
            # Order book snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_book_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    bid_price REAL,
                    ask_price REAL,
                    bid_volume REAL,
                    ask_volume REAL,
                    spread_pct REAL,
                    bid_ask_imbalance REAL,
                    depth_score REAL
                )
            ''')
            
            # Market quality assessments
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    overall_quality TEXT,
                    spread_quality TEXT,
                    liquidity_quality TEXT,
                    stability_quality TEXT,
                    spoofing_risk TEXT,
                    execution_recommendation TEXT
                )
            ''')
            
            # Spoofing detection logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spoofing_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    alert_type TEXT,
                    confidence REAL,
                    details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Order book monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            import os
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            self.exchange.load_markets()
            logger.info("Order book monitor connected to OKX")
            
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Fetch order book for symbol"""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            if not order_book or not order_book.get('bids') or not order_book.get('asks'):
                return None
            
            # Calculate metrics
            best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
            best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
            
            if best_bid == 0 or best_ask == 0:
                return None
            
            # Calculate bid/ask volumes
            bid_volume = sum([bid[1] for bid in order_book['bids'][:5]])  # Top 5 levels
            ask_volume = sum([ask[1] for ask in order_book['asks'][:5]])  # Top 5 levels
            
            # Calculate spread
            spread_pct = (best_ask - best_bid) / best_bid
            
            # Calculate bid/ask imbalance
            total_volume = bid_volume + ask_volume
            bid_imbalance = bid_volume / total_volume if total_volume > 0 else 0.5
            
            # Calculate depth score (liquidity measure)
            depth_score = min(10.0, np.log1p(total_volume))
            
            metrics = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'spread_pct': spread_pct,
                'bid_imbalance': bid_imbalance,
                'depth_score': depth_score,
                'order_book': order_book
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Order book fetch failed for {symbol}: {e}")
            return None
    
    def analyze_market_quality(self, symbol: str, order_book_data: Dict) -> Dict:
        """Analyze overall market quality for execution"""
        try:
            spread_pct = order_book_data['spread_pct']
            bid_imbalance = order_book_data['bid_imbalance']
            depth_score = order_book_data['depth_score']
            
            # Spread quality assessment
            if spread_pct < 0.001:  # < 0.1%
                spread_quality = 'excellent'
            elif spread_pct < 0.002:  # < 0.2%
                spread_quality = 'good'
            elif spread_pct < 0.005:  # < 0.5%
                spread_quality = 'fair'
            else:
                spread_quality = 'poor'
            
            # Liquidity quality assessment
            if depth_score > 7:
                liquidity_quality = 'excellent'
            elif depth_score > 5:
                liquidity_quality = 'good'
            elif depth_score > 3:
                liquidity_quality = 'fair'
            else:
                liquidity_quality = 'poor'
            
            # Imbalance assessment
            if 0.4 <= bid_imbalance <= 0.6:
                stability_quality = 'excellent'
            elif 0.3 <= bid_imbalance <= 0.7:
                stability_quality = 'good'
            elif 0.2 <= bid_imbalance <= 0.8:
                stability_quality = 'fair'
            else:
                stability_quality = 'poor'
            
            # Spoofing risk assessment
            spoofing_risk = self.detect_spoofing_patterns(order_book_data)
            
            # Overall quality and execution recommendation
            quality_scores = {
                'excellent': 4,
                'good': 3,
                'fair': 2,
                'poor': 1
            }
            
            avg_score = (quality_scores[spread_quality] + 
                        quality_scores[liquidity_quality] + 
                        quality_scores[stability_quality]) / 3
            
            if avg_score >= 3.5:
                overall_quality = 'excellent'
                execution_recommendation = 'immediate'
            elif avg_score >= 2.5:
                overall_quality = 'good'
                execution_recommendation = 'normal'
            elif avg_score >= 1.5:
                overall_quality = 'fair'
                execution_recommendation = 'cautious'
            else:
                overall_quality = 'poor'
                execution_recommendation = 'delay'
            
            # Additional checks
            if spread_pct > self.spread_threshold:
                execution_recommendation = 'delay'
            
            if spoofing_risk == 'high':
                execution_recommendation = 'delay'
            
            quality_assessment = {
                'symbol': symbol,
                'overall_quality': overall_quality,
                'spread_quality': spread_quality,
                'liquidity_quality': liquidity_quality,
                'stability_quality': stability_quality,
                'spoofing_risk': spoofing_risk,
                'execution_recommendation': execution_recommendation,
                'spread_pct': spread_pct,
                'bid_imbalance': bid_imbalance,
                'depth_score': depth_score
            }
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Market quality analysis failed: {e}")
            return {
                'symbol': symbol,
                'overall_quality': 'unknown',
                'execution_recommendation': 'cautious'
            }
    
    def detect_spoofing_patterns(self, order_book_data: Dict) -> str:
        """Detect potential spoofing patterns in order book"""
        try:
            order_book = order_book_data.get('order_book', {})
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 5 or len(asks) < 5:
                return 'low'
            
            spoofing_indicators = 0
            
            # Check for abnormally large orders at best prices
            best_bid_size = bids[0][1]
            best_ask_size = asks[0][1]
            
            # Calculate average size of next 4 levels
            avg_bid_size = np.mean([bid[1] for bid in bids[1:5]])
            avg_ask_size = np.mean([ask[1] for ask in asks[1:5]])
            
            # Spoofing indicator 1: Best level much larger than others
            if best_bid_size > avg_bid_size * 5:
                spoofing_indicators += 1
            
            if best_ask_size > avg_ask_size * 5:
                spoofing_indicators += 1
            
            # Spoofing indicator 2: Large gaps in order book
            bid_gaps = 0
            ask_gaps = 0
            
            for i in range(1, min(5, len(bids))):
                price_gap = (bids[i-1][0] - bids[i][0]) / bids[i][0]
                if price_gap > 0.01:  # 1% gap
                    bid_gaps += 1
            
            for i in range(1, min(5, len(asks))):
                price_gap = (asks[i][0] - asks[i-1][0]) / asks[i-1][0]
                if price_gap > 0.01:  # 1% gap
                    ask_gaps += 1
            
            if bid_gaps >= 2 or ask_gaps >= 2:
                spoofing_indicators += 1
            
            # Spoofing indicator 3: Extreme imbalance with large orders
            bid_imbalance = order_book_data.get('bid_imbalance', 0.5)
            if (bid_imbalance > 0.8 and best_bid_size > avg_bid_size * 3) or \
               (bid_imbalance < 0.2 and best_ask_size > avg_ask_size * 3):
                spoofing_indicators += 1
            
            # Determine spoofing risk level
            if spoofing_indicators >= 3:
                return 'high'
            elif spoofing_indicators >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Spoofing detection failed: {e}")
            return 'low'
    
    def save_order_book_snapshot(self, data: Dict):
        """Save order book snapshot to database"""
        try:
            conn = sqlite3.connect('order_book_monitor.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO order_book_snapshots (
                    symbol, bid_price, ask_price, bid_volume, ask_volume,
                    spread_pct, bid_ask_imbalance, depth_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['symbol'],
                data['best_bid'],
                data['best_ask'],
                data['bid_volume'],
                data['ask_volume'],
                data['spread_pct'],
                data['bid_imbalance'],
                data['depth_score']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save order book snapshot: {e}")
    
    def save_market_quality(self, assessment: Dict):
        """Save market quality assessment to database"""
        try:
            conn = sqlite3.connect('order_book_monitor.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_quality (
                    symbol, overall_quality, spread_quality, liquidity_quality,
                    stability_quality, spoofing_risk, execution_recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                assessment['symbol'],
                assessment['overall_quality'],
                assessment['spread_quality'],
                assessment['liquidity_quality'],
                assessment['stability_quality'],
                assessment['spoofing_risk'],
                assessment['execution_recommendation']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save market quality: {e}")
    
    def should_execute_trade(self, symbol: str) -> Tuple[bool, str]:
        """Determine if trade should be executed based on market quality"""
        try:
            # Get recent order book data
            order_book_data = self.fetch_order_book(symbol)
            if not order_book_data:
                return False, "Order book data unavailable"
            
            # Analyze market quality
            quality_assessment = self.analyze_market_quality(symbol, order_book_data)
            
            # Save data
            self.save_order_book_snapshot(order_book_data)
            self.save_market_quality(quality_assessment)
            
            recommendation = quality_assessment['execution_recommendation']
            
            if recommendation == 'immediate':
                return True, "Excellent market conditions"
            elif recommendation == 'normal':
                return True, "Good market conditions"
            elif recommendation == 'cautious':
                return True, "Fair market conditions - proceed with caution"
            else:  # delay
                return False, f"Poor market conditions: {quality_assessment['overall_quality']}"
            
        except Exception as e:
            logger.error(f"Trade execution assessment failed: {e}")
            return True, "Assessment failed - defaulting to execute"
    
    def get_market_quality_summary(self, symbols: List[str]) -> Dict:
        """Get market quality summary for multiple symbols"""
        try:
            summary = {
                'excellent': 0,
                'good': 0,
                'fair': 0,
                'poor': 0,
                'total_assessed': 0
            }
            
            details = {}
            
            for symbol in symbols[:10]:  # Limit to prevent rate limiting
                try:
                    order_book_data = self.fetch_order_book(symbol)
                    if order_book_data:
                        quality = self.analyze_market_quality(symbol, order_book_data)
                        summary[quality['overall_quality']] += 1
                        summary['total_assessed'] += 1
                        
                        details[symbol] = {
                            'quality': quality['overall_quality'],
                            'spread': f"{quality['spread_pct']*100:.3f}%",
                            'recommendation': quality['execution_recommendation']
                        }
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Quality check failed for {symbol}: {e}")
                    continue
            
            return {
                'summary': summary,
                'details': details,
                'assessment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market quality summary failed: {e}")
            return {'summary': {}, 'details': {}}
    
    def get_monitoring_insights(self) -> List[str]:
        """Get insights about order book monitoring"""
        try:
            insights = []
            
            # Get recent quality assessments
            conn = sqlite3.connect('order_book_monitor.db')
            recent_quality = pd.read_sql_query('''
                SELECT overall_quality, COUNT(*) as count
                FROM market_quality 
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY overall_quality
            ''', conn)
            
            if not recent_quality.empty:
                total = recent_quality['count'].sum()
                for _, row in recent_quality.iterrows():
                    pct = (row['count'] / total) * 100
                    insights.append(f"{row['overall_quality'].title()} market quality: {pct:.1f}% of assessments")
            
            # Check for high spread conditions
            high_spread = pd.read_sql_query('''
                SELECT COUNT(*) as count FROM order_book_snapshots 
                WHERE timestamp > datetime('now', '-1 hour') AND spread_pct > 0.005
            ''', conn).iloc[0]['count']
            
            if high_spread > 0:
                insights.append(f"High spread conditions detected in {high_spread} recent snapshots")
            
            conn.close()
            
            return insights if insights else ["Order book monitoring active - collecting data..."]
            
        except Exception as e:
            logger.error(f"Monitoring insights failed: {e}")
            return ["Order book analysis temporarily unavailable"]