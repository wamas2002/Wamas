"""
GPT-Enhanced Trading Analyzer
Integrates ChatGPT API for advanced market sentiment analysis and trading signal interpretation
"""

import openai
import os
import json
import logging
import sqlite3
import pandas as pd
from datetime import datetime
import ccxt
import requests
from typing import Dict, List, Optional

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user

logger = logging.getLogger(__name__)

class GPTEnhancedTradingAnalyzer:
    def __init__(self):
        """Initialize GPT-Enhanced Trading Analyzer"""
        self.openai_client = None
        self.exchange = None
        self.initialize_openai()
        self.initialize_exchange()
        self.setup_database()
        
    def initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            openai_key = os.environ.get('OPENAI_API_KEY')
            if not openai_key:
                logger.error("OPENAI_API_KEY not found in environment")
                return False
                
            self.openai_client = openai.OpenAI(api_key=openai_key)
            logger.info("GPT analyzer connected to OpenAI API")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            return False
    
    def initialize_exchange(self):
        """Initialize OKX exchange"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            return True
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            return False
    
    def setup_database(self):
        """Setup GPT analysis database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS gpt_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        market_data TEXT NOT NULL,
                        gpt_analysis TEXT NOT NULL,
                        confidence_adjustment REAL,
                        trading_recommendation TEXT,
                        risk_assessment TEXT,
                        sentiment_score REAL,
                        timestamp TEXT NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS enhanced_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        original_confidence REAL,
                        gpt_enhanced_confidence REAL,
                        gpt_reasoning TEXT,
                        market_context TEXT,
                        risk_factors TEXT,
                        entry_strategy TEXT,
                        exit_strategy TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                logger.info("GPT analysis database initialized")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_context(self, symbol: str) -> Dict:
        """Gather comprehensive market context for GPT analysis"""
        try:
            # Get current market data
            ticker = self.exchange.fetch_ticker(symbol)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=24)
            
            # Calculate price movements
            current_price = ticker['last']
            price_24h_ago = ohlcv[0][4] if ohlcv else current_price
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
            
            # Volume analysis
            avg_volume = sum([candle[5] for candle in ohlcv[-10:]]) / 10 if ohlcv else 0
            current_volume = ticker['quoteVolume'] or 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Get recent price action
            recent_highs = [candle[2] for candle in ohlcv[-5:]] if ohlcv else [current_price]
            recent_lows = [candle[3] for candle in ohlcv[-5:]] if ohlcv else [current_price]
            volatility = (max(recent_highs) - min(recent_lows)) / current_price * 100
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'bid_ask_spread': (ticker['ask'] - ticker['bid']) / ticker['bid'] * 100 if ticker['bid'] and ticker['ask'] else 0,
                'market_cap_rank': self.get_market_cap_rank(symbol),
                'recent_price_action': 'bullish' if price_change_24h > 2 else 'bearish' if price_change_24h < -2 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Market context gathering failed for {symbol}: {e}")
            return {}
    
    def get_market_cap_rank(self, symbol: str) -> str:
        """Get approximate market cap ranking"""
        base_currency = symbol.split('/')[0]
        major_coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE']
        mid_cap = ['LINK', 'LTC', 'DOT', 'AVAX', 'UNI', 'ATOM']
        
        if base_currency in major_coins:
            return 'large_cap'
        elif base_currency in mid_cap:
            return 'mid_cap'
        else:
            return 'small_cap'
    
    def analyze_signal_with_gpt(self, symbol: str, technical_confidence: float, 
                               ai_score: float, signal_type: str) -> Dict:
        """Use GPT to analyze and enhance trading signals"""
        try:
            if not self.openai_client:
                return self.get_default_analysis(symbol, technical_confidence)
            
            # Get market context
            market_context = self.get_market_context(symbol)
            
            # Prepare prompt for GPT analysis
            prompt = f"""
            As an expert cryptocurrency trading analyst, analyze this trading signal and provide detailed insights:

            TRADING SIGNAL DATA:
            - Symbol: {symbol}
            - Signal Type: {signal_type}
            - Technical Confidence: {technical_confidence}%
            - AI Score: {ai_score}%
            - Current Price: ${market_context.get('current_price', 'N/A')}
            - 24h Price Change: {market_context.get('price_change_24h', 0):.2f}%
            - Volume Ratio: {market_context.get('volume_ratio', 1):.2f}x
            - Volatility: {market_context.get('volatility', 0):.2f}%
            - Market Cap Category: {market_context.get('market_cap_rank', 'unknown')}
            - Recent Price Action: {market_context.get('recent_price_action', 'neutral')}

            ANALYSIS REQUIRED:
            1. Overall signal quality assessment (1-100 scale)
            2. Key risk factors to consider
            3. Market timing evaluation
            4. Confidence adjustment recommendation (+/- 0-20 points)
            5. Entry strategy suggestions
            6. Exit strategy recommendations
            7. Position sizing recommendation (conservative/moderate/aggressive)

            Provide your analysis in JSON format:
            {{
                "signal_quality": <1-100>,
                "confidence_adjustment": <-20 to +20>,
                "risk_level": "<low/medium/high>",
                "market_timing": "<excellent/good/fair/poor>",
                "entry_strategy": "<strategy description>",
                "exit_strategy": "<exit plan>",
                "position_sizing": "<conservative/moderate/aggressive>",
                "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
                "reasoning": "<detailed analysis explanation>"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading analyst with deep knowledge of technical analysis, market psychology, and risk management."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Calculate enhanced confidence
            original_confidence = technical_confidence
            confidence_adjustment = analysis.get('confidence_adjustment', 0)
            enhanced_confidence = max(0, min(100, original_confidence + confidence_adjustment))
            
            # Save analysis to database
            self.save_gpt_analysis(symbol, market_context, analysis, enhanced_confidence)
            
            return {
                'original_confidence': original_confidence,
                'enhanced_confidence': enhanced_confidence,
                'confidence_adjustment': confidence_adjustment,
                'risk_level': analysis.get('risk_level', 'medium'),
                'market_timing': analysis.get('market_timing', 'fair'),
                'entry_strategy': analysis.get('entry_strategy', 'Standard entry'),
                'exit_strategy': analysis.get('exit_strategy', 'Standard exit'),
                'position_sizing': analysis.get('position_sizing', 'moderate'),
                'key_risks': analysis.get('key_risks', []),
                'reasoning': analysis.get('reasoning', 'Standard analysis'),
                'signal_quality': analysis.get('signal_quality', 50)
            }
            
        except Exception as e:
            logger.error(f"GPT analysis failed for {symbol}: {e}")
            return self.get_default_analysis(symbol, technical_confidence)
    
    def get_default_analysis(self, symbol: str, confidence: float) -> Dict:
        """Fallback analysis when GPT is unavailable"""
        return {
            'original_confidence': confidence,
            'enhanced_confidence': confidence,
            'confidence_adjustment': 0,
            'risk_level': 'medium',
            'market_timing': 'fair',
            'entry_strategy': 'Market entry with standard position sizing',
            'exit_strategy': '8% stop loss, 15% take profit',
            'position_sizing': 'moderate',
            'key_risks': ['Market volatility', 'Liquidity risk'],
            'reasoning': 'Standard technical analysis without GPT enhancement',
            'signal_quality': min(confidence, 75)
        }
    
    def save_gpt_analysis(self, symbol: str, market_context: Dict, analysis: Dict, enhanced_confidence: float):
        """Save GPT analysis to database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO gpt_analysis (
                        symbol, analysis_type, market_data, gpt_analysis,
                        confidence_adjustment, trading_recommendation, risk_assessment,
                        sentiment_score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, 'signal_enhancement', json.dumps(market_context),
                    json.dumps(analysis), analysis.get('confidence_adjustment', 0),
                    analysis.get('entry_strategy', ''), analysis.get('risk_level', 'medium'),
                    analysis.get('signal_quality', 50), datetime.now().isoformat()
                ))
                
                cursor.execute('''
                    INSERT INTO enhanced_signals (
                        symbol, original_confidence, gpt_enhanced_confidence,
                        gpt_reasoning, market_context, risk_factors,
                        entry_strategy, exit_strategy, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, analysis.get('original_confidence', 0), enhanced_confidence,
                    analysis.get('reasoning', ''), json.dumps(market_context),
                    json.dumps(analysis.get('key_risks', [])),
                    analysis.get('entry_strategy', ''), analysis.get('exit_strategy', ''),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save GPT analysis: {e}")
    
    def analyze_market_sentiment(self, symbols: List[str]) -> Dict:
        """Analyze overall market sentiment using GPT"""
        try:
            if not self.openai_client:
                return {'sentiment': 'neutral', 'confidence': 50}
            
            # Gather market data for multiple symbols
            market_overview = {}
            for symbol in symbols[:5]:  # Analyze top 5 symbols
                try:
                    context = self.get_market_context(symbol)
                    market_overview[symbol] = context
                except:
                    continue
            
            if not market_overview:
                return {'sentiment': 'neutral', 'confidence': 50}
            
            prompt = f"""
            Analyze the current cryptocurrency market sentiment based on this data:

            MARKET DATA:
            {json.dumps(market_overview, indent=2)}

            Provide a comprehensive market sentiment analysis in JSON format:
            {{
                "overall_sentiment": "<bullish/bearish/neutral>",
                "sentiment_strength": <1-100>,
                "market_regime": "<trending/ranging/volatile>",
                "risk_level": "<low/medium/high>",
                "trading_recommendation": "<aggressive/moderate/conservative>",
                "key_observations": ["<obs1>", "<obs2>", "<obs3>"],
                "market_outlook": "<short analysis of market direction>"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency market analyst."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.3
            )
            
            sentiment_analysis = json.loads(response.choices[0].message.content)
            
            # Save market sentiment analysis
            self.save_market_sentiment(sentiment_analysis)
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 50}
    
    def save_market_sentiment(self, sentiment_data: Dict):
        """Save market sentiment analysis"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO gpt_analysis (
                        symbol, analysis_type, market_data, gpt_analysis,
                        sentiment_score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    'MARKET_OVERVIEW', 'sentiment_analysis', '',
                    json.dumps(sentiment_data), sentiment_data.get('sentiment_strength', 50),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save market sentiment: {e}")
    
    def get_enhanced_trading_recommendations(self) -> List[Dict]:
        """Get recent GPT-enhanced trading recommendations"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                df = pd.read_sql_query('''
                    SELECT * FROM enhanced_signals 
                    WHERE timestamp > datetime('now', '-1 hour')
                    ORDER BY gpt_enhanced_confidence DESC
                    LIMIT 10
                ''', conn)
                
                return df.to_dict('records') if not df.empty else []
                
        except Exception as e:
            logger.error(f"Failed to get enhanced recommendations: {e}")
            return []

def test_gpt_enhancement():
    """Test GPT enhancement functionality"""
    try:
        analyzer = GPTEnhancedTradingAnalyzer()
        
        # Test signal analysis
        test_symbol = 'BTC/USDT'
        enhanced_analysis = analyzer.analyze_signal_with_gpt(
            symbol=test_symbol,
            technical_confidence=65.0,
            ai_score=70.0,
            signal_type='BUY'
        )
        
        print(f"GPT Enhanced Analysis for {test_symbol}:")
        print(f"Original Confidence: {enhanced_analysis['original_confidence']}%")
        print(f"Enhanced Confidence: {enhanced_analysis['enhanced_confidence']}%")
        print(f"Risk Level: {enhanced_analysis['risk_level']}")
        print(f"Entry Strategy: {enhanced_analysis['entry_strategy']}")
        print(f"Reasoning: {enhanced_analysis['reasoning'][:100]}...")
        
        # Test market sentiment
        market_sentiment = analyzer.analyze_market_sentiment(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
        print(f"\nMarket Sentiment: {market_sentiment}")
        
        return True
        
    except Exception as e:
        logger.error(f"GPT enhancement test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gpt_enhancement()