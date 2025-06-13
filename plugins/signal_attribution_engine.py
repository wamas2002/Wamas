"""
Signal Attribution Engine
Tracks and logs the origin of each trading signal for performance analysis
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SignalAttribution:
    signal_id: str
    symbol: str
    signal_type: str
    origin_source: str
    sub_indicators: List[str]
    confidence: float
    timestamp: datetime
    outcome: Optional[str] = None
    profit_loss: Optional[float] = None

class SignalAttributionEngine:
    def __init__(self):
        self.log_file = "logs/signal_explanation.json"
        self.db_path = "attribution.db"
        self.ensure_directories()
        self.setup_database()
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plugins", exist_ok=True)
    
    def setup_database(self):
        """Setup attribution tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_attribution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    origin_source TEXT NOT NULL,
                    sub_indicators TEXT,
                    confidence REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    outcome TEXT,
                    profit_loss REAL,
                    execution_time TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS source_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    successful_signals INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_profit_loss REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Signal attribution database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def log_signal_origin(self, signal_data: Dict) -> str:
        """Log the origin and composition of a trading signal"""
        try:
            signal_id = f"{signal_data['symbol']}_{int(datetime.now().timestamp())}"
            
            # Determine signal origin
            origin_source = self._identify_signal_source(signal_data)
            sub_indicators = self._extract_sub_indicators(signal_data)
            
            attribution = SignalAttribution(
                signal_id=signal_id,
                symbol=signal_data['symbol'],
                signal_type=signal_data.get('signal', 'UNKNOWN'),
                origin_source=origin_source,
                sub_indicators=sub_indicators,
                confidence=signal_data.get('confidence', 0),
                timestamp=datetime.now()
            )
            
            # Save to database
            self._save_attribution(attribution)
            
            # Save to JSON log
            self._save_to_json_log(attribution)
            
            logger.info(f"Signal attribution logged: {signal_id} from {origin_source}")
            return signal_id
            
        except Exception as e:
            logger.error(f"Failed to log signal origin: {e}")
            return ""
    
    def _identify_signal_source(self, signal_data: Dict) -> str:
        """Identify the primary source of the signal"""
        try:
            # Check for GPT enhancement
            if 'gpt_enhanced' in str(signal_data.get('entry_reasons', [])):
                return "GPT_ENHANCED"
            
            # Check for AI/ML predictions
            if signal_data.get('ai_score', 0) > 70:
                return "ML_MODEL"
            
            # Check for technical indicators
            technical_score = signal_data.get('technical_score', 0)
            if technical_score > 70:
                return "TECHNICAL_ANALYSIS"
            
            # Check for specific strategy origins
            if 'momentum' in str(signal_data.get('entry_reasons', [])).lower():
                return "MOMENTUM_STRATEGY"
            elif 'mean_reversion' in str(signal_data.get('entry_reasons', [])).lower():
                return "MEAN_REVERSION"
            elif 'trend_following' in str(signal_data.get('entry_reasons', [])).lower():
                return "TREND_FOLLOWING"
            
            # Check for futures vs spot
            if signal_data.get('strategy') == 'futures':
                return "FUTURES_ENGINE"
            
            return "HYBRID_SIGNAL"
            
        except Exception as e:
            logger.error(f"Error identifying signal source: {e}")
            return "UNKNOWN"
    
    def _extract_sub_indicators(self, signal_data: Dict) -> List[str]:
        """Extract contributing sub-indicators"""
        indicators = []
        
        try:
            entry_reasons = signal_data.get('entry_reasons', [])
            if isinstance(entry_reasons, str):
                entry_reasons = [entry_reasons]
            
            # Map technical indicators
            indicator_mapping = {
                'RSI_OVERSOLD': 'RSI',
                'RSI_OVERBOUGHT': 'RSI',
                'MACD_BULLISH': 'MACD',
                'MACD_BEARISH': 'MACD',
                'BB_OVERSOLD': 'Bollinger Bands',
                'BB_OVERBOUGHT': 'Bollinger Bands',
                'VOLUME_SURGE': 'Volume',
                'HIGH_VOLUME': 'Volume',
                'RESISTANCE_BREAKOUT': 'Support/Resistance',
                'SUPPORT_BREAKDOWN': 'Support/Resistance',
                'EMA_BULLISH': 'EMA',
                'EMA_BEARISH': 'EMA',
                'STRONG_TREND': 'Trend Analysis',
                'ADX_STRONG': 'ADX'
            }
            
            for reason in entry_reasons:
                if reason in indicator_mapping:
                    indicators.append(indicator_mapping[reason])
            
            # Add additional metrics
            if signal_data.get('rsi'):
                indicators.append('RSI')
            if signal_data.get('volume_ratio', 0) > 1.5:
                indicators.append('Volume Surge')
                
            return list(set(indicators))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting indicators: {e}")
            return ['Unknown']
    
    def _save_attribution(self, attribution: SignalAttribution):
        """Save attribution to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO signal_attribution 
                (signal_id, symbol, signal_type, origin_source, sub_indicators, 
                 confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                attribution.signal_id,
                attribution.symbol,
                attribution.signal_type,
                attribution.origin_source,
                json.dumps(attribution.sub_indicators),
                attribution.confidence,
                attribution.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save attribution to database: {e}")
    
    def _save_to_json_log(self, attribution: SignalAttribution):
        """Save attribution to JSON log file"""
        try:
            log_entry = {
                'signal_id': attribution.signal_id,
                'symbol': attribution.symbol,
                'signal_type': attribution.signal_type,
                'origin_source': attribution.origin_source,
                'sub_indicators': attribution.sub_indicators,
                'confidence': attribution.confidence,
                'timestamp': attribution.timestamp.isoformat(),
                'outcome': attribution.outcome,
                'profit_loss': attribution.profit_loss
            }
            
            # Read existing logs
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            
            # Add new entry
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save to JSON log: {e}")
    
    def update_signal_outcome(self, signal_id: str, outcome: str, profit_loss: float):
        """Update signal outcome after trade completion"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE signal_attribution 
                SET outcome = ?, profit_loss = ?, execution_time = ?
                WHERE signal_id = ?
            ''', (outcome, profit_loss, datetime.now().isoformat(), signal_id))
            
            conn.commit()
            conn.close()
            
            # Update JSON log
            self._update_json_outcome(signal_id, outcome, profit_loss)
            
            # Update source performance
            self._update_source_performance(signal_id)
            
            logger.info(f"Signal outcome updated: {signal_id} -> {outcome}")
            
        except Exception as e:
            logger.error(f"Failed to update signal outcome: {e}")
    
    def _update_json_outcome(self, signal_id: str, outcome: str, profit_loss: float):
        """Update outcome in JSON log"""
        try:
            if not os.path.exists(self.log_file):
                return
                
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            for log in logs:
                if log['signal_id'] == signal_id:
                    log['outcome'] = outcome
                    log['profit_loss'] = profit_loss
                    break
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update JSON outcome: {e}")
    
    def _update_source_performance(self, signal_id: str):
        """Update performance metrics for signal source"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get signal details
            cursor.execute('''
                SELECT origin_source, outcome, profit_loss 
                FROM signal_attribution 
                WHERE signal_id = ?
            ''', (signal_id,))
            
            result = cursor.fetchone()
            if not result:
                return
                
            source_name, outcome, profit_loss = result
            
            # Update source performance
            cursor.execute('''
                INSERT OR IGNORE INTO source_performance 
                (source_name, total_signals, successful_signals, win_rate, avg_profit_loss)
                VALUES (?, 0, 0, 0, 0)
            ''', (source_name,))
            
            # Get current stats
            cursor.execute('''
                SELECT total_signals, successful_signals, avg_profit_loss 
                FROM source_performance 
                WHERE source_name = ?
            ''', (source_name,))
            
            stats = cursor.fetchone()
            if stats:
                total, successful, avg_pl = stats
                
                new_total = total + 1
                new_successful = successful + (1 if outcome == 'WIN' else 0)
                new_win_rate = (new_successful / new_total) * 100 if new_total > 0 else 0
                new_avg_pl = ((avg_pl * total) + (profit_loss or 0)) / new_total
                
                cursor.execute('''
                    UPDATE source_performance 
                    SET total_signals = ?, successful_signals = ?, 
                        win_rate = ?, avg_profit_loss = ?, last_updated = ?
                    WHERE source_name = ?
                ''', (new_total, new_successful, new_win_rate, new_avg_pl, 
                     datetime.now().isoformat(), source_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update source performance: {e}")
    
    def get_source_performance(self) -> Dict:
        """Get performance metrics by signal source"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT source_name, total_signals, successful_signals, 
                       win_rate, avg_profit_loss, last_updated
                FROM source_performance
                ORDER BY win_rate DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            performance = {}
            for row in results:
                source, total, successful, win_rate, avg_pl, updated = row
                performance[source] = {
                    'total_signals': total,
                    'successful_signals': successful,
                    'win_rate': win_rate,
                    'avg_profit_loss': avg_pl,
                    'last_updated': updated
                }
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to get source performance: {e}")
            return {}
    
    def get_recent_attributions(self, limit: int = 50) -> List[Dict]:
        """Get recent signal attributions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT signal_id, symbol, signal_type, origin_source, 
                       sub_indicators, confidence, timestamp, outcome, profit_loss
                FROM signal_attribution
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            attributions = []
            for row in results:
                attributions.append({
                    'signal_id': row[0],
                    'symbol': row[1],
                    'signal_type': row[2],
                    'origin_source': row[3],
                    'sub_indicators': json.loads(row[4]) if row[4] else [],
                    'confidence': row[5],
                    'timestamp': row[6],
                    'outcome': row[7],
                    'profit_loss': row[8]
                })
            
            return attributions
            
        except Exception as e:
            logger.error(f"Failed to get recent attributions: {e}")
            return []

# Global instance
attribution_engine = SignalAttributionEngine()