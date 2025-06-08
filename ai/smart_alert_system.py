"""
Smart Alert System
Telegram/Email webhook alerts for trading signals and risk events
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import requests
import json
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for environments with email import issues
    MimeText = None
    MimeMultipart = None
import os
import time
import warnings
warnings.filterwarnings('ignore')

class SmartAlertSystem:
    """Advanced alert system with multiple notification channels"""
    
    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = db_path
        self.setup_database()
        
        # Alert configuration
        self.alert_channels = {
            'telegram': {
                'enabled': False,
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
                'rate_limit': 30  # seconds between messages
            },
            'email': {
                'enabled': False,
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'username': os.getenv('EMAIL_USERNAME', ''),
                'password': os.getenv('EMAIL_PASSWORD', ''),
                'to_email': os.getenv('ALERT_EMAIL', ''),
                'rate_limit': 300  # seconds between emails
            },
            'webhook': {
                'enabled': False,
                'url': os.getenv('WEBHOOK_URL', ''),
                'headers': {'Content-Type': 'application/json'},
                'rate_limit': 10  # seconds between webhooks
            }
        }
        
        # Alert types and their priorities
        self.alert_types = {
            'buy_signal': {'priority': 'medium', 'enabled': True},
            'sell_signal': {'priority': 'medium', 'enabled': True},
            'take_profit': {'priority': 'high', 'enabled': True},
            'stop_loss': {'priority': 'high', 'enabled': True},
            'circuit_breaker': {'priority': 'critical', 'enabled': True},
            'high_confidence_signal': {'priority': 'high', 'enabled': True},
            'risk_threshold': {'priority': 'high', 'enabled': True},
            'system_error': {'priority': 'critical', 'enabled': True},
            'portfolio_milestone': {'priority': 'low', 'enabled': True}
        }
        
        # Rate limiting tracking
        self.last_alert_time = {}
        
    def setup_database(self):
        """Setup alerts database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Alerts log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                symbol TEXT,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                channels_sent TEXT,
                success INTEGER DEFAULT 0,
                error_message TEXT,
                timestamp INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alert settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default',
                alert_type TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                channels TEXT NOT NULL,
                min_confidence REAL DEFAULT 0.6,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configure Telegram alerts"""
        self.alert_channels['telegram']['bot_token'] = bot_token
        self.alert_channels['telegram']['chat_id'] = chat_id
        self.alert_channels['telegram']['enabled'] = True
        
        # Test connection
        return self._test_telegram_connection()
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, 
                       password: str, to_email: str):
        """Configure email alerts"""
        self.alert_channels['email']['smtp_server'] = smtp_server
        self.alert_channels['email']['smtp_port'] = smtp_port
        self.alert_channels['email']['username'] = username
        self.alert_channels['email']['password'] = password
        self.alert_channels['email']['to_email'] = to_email
        self.alert_channels['email']['enabled'] = True
        
        # Test connection
        return self._test_email_connection()
    
    def configure_webhook(self, url: str, headers: Dict[str, str] = None):
        """Configure webhook alerts"""
        self.alert_channels['webhook']['url'] = url
        if headers:
            self.alert_channels['webhook']['headers'].update(headers)
        self.alert_channels['webhook']['enabled'] = True
        
        # Test connection
        return self._test_webhook_connection()
    
    def send_signal_alert(self, signal_data: Dict[str, Any]):
        """Send alert for new trading signal"""
        alert_type = f"{signal_data['action'].lower()}_signal"
        
        if not self._should_send_alert(alert_type, signal_data):
            return
        
        # Determine if high confidence signal
        if signal_data.get('confidence', 0) >= 0.8:
            alert_type = 'high_confidence_signal'
        
        title = f"🚨 {signal_data['action']} Signal - {signal_data['symbol']}"
        
        message = self._format_signal_message(signal_data)
        
        self._send_alert(alert_type, title, message, signal_data.get('symbol'))
    
    def send_trade_execution_alert(self, trade_data: Dict[str, Any]):
        """Send alert for trade execution (TP/SL)"""
        if trade_data['type'] == 'take_profit':
            alert_type = 'take_profit'
            title = f"💰 Take Profit Hit - {trade_data['symbol']}"
            emoji = "💰"
        elif trade_data['type'] == 'stop_loss':
            alert_type = 'stop_loss'
            title = f"🛡️ Stop Loss Triggered - {trade_data['symbol']}"
            emoji = "🛡️"
        else:
            return
        
        message = f"""
{emoji} **Trade Closed**

**Symbol:** {trade_data['symbol']}
**Type:** {trade_data['type'].replace('_', ' ').title()}
**Entry Price:** ${trade_data.get('entry_price', 0):.4f}
**Exit Price:** ${trade_data.get('exit_price', 0):.4f}
**PnL:** ${trade_data.get('pnl', 0):.2f} ({trade_data.get('pnl_percentage', 0):.2f}%)
**Position Size:** {trade_data.get('position_size', 0):.4f}

**Strategy:** {trade_data.get('strategy', 'Unknown')}
**Hold Time:** {trade_data.get('hold_time', 'Unknown')}
        """.strip()
        
        self._send_alert(alert_type, title, message, trade_data.get('symbol'))
    
    def send_circuit_breaker_alert(self, reason: str, details: Dict[str, Any]):
        """Send critical circuit breaker alert"""
        title = "🚨 CIRCUIT BREAKER TRIGGERED"
        
        message = f"""
🚨 **CRITICAL ALERT**

**Reason:** {reason}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Details:**
        """
        
        for key, value in details.items():
            message += f"\n**{key.replace('_', ' ').title()}:** {value}"
        
        message += "\n\n⚠️ **All trading has been halted for safety**"
        
        self._send_alert('circuit_breaker', title, message)
    
    def send_risk_alert(self, risk_data: Dict[str, Any]):
        """Send risk threshold alert"""
        title = f"⚠️ Risk Alert - {risk_data.get('type', 'Unknown')}"
        
        message = f"""
⚠️ **Risk Threshold Exceeded**

**Risk Type:** {risk_data.get('type', 'Unknown')}
**Current Value:** {risk_data.get('current_value', 'N/A')}
**Threshold:** {risk_data.get('threshold', 'N/A')}
**Severity:** {risk_data.get('severity', 'Medium')}

**Recommendation:** {risk_data.get('recommendation', 'Review position sizes and risk parameters')}
        """.strip()
        
        self._send_alert('risk_threshold', title, message)
    
    def send_system_error_alert(self, error_type: str, error_message: str, 
                               component: str = None):
        """Send system error alert"""
        title = f"🔥 System Error - {error_type}"
        
        message = f"""
🔥 **System Error Detected**

**Error Type:** {error_type}
**Component:** {component or 'Unknown'}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Error Message:**
```
{error_message}
```

**Action Required:** Check system logs and resolve the issue
        """.strip()
        
        self._send_alert('system_error', title, message)
    
    def send_portfolio_milestone_alert(self, milestone_data: Dict[str, Any]):
        """Send portfolio milestone alert"""
        title = f"🎯 Portfolio Milestone - {milestone_data.get('type', 'Achievement')}"
        
        message = f"""
🎯 **Portfolio Milestone Reached**

**Achievement:** {milestone_data.get('description', 'Portfolio milestone')}
**Current Value:** ${milestone_data.get('current_value', 0):,.2f}
**Gain/Loss:** ${milestone_data.get('change_value', 0):,.2f} ({milestone_data.get('change_percentage', 0):.2f}%)
**Time Period:** {milestone_data.get('period', 'Unknown')}

**Performance Summary:**
- **Total Trades:** {milestone_data.get('total_trades', 0)}
- **Win Rate:** {milestone_data.get('win_rate', 0):.1%}
- **Best Trade:** ${milestone_data.get('best_trade', 0):,.2f}
        """.strip()
        
        self._send_alert('portfolio_milestone', title, message)
    
    def _format_signal_message(self, signal_data: Dict[str, Any]) -> str:
        """Format trading signal message"""
        action = signal_data['action']
        symbol = signal_data['symbol']
        confidence = signal_data.get('confidence', 0)
        current_price = signal_data.get('current_price', 0)
        
        # Select emoji based on action
        emoji_map = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
        emoji = emoji_map.get(action, '🔵')
        
        message = f"""
{emoji} **{action} Signal Generated**

**Symbol:** {symbol}
**Current Price:** ${current_price:.4f}
**Confidence:** {confidence:.1%}
**Signal Strength:** {signal_data.get('signal_strength', 0):.3f}

**Strategy:** {signal_data.get('strategy', 'Unknown')}
**Timeframe:** {signal_data.get('timeframe', 'Unknown')}
        """
        
        # Add position sizing info if available
        if 'position_size' in signal_data:
            message += f"\n**Recommended Size:** {signal_data['position_size']:.1%} of portfolio"
        
        # Add stop loss and take profit levels
        if signal_data.get('stop_loss'):
            message += f"\n**Stop Loss:** ${signal_data['stop_loss']:.4f}"
        if signal_data.get('take_profit'):
            message += f"\n**Take Profit:** ${signal_data['take_profit']:.4f}"
        
        # Add expected hold time
        if signal_data.get('expected_hold_time'):
            message += f"\n**Expected Hold Time:** {signal_data['expected_hold_time']}"
        
        # Add signal components breakdown
        if 'components' in signal_data:
            components = signal_data['components']
            message += f"""

**Signal Breakdown:**
- **ML Score:** {components.get('ml', {}).get('signal', 0):.3f}
- **Technical Score:** {components.get('technical', {}).get('signal', 0):.3f}
- **Sentiment Score:** {components.get('sentiment', {}).get('signal', 0):.3f}
            """
        
        return message.strip()
    
    def _should_send_alert(self, alert_type: str, data: Dict[str, Any] = None) -> bool:
        """Check if alert should be sent based on settings and rate limiting"""
        
        # Check if alert type is enabled
        if not self.alert_types.get(alert_type, {}).get('enabled', False):
            return False
        
        # Check rate limiting
        now = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        
        # Different rate limits for different alert types
        rate_limits = {
            'buy_signal': 300,  # 5 minutes
            'sell_signal': 300,  # 5 minutes
            'high_confidence_signal': 120,  # 2 minutes
            'take_profit': 0,  # No rate limit
            'stop_loss': 0,  # No rate limit
            'circuit_breaker': 0,  # No rate limit
            'risk_threshold': 600,  # 10 minutes
            'system_error': 300,  # 5 minutes
            'portfolio_milestone': 3600  # 1 hour
        }
        
        min_interval = rate_limits.get(alert_type, 300)
        if min_interval > 0 and (now - last_time) < min_interval:
            return False
        
        # Check confidence threshold for signal alerts
        if alert_type.endswith('_signal') and data:
            min_confidence = 0.6  # Default minimum confidence
            if data.get('confidence', 0) < min_confidence:
                return False
        
        return True
    
    def _send_alert(self, alert_type: str, title: str, message: str, symbol: str = None):
        """Send alert through all enabled channels"""
        
        channels_sent = []
        success = True
        error_message = ""
        
        # Send through Telegram
        if self.alert_channels['telegram']['enabled']:
            try:
                self._send_telegram_alert(title, message)
                channels_sent.append('telegram')
            except Exception as e:
                success = False
                error_message += f"Telegram error: {str(e)}; "
        
        # Send through Email
        if self.alert_channels['email']['enabled']:
            try:
                self._send_email_alert(title, message)
                channels_sent.append('email')
            except Exception as e:
                success = False
                error_message += f"Email error: {str(e)}; "
        
        # Send through Webhook
        if self.alert_channels['webhook']['enabled']:
            try:
                self._send_webhook_alert(title, message, alert_type, symbol)
                channels_sent.append('webhook')
            except Exception as e:
                success = False
                error_message += f"Webhook error: {str(e)}; "
        
        # Update rate limiting
        self.last_alert_time[alert_type] = time.time()
        
        # Log alert
        self._log_alert(alert_type, title, message, channels_sent, success, error_message, symbol)
    
    def _send_telegram_alert(self, title: str, message: str):
        """Send alert via Telegram"""
        config = self.alert_channels['telegram']
        
        if not config['bot_token'] or not config['chat_id']:
            raise Exception("Telegram bot token or chat ID not configured")
        
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        
        # Format message for Telegram
        formatted_message = f"*{title}*\n\n{message}"
        
        payload = {
            'chat_id': config['chat_id'],
            'text': formatted_message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if not response.ok:
            raise Exception(f"Telegram API error: {response.text}")
    
    def _send_email_alert(self, title: str, message: str):
        """Send alert via Email"""
        if MimeText is None or MimeMultipart is None:
            raise Exception("Email functionality not available in this environment")
            
        config = self.alert_channels['email']
        
        if not all([config['username'], config['password'], config['to_email']]):
            raise Exception("Email configuration incomplete")
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = config['username']
        msg['To'] = config['to_email']
        msg['Subject'] = f"Trading Alert: {title}"
        
        # Convert markdown-style formatting to plain text
        plain_message = message.replace('**', '').replace('*', '').replace('```', '')
        msg.attach(MimeText(plain_message, 'plain'))
        
        # Send email
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['username'], config['password'])
        text = msg.as_string()
        server.sendmail(config['username'], config['to_email'], text)
        server.quit()
    
    def _send_webhook_alert(self, title: str, message: str, alert_type: str, symbol: str = None):
        """Send alert via Webhook"""
        config = self.alert_channels['webhook']
        
        if not config['url']:
            raise Exception("Webhook URL not configured")
        
        payload = {
            'alert_type': alert_type,
            'title': title,
            'message': message,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'priority': self.alert_types.get(alert_type, {}).get('priority', 'medium')
        }
        
        response = requests.post(
            config['url'],
            json=payload,
            headers=config['headers'],
            timeout=10
        )
        
        if not response.ok:
            raise Exception(f"Webhook error: {response.status_code} - {response.text}")
    
    def _test_telegram_connection(self) -> Dict[str, Any]:
        """Test Telegram connection"""
        try:
            self._send_telegram_alert("Test Alert", "Telegram alerts are working correctly!")
            return {'success': True, 'message': 'Telegram connection successful'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_email_connection(self) -> Dict[str, Any]:
        """Test email connection"""
        try:
            self._send_email_alert("Test Alert", "Email alerts are working correctly!")
            return {'success': True, 'message': 'Email connection successful'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_webhook_connection(self) -> Dict[str, Any]:
        """Test webhook connection"""
        try:
            self._send_webhook_alert("Test Alert", "Webhook alerts are working correctly!", "test", None)
            return {'success': True, 'message': 'Webhook connection successful'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _log_alert(self, alert_type: str, title: str, message: str, 
                  channels_sent: List[str], success: bool, error_message: str, symbol: str = None):
        """Log alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = int(datetime.now().timestamp())
        priority = self.alert_types.get(alert_type, {}).get('priority', 'medium')
        
        cursor.execute('''
            INSERT INTO alerts_log 
            (alert_type, priority, symbol, title, message, channels_sent, 
             success, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert_type, priority, symbol, title, message,
            ','.join(channels_sent), int(success), error_message, timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_alert_history(self, days: int = 7, alert_type: str = None) -> pd.DataFrame:
        """Get alert history"""
        conn = sqlite3.connect(self.db_path)
        
        since_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
        
        if alert_type:
            query = '''
                SELECT * FROM alerts_log 
                WHERE timestamp >= ? AND alert_type = ?
                ORDER BY timestamp DESC
            '''
            params = (since_timestamp, alert_type)
        else:
            query = '''
                SELECT * FROM alerts_log 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            params = (since_timestamp,)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get alert statistics"""
        df = self.get_alert_history(days)
        
        if df.empty:
            return {
                'total_alerts': 0,
                'success_rate': 0.0,
                'alerts_by_type': {},
                'alerts_by_priority': {},
                'channels_usage': {}
            }
        
        # Convert timestamp to datetime for analysis
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return {
            'total_alerts': len(df),
            'success_rate': df['success'].mean(),
            'alerts_by_type': df['alert_type'].value_counts().to_dict(),
            'alerts_by_priority': df['priority'].value_counts().to_dict(),
            'failed_alerts': len(df[df['success'] == 0]),
            'most_recent_alert': df['datetime'].max().isoformat() if not df.empty else None,
            'alerts_per_day': len(df) / max(days, 1)
        }
    
    def update_alert_settings(self, alert_type: str, enabled: bool, 
                            channels: List[str] = None, min_confidence: float = None):
        """Update alert settings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update in-memory settings
        if alert_type in self.alert_types:
            self.alert_types[alert_type]['enabled'] = enabled
        
        # Update database settings
        channels_str = ','.join(channels) if channels else 'telegram,email,webhook'
        min_conf = min_confidence if min_confidence is not None else 0.6
        
        cursor.execute('''
            INSERT OR REPLACE INTO alert_settings 
            (alert_type, enabled, channels, min_confidence)
            VALUES (?, ?, ?, ?)
        ''', (alert_type, int(enabled), channels_str, min_conf))
        
        conn.commit()
        conn.close()
    
    def get_channel_status(self) -> Dict[str, Any]:
        """Get status of all alert channels"""
        status = {}
        
        for channel_name, config in self.alert_channels.items():
            status[channel_name] = {
                'enabled': config['enabled'],
                'configured': self._is_channel_configured(channel_name),
                'last_test': 'Never'  # Would track last test time in production
            }
        
        return status
    
    def _is_channel_configured(self, channel_name: str) -> bool:
        """Check if channel is properly configured"""
        config = self.alert_channels[channel_name]
        
        if channel_name == 'telegram':
            return bool(config['bot_token'] and config['chat_id'])
        elif channel_name == 'email':
            return bool(config['username'] and config['password'] and config['to_email'])
        elif channel_name == 'webhook':
            return bool(config['url'])
        
        return False