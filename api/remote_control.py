"""
Remote Control API - Secure Emergency Stop and Remote Management
Provides secure endpoints for emergency system control from mobile or cloud triggers
"""

from flask import Flask, request, jsonify
import hashlib
import hmac
import time
import json
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import threading

logger = logging.getLogger(__name__)

class RemoteControlAPI:
    """Secure remote control system for trading operations"""
    
    def __init__(self, api_key: str = None, webhook_secret: str = None):
        self.api_key = api_key or os.getenv('REMOTE_API_KEY', 'default_key_change_me')
        self.webhook_secret = webhook_secret or os.getenv('WEBHOOK_SECRET', 'default_secret')
        self.emergency_stop_active = False
        self.authorized_actions = []
        
        # Rate limiting
        self.request_history = []
        self.max_requests_per_minute = 10
        
        # System state
        self.system_status = {
            'trading_enabled': True,
            'emergency_stop': False,
            'last_heartbeat': datetime.now(),
            'active_protection': 'none'
        }
        
        self.setup_database()
        self.app = self.create_flask_app()
    
    def setup_database(self):
        """Initialize remote control database"""
        try:
            conn = sqlite3.connect('remote_control.db')
            cursor = conn.cursor()
            
            # Remote control actions log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS remote_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action_type TEXT NOT NULL,
                    source_ip TEXT,
                    api_key_hash TEXT,
                    
                    -- Action details
                    action_data TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    
                    -- Security
                    signature_valid BOOLEAN,
                    rate_limit_passed BOOLEAN
                )
            ''')
            
            # System state log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_state_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    state_type TEXT,
                    previous_state TEXT,
                    new_state TEXT,
                    trigger_source TEXT,
                    notes TEXT
                )
            ''')
            
            # Emergency stops log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emergency_stops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stop_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resume_timestamp DATETIME,
                    trigger_reason TEXT,
                    trigger_source TEXT,
                    
                    -- Impact assessment
                    trades_halted INTEGER,
                    duration_minutes REAL,
                    recovery_method TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Remote control database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def verify_signature(self, payload: str, signature: str) -> bool:
        """Verify webhook signature for security"""
        try:
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def verify_api_key(self, provided_key: str) -> bool:
        """Verify API key"""
        try:
            return hmac.compare_digest(provided_key, self.api_key)
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            return False
    
    def check_rate_limit(self, source_ip: str) -> bool:
        """Check if request is within rate limits"""
        try:
            current_time = time.time()
            
            # Clean old requests
            self.request_history = [
                (ip, timestamp) for ip, timestamp in self.request_history
                if current_time - timestamp < 60  # Last minute
            ]
            
            # Count requests from this IP
            ip_requests = len([r for r in self.request_history if r[0] == source_ip])
            
            if ip_requests >= self.max_requests_per_minute:
                return False
            
            # Add current request
            self.request_history.append((source_ip, current_time))
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return False
    
    def log_remote_action(self, action_type: str, source_ip: str, api_key_hash: str,
                         action_data: Dict, success: bool, error_message: str = None,
                         signature_valid: bool = True, rate_limit_passed: bool = True):
        """Log remote control action"""
        try:
            conn = sqlite3.connect('remote_control.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO remote_actions (
                    action_type, source_ip, api_key_hash, action_data,
                    success, error_message, signature_valid, rate_limit_passed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                action_type, source_ip, api_key_hash, json.dumps(action_data),
                success, error_message, signature_valid, rate_limit_passed
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log remote action: {e}")
    
    def emergency_stop(self, reason: str, source: str) -> Dict:
        """Activate emergency stop"""
        try:
            if self.emergency_stop_active:
                return {
                    'success': False,
                    'message': 'Emergency stop already active',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Activate emergency stop
            self.emergency_stop_active = True
            self.system_status['emergency_stop'] = True
            self.system_status['trading_enabled'] = False
            
            # Log emergency stop
            conn = sqlite3.connect('remote_control.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO emergency_stops (trigger_reason, trigger_source)
                VALUES (?, ?)
            ''', (reason, source))
            
            conn.commit()
            conn.close()
            
            # Log system state change
            self.log_system_state_change('emergency_stop', 'false', 'true', source, reason)
            
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason} (Source: {source})")
            
            return {
                'success': True,
                'message': 'Emergency stop activated',
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status.copy()
            }
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def resume_trading(self, reason: str, source: str) -> Dict:
        """Resume trading after emergency stop"""
        try:
            if not self.emergency_stop_active:
                return {
                    'success': False,
                    'message': 'Emergency stop not active',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate stop duration
            conn = sqlite3.connect('remote_control.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT stop_timestamp FROM emergency_stops 
                WHERE resume_timestamp IS NULL 
                ORDER BY stop_timestamp DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            if result:
                stop_time = datetime.fromisoformat(result[0])
                duration_minutes = (datetime.now() - stop_time).total_seconds() / 60
                
                # Update emergency stop record
                cursor.execute('''
                    UPDATE emergency_stops 
                    SET resume_timestamp = ?, duration_minutes = ?, recovery_method = ?
                    WHERE resume_timestamp IS NULL
                ''', (datetime.now(), duration_minutes, reason))
            
            conn.commit()
            conn.close()
            
            # Deactivate emergency stop
            self.emergency_stop_active = False
            self.system_status['emergency_stop'] = False
            self.system_status['trading_enabled'] = True
            
            # Log system state change
            self.log_system_state_change('emergency_stop', 'true', 'false', source, reason)
            
            logger.info(f"Trading resumed: {reason} (Source: {source})")
            
            return {
                'success': True,
                'message': 'Trading resumed',
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status.copy()
            }
            
        except Exception as e:
            logger.error(f"Resume trading failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def log_system_state_change(self, state_type: str, previous_state: str, 
                               new_state: str, trigger_source: str, notes: str):
        """Log system state changes"""
        try:
            conn = sqlite3.connect('remote_control.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_state_log (
                    state_type, previous_state, new_state, trigger_source, notes
                ) VALUES (?, ?, ?, ?, ?)
            ''', (state_type, previous_state, new_state, trigger_source, notes))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log state change: {e}")
    
    def create_flask_app(self) -> Flask:
        """Create Flask application with secure endpoints"""
        app = Flask(__name__)
        
        @app.route('/api/emergency_stop', methods=['POST'])
        def api_emergency_stop():
            try:
                # Get client IP
                source_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                
                # Rate limiting
                if not self.check_rate_limit(source_ip):
                    return jsonify({
                        'success': False,
                        'error': 'Rate limit exceeded'
                    }), 429
                
                # Verify API key
                api_key = request.headers.get('X-API-Key')
                if not api_key or not self.verify_api_key(api_key):
                    self.log_remote_action('emergency_stop', source_ip, 
                                         hashlib.sha256(api_key.encode() if api_key else b'').hexdigest(),
                                         {}, False, 'Invalid API key', True, True)
                    return jsonify({
                        'success': False,
                        'error': 'Invalid API key'
                    }), 401
                
                # Get request data
                data = request.get_json() or {}
                reason = data.get('reason', 'Remote emergency stop')
                
                # Verify signature if provided
                signature = request.headers.get('X-Signature')
                signature_valid = True
                if signature:
                    payload = request.get_data(as_text=True)
                    signature_valid = self.verify_signature(payload, signature)
                    if not signature_valid:
                        return jsonify({
                            'success': False,
                            'error': 'Invalid signature'
                        }), 401
                
                # Execute emergency stop
                result = self.emergency_stop(reason, f'API:{source_ip}')
                
                # Log action
                self.log_remote_action('emergency_stop', source_ip,
                                     hashlib.sha256(api_key.encode()).hexdigest(),
                                     data, result['success'], 
                                     result.get('error'), signature_valid, True)
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"Emergency stop API error: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error'
                }), 500
        
        @app.route('/api/resume_trading', methods=['POST'])
        def api_resume_trading():
            try:
                # Similar security checks as emergency_stop
                source_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                
                if not self.check_rate_limit(source_ip):
                    return jsonify({
                        'success': False,
                        'error': 'Rate limit exceeded'
                    }), 429
                
                api_key = request.headers.get('X-API-Key')
                if not api_key or not self.verify_api_key(api_key):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid API key'
                    }), 401
                
                data = request.get_json() or {}
                reason = data.get('reason', 'Remote resume')
                
                # Execute resume
                result = self.resume_trading(reason, f'API:{source_ip}')
                
                # Log action
                self.log_remote_action('resume_trading', source_ip,
                                     hashlib.sha256(api_key.encode()).hexdigest(),
                                     data, result['success'], result.get('error'))
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"Resume trading API error: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error'
                }), 500
        
        @app.route('/api/system_status', methods=['GET'])
        def api_system_status():
            try:
                source_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                
                if not self.check_rate_limit(source_ip):
                    return jsonify({
                        'success': False,
                        'error': 'Rate limit exceeded'
                    }), 429
                
                api_key = request.headers.get('X-API-Key')
                if not api_key or not self.verify_api_key(api_key):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid API key'
                    }), 401
                
                # Update heartbeat
                self.system_status['last_heartbeat'] = datetime.now()
                
                return jsonify({
                    'success': True,
                    'system_status': self.system_status,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"System status API error: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Internal server error'
                }), 500
        
        @app.route('/api/health', methods=['GET'])
        def api_health():
            """Public health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'emergency_stop_active': self.emergency_stop_active
            }), 200
        
        return app
    
    def update_system_status(self, status_updates: Dict):
        """Update system status from trading components"""
        try:
            self.system_status.update(status_updates)
            self.system_status['last_heartbeat'] = datetime.now()
        except Exception as e:
            logger.error(f"System status update failed: {e}")
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        if self.emergency_stop_active:
            return False, "Emergency stop active"
        
        if not self.system_status.get('trading_enabled', True):
            return False, "Trading disabled"
        
        return True, "Trading allowed"
    
    def get_control_insights(self) -> List[str]:
        """Get insights about remote control status"""
        try:
            insights = []
            
            if self.emergency_stop_active:
                insights.append("🚨 EMERGENCY STOP ACTIVE")
            else:
                insights.append("✅ Remote control ready")
            
            # Recent actions
            conn = sqlite3.connect('remote_control.db')
            recent_actions = pd.read_sql_query('''
                SELECT action_type, COUNT(*) as count
                FROM remote_actions 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY action_type
            ''', conn)
            
            if not recent_actions.empty:
                for _, row in recent_actions.iterrows():
                    insights.append(f"24h {row['action_type']}: {row['count']} calls")
            
            conn.close()
            
            insights.append(f"API security: {self.max_requests_per_minute} req/min limit")
            
            return insights
            
        except Exception as e:
            logger.error(f"Control insights failed: {e}")
            return ["Remote control monitoring active"]
    
    def run_server(self, host: str = '0.0.0.0', port: int = 8000):
        """Run remote control server"""
        try:
            logger.info(f"Starting remote control API server on {host}:{port}")
            self.app.run(host=host, port=port, debug=False)
        except Exception as e:
            logger.error(f"Remote control server failed: {e}")

# Global instance for integration
remote_control = RemoteControlAPI()

def start_remote_control_server(port: int = 8000):
    """Start remote control server in background thread"""
    server_thread = threading.Thread(
        target=remote_control.run_server,
        args=('0.0.0.0', port),
        daemon=True
    )
    server_thread.start()
    logger.info(f"Remote control server started on port {port}")

def emergency_stop(reason: str = "Manual trigger") -> Dict:
    """Trigger emergency stop"""
    return remote_control.emergency_stop(reason, "local")

def resume_trading(reason: str = "Manual resume") -> Dict:
    """Resume trading"""
    return remote_control.resume_trading(reason, "local")

def is_trading_allowed() -> Tuple[bool, str]:
    """Check if trading is currently allowed"""
    return remote_control.is_trading_allowed()