#!/usr/bin/env python3
"""
Comprehensive Professional Trading UI Launcher
Complete system initialization and verification
"""

import os
import sys
import time
import socket
import logging
import subprocess
from datetime import datetime
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_required_files():
    """Verify all required files exist"""
    required_files = [
        'modern_trading_app.py',
        'templates/modern/base.html',
        'templates/modern/dashboard.html',
        'templates/modern/portfolio.html',
        'templates/modern/strategy_builder.html',
        'templates/modern/analytics.html',
        'templates/modern/ai_panel.html',
        'templates/modern/settings.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("All required files present")
    return True

def find_available_port(start_port=8080):
    """Find available port for Flask application"""
    for port in range(start_port, start_port + 20):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex(('127.0.0.1', port))
            if result != 0:
                logger.info(f"Found available port: {port}")
                return port
        finally:
            sock.close()
    
    logger.error("No available ports found")
    return None

def kill_existing_processes():
    """Kill any existing Flask processes"""
    try:
        subprocess.run(['pkill', '-f', 'modern_trading_app'], check=False)
        subprocess.run(['pkill', '-f', 'flask'], check=False)
        time.sleep(2)
        logger.info("Cleared existing processes")
    except Exception as e:
        logger.warning(f"Error clearing processes: {e}")

def verify_imports():
    """Verify all required imports work"""
    try:
        from modern_trading_app import create_app
        logger.info("Successfully imported Flask application")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected import error: {e}")
        return False

def start_flask_application(port):
    """Start Flask application with comprehensive error handling"""
    try:
        # Set environment variables
        os.environ['PORT'] = str(port)
        os.environ['FLASK_ENV'] = 'production'
        
        # Import and create Flask app
        from modern_trading_app import create_app
        app = create_app()
        
        logger.info(f"Starting Professional Trading Platform on port {port}")
        logger.info("Features: 3Commas/TradingView inspired interface")
        
        # Configure Flask app
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
        
        # Start server
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start Flask application: {e}")
        return False

def verify_server_health(port, timeout=30):
    """Verify server is responding to requests"""
    import requests
    
    endpoints = [
        f"http://localhost:{port}/health",
        f"http://localhost:{port}/",
        f"http://localhost:{port}/api/dashboard-data"
    ]
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Server responding at {endpoint}")
                    return True
            except requests.exceptions.RequestException:
                pass
        
        time.sleep(3)
    
    logger.warning("Server health verification failed")
    return False

def create_comprehensive_status_report():
    """Create detailed system status report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': 'initializing',
        'components': {},
        'errors': [],
        'port': None,
        'accessibility': False
    }
    
    # Check file system
    if check_required_files():
        report['components']['files'] = 'present'
    else:
        report['components']['files'] = 'missing'
        report['errors'].append('Missing required template files')
    
    # Check imports
    if verify_imports():
        report['components']['imports'] = 'working'
    else:
        report['components']['imports'] = 'failed'
        report['errors'].append('Import errors detected')
    
    # Check port availability
    port = find_available_port()
    if port:
        report['port'] = port
        report['components']['port'] = 'available'
    else:
        report['components']['port'] = 'unavailable'
        report['errors'].append('No available ports found')
    
    # Determine overall status
    if not report['errors']:
        report['system_status'] = 'ready'
    else:
        report['system_status'] = 'errors_detected'
    
    return report

def main():
    """Main launcher function"""
    logger.info("=== Professional Trading UI Launcher ===")
    
    # Create status report
    status = create_comprehensive_status_report()
    logger.info(f"System Status: {status['system_status']}")
    
    if status['errors']:
        for error in status['errors']:
            logger.error(error)
        return False
    
    # Clear existing processes
    kill_existing_processes()
    
    # Get port
    port = status['port']
    if not port:
        logger.error("Cannot start without available port")
        return False
    
    # Start Flask application
    logger.info("Starting Flask application...")
    success = start_flask_application(port)
    
    if success:
        logger.info("Professional Trading UI launched successfully")
        logger.info(f"Access at: http://localhost:{port}")
        return True
    else:
        logger.error("Failed to launch Professional Trading UI")
        return False

if __name__ == '__main__':
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Professional Trading UI")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)