#!/usr/bin/env python3
"""
Professional Trading UI Launcher
Ensures the modern trading interface is properly started and accessible
"""

import os
import sys
import time
import socket
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_available_port(start_port=8080, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex(('127.0.0.1', port))
            if result != 0:  # Port is available
                return port
        finally:
            sock.close()
    return None

def check_port_status(port):
    """Check if a port is accessible"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0  # True if connected successfully
    finally:
        sock.close()

def launch_flask_app():
    """Launch the Flask application with proper error handling"""
    logger.info("Starting Professional Trading UI Launcher")
    
    # Find available port
    port = find_available_port()
    if not port:
        logger.error("No available ports found")
        return False
    
    logger.info(f"Using port {port}")
    
    # Set environment variable
    os.environ['PORT'] = str(port)
    
    try:
        # Import and run the Flask app
        from modern_trading_app import create_app
        app = create_app()
        
        logger.info("Flask application created successfully")
        logger.info(f"Professional UI with 3Commas/TradingView design starting on port {port}")
        
        # Start the Flask server
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        return False

def verify_ui_accessibility(port=8080, timeout=30):
    """Verify the UI is accessible via HTTP"""
    import requests
    
    url = f"http://localhost:{port}/health"
    logger.info(f"Verifying UI accessibility at {url}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("Professional Trading UI is accessible")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    logger.warning("UI accessibility verification failed")
    return False

def create_system_status_report():
    """Create a comprehensive system status report"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'ui_status': 'checking',
        'port': None,
        'accessibility': False,
        'components': {}
    }
    
    # Check for available ports
    port = find_available_port()
    status['port'] = port
    
    # Check if UI is running
    if port and check_port_status(port):
        status['ui_status'] = 'running'
        status['accessibility'] = True
    else:
        status['ui_status'] = 'not_running'
    
    # Component status
    components = [
        'templates/modern/base.html',
        'templates/modern/dashboard.html',
        'templates/modern/portfolio.html',
        'templates/modern/strategy_builder.html',
        'templates/modern/analytics.html',
        'templates/modern/ai_panel.html',
        'templates/modern/settings.html',
        'modern_trading_app.py'
    ]
    
    for component in components:
        status['components'][component] = os.path.exists(component)
    
    return status

if __name__ == '__main__':
    # Create status report
    status = create_system_status_report()
    logger.info(f"System Status: {status}")
    
    # Launch the application
    try:
        success = launch_flask_app()
        if success:
            logger.info("Professional Trading UI launched successfully")
        else:
            logger.error("Failed to launch Professional Trading UI")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Professional Trading UI")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)