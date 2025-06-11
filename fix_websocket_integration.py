
#!/usr/bin/env python3
"""
Fix WebSocket integration issues in the complete trading platform
"""

import re

def fix_websocket_issues():
    """Fix WebSocket configuration and error handling"""
    
    # Read the main platform file
    try:
        with open('complete_trading_platform.py', 'r') as f:
            content = f.read()
        
        # Check if SocketIO is already imported
        if 'from flask_socketio import SocketIO' not in content:
            # Add SocketIO import
            import_section = content.find('from flask import')
            if import_section != -1:
                # Find the end of the Flask import line
                line_end = content.find('\n', import_section)
                content = (content[:line_end] + 
                          '\nfrom flask_socketio import SocketIO, emit' + 
                          content[line_end:])
        
        # Check if SocketIO is initialized
        if 'socketio = SocketIO(' not in content:
            # Find where app is created
            app_creation = content.find('app = Flask(__name__)')
            if app_creation != -1:
                line_end = content.find('\n', app_creation)
                socketio_config = '''
# Configure SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
'''
                content = content[:line_end] + socketio_config + content[line_end:]
        
        # Add WebSocket route if not present
        if '@socketio.on(' not in content:
            websocket_routes = '''

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info("WebSocket client connected")
    emit('status', {'message': 'Connected successfully'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info("WebSocket client disconnected")

@socketio.on('subscribe_updates')
def handle_subscribe(data):
    """Handle subscription to real-time updates"""
    logger.info(f"Client subscribed to: {data}")
    emit('subscription_confirmed', {'status': 'subscribed'})

def broadcast_signal_update(signal_data):
    """Broadcast signal updates to connected clients"""
    try:
        socketio.emit('signal_update', signal_data)
    except Exception as e:
        logger.error(f"Error broadcasting signal update: {e}")

def broadcast_portfolio_update(portfolio_data):
    """Broadcast portfolio updates to connected clients"""
    try:
        socketio.emit('portfolio_update', portfolio_data)
    except Exception as e:
        logger.error(f"Error broadcasting portfolio update: {e}")
'''
            # Insert before the main execution block
            main_block = content.find('if __name__ == "__main__":')
            if main_block != -1:
                content = content[:main_block] + websocket_routes + '\n\n' + content[main_block:]
        
        # Update the app.run() call to use socketio.run()
        if 'socketio.run(' not in content:
            content = re.sub(
                r'app\.run\(host=[^)]+\)',
                'socketio.run(app, host="0.0.0.0", port=5000, debug=False)',
                content
            )
        
        # Write the updated content back
        with open('complete_trading_platform.py', 'w') as f:
            f.write(content)
        
        print("✅ WebSocket configuration fixed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing WebSocket configuration: {e}")
        return False

if __name__ == "__main__":
    fix_websocket_issues()
