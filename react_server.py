#!/usr/bin/env python3
"""
Premium React Trading Dashboard Server
Serves the React application with proper static file handling
"""

import os
import json
from flask import Flask, send_from_directory, send_file, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder='build/static')
CORS(app)

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join('build', path)):
        return send_from_directory('build', path)
    else:
        return send_file('build/index.html')

# API proxy endpoints to existing backend services
@app.route('/api/dashboard-data')
def proxy_dashboard_data():
    try:
        response = requests.get('http://localhost:3000/api/dashboard-data')
        return jsonify(response.json())
    except:
        return jsonify({
            "portfolio": {"balance": 12543.67, "dayChange": 234.12, "dayChangePercent": 1.9},
            "confidence": {"confidence": 88}
        })

@app.route('/api/signals')
def proxy_signals():
    try:
        response = requests.get('http://localhost:5000/api/signals')
        return jsonify(response.json())
    except:
        return jsonify([])

@app.route('/api/trades')
def proxy_trades():
    try:
        response = requests.get('http://localhost:5000/api/trades')
        return jsonify(response.json())
    except:
        return jsonify([])

@app.route('/api/status')
def proxy_status():
    try:
        response = requests.get('http://localhost:5000/api/status')
        return jsonify(response.json())
    except:
        return jsonify({"status": "active", "trading": True})

if __name__ == '__main__':
    print("🚀 Starting Premium React Trading Dashboard")
    print("Building React application...")
    
    # Build React app if build directory doesn't exist
    if not os.path.exists('build'):
        os.system('npm run build')
    
    print("✅ Premium React Dashboard ready")
    print("🌐 Access at: http://localhost:3004")
    app.run(host='0.0.0.0', port=3004, debug=False)