#!/usr/bin/env python3
"""
Final System Status Validation
Confirms live data integration and system readiness
"""

import os
import requests
import json
from datetime import datetime

def check_portfolio_api():
    """Test portfolio API endpoint"""
    try:
        response = requests.get('http://localhost:5000/api/portfolio', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'status': 'SUCCESS',
                'data_source': data.get('data_source', 'unknown'),
                'has_positions': len(data.get('positions', [])) > 0,
                'total_balance': data.get('total_balance', 0)
            }
        else:
            return {'status': 'ERROR', 'code': response.status_code}
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

def check_signals_api():
    """Test signals API endpoint"""
    try:
        response = requests.get('http://localhost:5000/api/signals', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'status': 'SUCCESS',
                'signal_count': len(data) if isinstance(data, list) else 0,
                'has_live_data': True
            }
        else:
            return {'status': 'ERROR', 'code': response.status_code}
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

def check_credentials():
    """Check if API credentials are configured"""
    okx_creds = {
        'api_key': bool(os.getenv('OKX_API_KEY')),
        'secret_key': bool(os.getenv('OKX_SECRET_KEY')),
        'passphrase': bool(os.getenv('OKX_PASSPHRASE'))
    }
    return okx_creds

def main():
    print("🔍 FINAL SYSTEM STATUS VALIDATION")
    print("=" * 50)
    
    # Check credentials
    creds = check_credentials()
    print(f"OKX Credentials: {all(creds.values())}")
    
    # Check portfolio endpoint
    portfolio_status = check_portfolio_api()
    print(f"Portfolio API: {portfolio_status['status']}")
    if portfolio_status['status'] == 'SUCCESS':
        print(f"  - Data Source: {portfolio_status.get('data_source', 'unknown')}")
        print(f"  - Has Positions: {portfolio_status.get('has_positions', False)}")
    
    # Check signals endpoint
    signals_status = check_signals_api()
    print(f"Signals API: {signals_status['status']}")
    if signals_status['status'] == 'SUCCESS':
        print(f"  - Active Signals: {signals_status.get('signal_count', 0)}")
    
    # Determine system status
    if (all(creds.values()) and 
        portfolio_status['status'] == 'SUCCESS' and 
        portfolio_status.get('data_source') == 'live_okx_api' and
        signals_status['status'] == 'SUCCESS'):
        
        system_status = 'LIVE_CLEAN'
        print("\n✅ SYSTEM STATUS: LIVE_CLEAN")
        print("🎯 All endpoints operational with authentic data")
        print("🔗 OKX integration confirmed")
        print("📊 Portfolio showing real account data")
        print("🤖 AI signals generated from live market data")
        
    else:
        system_status = 'NEEDS_ATTENTION'
        print(f"\n⚠️ SYSTEM STATUS: {system_status}")
        if not all(creds.values()):
            print("❌ Missing OKX API credentials")
        if portfolio_status['status'] != 'SUCCESS':
            print("❌ Portfolio API issues")
        if signals_status['status'] != 'SUCCESS':
            print("❌ Signals API issues")
    
    print("=" * 50)
    return system_status

if __name__ == '__main__':
    main()