#!/usr/bin/env python3
"""
Test OKX Connection
"""
import os
import ccxt

def test_okx_connection():
    try:
        api_key = os.environ.get('OKX_API_KEY')
        secret = os.environ.get('OKX_SECRET_KEY') 
        passphrase = os.environ.get('OKX_PASSPHRASE')
        
        print(f"API Key exists: {bool(api_key)}")
        print(f"Secret exists: {bool(secret)}")
        print(f"Passphrase exists: {bool(passphrase)}")
        
        if not all([api_key, secret, passphrase]):
            print("ERROR: Missing OKX credentials")
            return False
            
        okx = ccxt.okx({
            'apiKey': api_key,
            'secret': secret,
            'password': passphrase,
            'sandbox': False,
            'enableRateLimit': True
        })
        
        # Test basic connection
        balance = okx.fetch_balance()
        print(f"SUCCESS: Connected to OKX")
        print(f"USDT Balance: {balance.get('USDT', {}).get('total', 0)}")
        return True
        
    except ccxt.AuthenticationError as e:
        print(f"AUTHENTICATION ERROR: {e}")
        return False
    except ccxt.NetworkError as e:
        print(f"NETWORK ERROR: {e}")
        return False
    except Exception as e:
        print(f"GENERAL ERROR: {e}")
        return False

if __name__ == "__main__":
    test_okx_connection()