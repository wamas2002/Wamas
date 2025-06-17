#!/usr/bin/env python3
"""
OKX Connection Test - Verify API credentials and connection
اختبار اتصال OKX - التحقق من بيانات الاعتماد والاتصال
"""

import os
import sys
import time
import ccxt
from datetime import datetime

def load_environment():
    """Load environment variables from .env file"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print("❌ ملف .env غير موجود")
        return False
    return True

def test_okx_credentials():
    """Test OKX API credentials"""
    print("🔑 اختبار بيانات الاعتماد...")
    
    api_key = os.environ.get('OKX_API_KEY')
    secret = os.environ.get('OKX_SECRET_KEY')
    passphrase = os.environ.get('OKX_PASSPHRASE')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ OKX_API_KEY غير محدد أو مازال قيمة افتراضية")
        return None
        
    if not secret or secret == 'your_secret_key_here':
        print("❌ OKX_SECRET_KEY غير محدد أو مازال قيمة افتراضية")
        return None
        
    if not passphrase or passphrase == 'your_passphrase_here':
        print("❌ OKX_PASSPHRASE غير محدد أو مازال قيمة افتراضية")
        return None
    
    print("✅ جميع بيانات الاعتماد متوفرة")
    return {'apiKey': api_key, 'secret': secret, 'password': passphrase}

def test_okx_connection(credentials):
    """Test OKX exchange connection"""
    print("🌐 اختبار الاتصال بـ OKX...")
    
    try:
        exchange = ccxt.okx({
            'apiKey': credentials['apiKey'],
            'secret': credentials['secret'],
            'password': credentials['password'],
            'sandbox': False,
            'enableRateLimit': True,
            'rateLimit': 2000,
            'timeout': 30000
        })
        
        # Test connection with balance fetch
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('total', 0)
        
        print(f"✅ اتصال ناجح - الرصيد: ${usdt_balance:.2f} USDT")
        return exchange, balance
        
    except ccxt.AuthenticationError:
        print("❌ خطأ في المصادقة - تحقق من صحة المفاتيح")
        return None, None
    except ccxt.NetworkError as e:
        print(f"❌ خطأ في الشبكة: {e}")
        return None, None
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        return None, None

def test_market_data(exchange):
    """Test market data access"""
    print("📊 اختبار الوصول لبيانات السوق...")
    
    try:
        # Test ticker data
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ بيانات BTC/USDT: ${ticker['last']:.2f}")
        
        # Test orderbook
        orderbook = exchange.fetch_order_book('BTC/USDT', 5)
        print(f"✅ كتاب الطلبات: {len(orderbook['bids'])} عروض، {len(orderbook['asks'])} طلبات")
        
        return True
    except Exception as e:
        print(f"❌ فشل في الوصول لبيانات السوق: {e}")
        return False

def test_futures_access(exchange):
    """Test futures trading access"""
    print("📈 اختبار الوصول لتداول الآجلة...")
    
    try:
        # Get futures positions
        positions = exchange.fetch_positions()
        active_positions = [p for p in positions if float(p.get('size', 0)) > 0]
        
        print(f"✅ المراكز النشطة: {len(active_positions)}")
        
        # Test futures balance
        futures_balance = exchange.fetch_balance({'type': 'swap'})
        print(f"✅ رصيد الآجلة: ${futures_balance.get('USDT', {}).get('total', 0):.2f}")
        
        return True
    except Exception as e:
        print(f"❌ محدود الوصول للآجلة: {e}")
        return False

def comprehensive_system_test():
    """Run comprehensive system test"""
    print("🤖 اختبار شامل لنظام التداول")
    print("="*50)
    
    # Load environment
    if not load_environment():
        return False
    
    # Test credentials
    credentials = test_okx_credentials()
    if not credentials:
        print("\n📝 يرجى تحديث ملف .env بمفاتيح OKX الصحيحة")
        return False
    
    # Test connection
    exchange, balance = test_okx_connection(credentials)
    if not exchange:
        return False
    
    # Test market data
    market_ok = test_market_data(exchange)
    
    # Test futures
    futures_ok = test_futures_access(exchange)
    
    # Summary
    print("\n" + "="*50)
    print("📋 ملخص النتائج:")
    print("="*50)
    print(f"✅ اتصال OKX: نجح")
    print(f"{'✅' if market_ok else '❌'} بيانات السوق: {'نجح' if market_ok else 'فشل'}")
    print(f"{'✅' if futures_ok else '❌'} تداول الآجلة: {'نجح' if futures_ok else 'فشل'}")
    
    if balance:
        usdt = balance.get('USDT', {}).get('total', 0)
        print(f"💰 الرصيد الإجمالي: ${usdt:.2f} USDT")
        
        if usdt < 50:
            print("⚠️ تحذير: الرصيد أقل من الحد الأدنى للتداول ($50)")
        else:
            print("✅ الرصيد كافي للتداول")
    
    print("="*50)
    
    if market_ok and futures_ok:
        print("🎉 النظام جاهز للتداول!")
        return True
    else:
        print("⚠️ بعض الميزات قد تكون محدودة")
        return False

if __name__ == "__main__":
    success = comprehensive_system_test()
    sys.exit(0 if success else 1)