#!/usr/bin/env python3
"""
Complete Local Trading System Setup
إعداد نظام التداول المحلي الكامل
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

def create_project_structure():
    """Create necessary directories"""
    dirs = ['logs', 'data', 'backups', 'config']
    for directory in dirs:
        Path(directory).mkdir(exist_ok=True)
    print("✅ تم إنشاء هيكل المشروع")

def setup_environment():
    """Setup environment variables"""
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# OKX API Configuration
OKX_API_KEY=your_api_key_here
OKX_SECRET_KEY=your_secret_key_here  
OKX_PASSPHRASE=your_passphrase_here

# Trading Configuration
RISK_PERCENTAGE=2.0
MAX_POSITIONS=3
MIN_BALANCE=50.0
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("📝 تم إنشاء ملف .env - يرجى إضافة مفاتيح OKX الخاصة بك")
        return False
    return True

def install_packages():
    """Install required Python packages"""
    packages = [
        'ccxt>=4.0.0', 'pandas>=2.0.0', 'numpy>=1.24.0', 'scikit-learn>=1.3.0',
        'lightgbm>=4.0.0', 'xgboost>=2.0.0', 'flask>=3.0.0', 'flask-cors>=4.0.0',
        'flask-socketio>=5.3.0', 'requests>=2.31.0', 'psutil>=5.9.0', 'schedule>=1.2.0',
        'streamlit>=1.28.0', 'plotly>=5.17.0', 'pandas-ta>=0.3.14b'
    ]
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + packages, 
                      check=True, capture_output=True)
        print("✅ تم تثبيت جميع المكتبات المطلوبة")
        return True
    except subprocess.CalledProcessError:
        print("❌ فشل في تثبيت المكتبات")
        return False

def test_okx_connection():
    """Test OKX API connection"""
    try:
        import ccxt
        from dotenv import load_dotenv
        load_dotenv()
        
        exchange = ccxt.okx({
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET_KEY'),
            'password': os.getenv('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True,
            'rateLimit': 2000
        })
        
        balance = exchange.fetch_balance()
        print(f"✅ اتصال OKX ناجح - الرصيد: ${balance.get('USDT', {}).get('total', 0):.2f}")
        return True
    except Exception as e:
        print(f"❌ فشل اتصال OKX: {e}")
        return False

def create_startup_script():
    """Create startup script for all components"""
    startup_content = """#!/bin/bash
# AI Trading System Startup Script

echo "🚀 بدء تشغيل نظام التداول بالذكاء الاصطناعي"

# Start Elite Dashboard
python elite_dashboard_fixed.py &
echo "📊 تم تشغيل لوحة التحكم على http://localhost:3005"

# Start Position Monitor  
python live_position_monitor.py &
echo "📈 تم تشغيل مراقب المراكز"

# Start Signal Executor
python advanced_signal_executor.py &
echo "⚡ تم تشغيل منفذ الإشارات"

# Start Position Manager
python advanced_position_manager.py &
echo "🎯 تم تشغيل مدير المراكز"

# Start Profit Optimizer
python intelligent_profit_optimizer.py &
echo "💎 تم تشغيل محسن الأرباح"

echo "✅ تم تشغيل جميع المكونات"
echo "🌐 الوصول للوحة التحكم: http://localhost:3005"
echo "⚠️ اضغط Ctrl+C لإيقاف النظام"

wait
"""
    
    with open('start_trading.sh', 'w') as f:
        f.write(startup_content)
    
    # Make executable
    os.chmod('start_trading.sh', 0o755)
    print("✅ تم إنشاء سكريبت التشغيل: ./start_trading.sh")

def main():
    """Main setup function"""
    print("🤖 إعداد نظام التداول بالذكاء الاصطناعي")
    print("="*50)
    
    # Create project structure
    create_project_structure()
    
    # Setup environment
    if not setup_environment():
        print("⚠️ يرجى تحديث ملف .env بمفاتيح OKX ثم إعادة تشغيل الإعداد")
        return
    
    # Install packages
    if not install_packages():
        return
    
    # Test OKX connection
    if not test_okx_connection():
        print("⚠️ يرجى التحقق من مفاتيح OKX في ملف .env")
        return
    
    # Create startup script
    create_startup_script()
    
    print("\n" + "="*50)
    print("✅ تم الإعداد بنجاح!")
    print("🚀 لبدء التشغيل:")
    print("   ./start_trading.sh")
    print("🌐 لوحة التحكم: http://localhost:3005")
    print("="*50)

if __name__ == "__main__":
    main()