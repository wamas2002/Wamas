#!/usr/bin/env python3
"""
Local System Launcher - AI Trading System
تشغيل النظام الكامل على الجهاز المحلي
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

class LocalTradingSystemLauncher:
    def __init__(self):
        self.processes = {}
        self.base_dir = Path(__file__).parent
        
    def check_requirements(self):
        """Check system requirements"""
        print("🔍 فحص المتطلبات الأساسية...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            print("❌ Python 3.11+ مطلوب")
            return False
            
        # Check environment variables
        required_env = ['OKX_API_KEY', 'OKX_SECRET_KEY', 'OKX_PASSPHRASE']
        missing_env = [env for env in required_env if not os.environ.get(env)]
        
        if missing_env:
            print(f"❌ متغيرات البيئة المفقودة: {', '.join(missing_env)}")
            print("يرجى إنشاء ملف .env مع مفاتيح OKX الخاصة بك")
            return False
            
        print("✅ جميع المتطلبات متوفرة")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        print("📦 تثبيت المكتبات المطلوبة...")
        
        packages = [
            'ccxt', 'pandas', 'numpy', 'scikit-learn', 'lightgbm', 
            'xgboost', 'flask', 'flask-cors', 'flask-socketio', 
            'requests', 'psutil', 'schedule', 'streamlit', 'plotly'
        ]
        
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + packages, 
                         check=True, capture_output=True)
            print("✅ تم تثبيت جميع المكتبات")
        except subprocess.CalledProcessError as e:
            print(f"❌ فشل في تثبيت المكتبات: {e}")
            return False
        return True
    
    def run_component(self, script_name, component_name):
        """Run a system component"""
        def runner():
            try:
                print(f"🚀 تشغيل {component_name}...")
                process = subprocess.Popen([sys.executable, script_name], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE,
                                         text=True)
                self.processes[component_name] = process
                
                # Monitor output
                for line in process.stdout:
                    print(f"[{component_name}] {line.strip()}")
                    
            except Exception as e:
                print(f"❌ خطأ في {component_name}: {e}")
        
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        time.sleep(2)  # Allow component to start
    
    def launch_core_systems(self):
        """Launch core trading systems"""
        print("🎯 تشغيل الأنظمة الأساسية...")
        
        components = [
            ('elite_dashboard_fixed.py', 'لوحة التحكم الرئيسية'),
            ('live_position_monitor.py', 'مراقب المراكز المباشر'),
            ('advanced_signal_executor.py', 'منفذ الإشارات المتقدم'),
            ('advanced_position_manager.py', 'مدير المراكز المتقدم'),
            ('intelligent_profit_optimizer.py', 'محسن الأرباح الذكي')
        ]
        
        for script, name in components:
            if os.path.exists(script):
                self.run_component(script, name)
            else:
                print(f"⚠️ ملف غير موجود: {script}")
    
    def display_access_info(self):
        """Display access information"""
        print("\n" + "="*60)
        print("🌐 معلومات الوصول:")
        print("="*60)
        print("📊 لوحة التحكم الرئيسية: http://localhost:3005")
        print("📈 تحليلات المحفظة: http://localhost:5000")
        print("💹 مراقب النظام: سجلات الكونسول")
        print("="*60)
        print("⚠️ اضغط Ctrl+C لإيقاف النظام")
        print("="*60)
    
    def monitor_system(self):
        """Monitor running systems"""
        try:
            while True:
                # Check if all processes are running
                active_count = sum(1 for p in self.processes.values() 
                                 if p.poll() is None)
                print(f"📊 الأنظمة النشطة: {active_count}/{len(self.processes)}")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n🛑 إيقاف النظام...")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown all components"""
        print("🔄 إغلاق جميع المكونات...")
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ تم إغلاق {name}")
            except:
                process.kill()
                print(f"🔒 تم إنهاء {name} بالقوة")
    
    def create_env_template(self):
        """Create .env template"""
        env_content = """# مفاتيح OKX API
OKX_API_KEY=your_api_key_here
OKX_SECRET_KEY=your_secret_key_here
OKX_PASSPHRASE=your_passphrase_here

# إعدادات إضافية
RISK_PERCENTAGE=2.0
MAX_POSITIONS=3
MIN_BALANCE=50.0
"""
        with open('.env.template', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("📝 تم إنشاء ملف .env.template")
    
    def run(self):
        """Main launcher function"""
        print("🤖 نظام التداول بالذكاء الاصطناعي - التشغيل المحلي")
        print("="*60)
        
        # Check requirements
        if not self.check_requirements():
            self.create_env_template()
            print("يرجى ملء ملف .env.template وإعادة تسميته إلى .env")
            return
        
        # Install dependencies if needed
        try:
            import ccxt, pandas, flask
        except ImportError:
            if not self.install_dependencies():
                return
        
        # Launch systems
        self.launch_core_systems()
        
        # Display access info
        self.display_access_info()
        
        # Monitor system
        self.monitor_system()

if __name__ == "__main__":
    launcher = LocalTradingSystemLauncher()
    launcher.run()