#!/usr/bin/env python3
"""
Create Complete Deployment Package
إنشاء حزمة النشر الكاملة
"""

import os
import shutil
from pathlib import Path

def create_deployment_package():
    """Create complete deployment package"""
    
    # Create package directory
    package_dir = Path("ai_trading_deployment")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # Create subdirectories
    (package_dir / "templates").mkdir()
    (package_dir / "logs").mkdir()
    (package_dir / "data").mkdir()
    (package_dir / "backups").mkdir()
    
    # Core system files to copy
    core_files = [
        'elite_dashboard_fixed.py',
        'live_position_monitor.py', 
        'advanced_signal_executor.py',
        'advanced_position_manager.py',
        'intelligent_profit_optimizer.py',
        'comprehensive_system_monitor.py',
        'master_portfolio_dashboard.py',
        'okx_data_validator.py',
        'advanced_portfolio_analytics.py'
    ]
    
    # Setup and configuration files
    setup_files = [
        'setup_local_trading.py',
        'local_system_launcher.py',
        'test_okx_connection.py',
        'local_requirements.txt',
        '.env.template'
    ]
    
    # Documentation files
    doc_files = [
        'LOCAL_DEPLOYMENT_GUIDE_AR.md',
        'QUICK_START_ARABIC.md',
        'copy_instructions_arabic.md',
        'deployment_files_list.txt'
    ]
    
    # Template files
    template_files = [
        'templates/elite_dashboard_production.html'
    ]
    
    print("📦 إنشاء حزمة النشر...")
    
    # Copy core files
    for file in core_files:
        if os.path.exists(file):
            shutil.copy2(file, package_dir)
            print(f"✅ تم نسخ {file}")
        else:
            print(f"⚠️ ملف غير موجود: {file}")
    
    # Copy setup files
    for file in setup_files:
        if os.path.exists(file):
            shutil.copy2(file, package_dir)
            print(f"✅ تم نسخ {file}")
    
    # Copy documentation
    for file in doc_files:
        if os.path.exists(file):
            shutil.copy2(file, package_dir)
            print(f"✅ تم نسخ {file}")
    
    # Copy template files
    for file in template_files:
        if os.path.exists(file):
            dest_file = package_dir / file
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest_file)
            print(f"✅ تم نسخ {file}")
    
    # Create startup script
    startup_script = package_dir / "start_system.sh"
    with open(startup_script, 'w') as f:
        f.write("""#!/bin/bash
# AI Trading System Startup
echo "🚀 بدء نظام التداول بالذكاء الاصطناعي"

# Check environment
if [ ! -f ".env" ]; then
    echo "❌ ملف .env غير موجود. يرجى إنشاؤه من .env.template"
    exit 1
fi

# Start components
python elite_dashboard_fixed.py &
python live_position_monitor.py &
python advanced_signal_executor.py &
python advanced_position_manager.py &
python intelligent_profit_optimizer.py &

echo "✅ تم تشغيل جميع المكونات"
echo "🌐 لوحة التحكم: http://localhost:3005"
echo "📊 التحليلات: http://localhost:5000"
echo "⚠️ اضغط Ctrl+C للإيقاف"

wait
""")
    
    # Make startup script executable
    os.chmod(startup_script, 0o755)
    
    # Create README
    readme_file = package_dir / "README_ARABIC.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("""# نظام التداول بالذكاء الاصطناعي

## التشغيل السريع:
1. انسخ مفاتيح OKX إلى ملف .env
2. نفذ: `pip install -r local_requirements.txt`
3. نفذ: `python test_okx_connection.py`
4. نفذ: `./start_system.sh`

## الوصول:
- لوحة التحكم: http://localhost:3005
- التحليلات: http://localhost:5000

## للمساعدة:
اقرأ QUICK_START_ARABIC.md
""")
    
    print("\n" + "="*50)
    print("✅ تم إنشاء حزمة النشر بنجاح!")
    print(f"📁 المجلد: {package_dir.absolute()}")
    print("📋 المحتويات:")
    
    # List contents
    for item in sorted(package_dir.rglob("*")):
        if item.is_file():
            print(f"   📄 {item.relative_to(package_dir)}")
    
    print("\n🚀 لبدء التشغيل:")
    print(f"cd {package_dir}")
    print("./start_system.sh")
    print("="*50)

if __name__ == "__main__":
    create_deployment_package()