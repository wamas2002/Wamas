#!/bin/bash
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
