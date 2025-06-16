# دليل التشغيل المحلي - نظام التداول بالذكاء الاصطناعي

## المتطلبات الأساسية

### 1. Python 3.11+
```bash
python --version
```

### 2. المكتبات المطلوبة
```bash
pip install ccxt pandas numpy scikit-learn lightgbm xgboost flask flask-cors flask-socketio requests psutil schedule streamlit plotly
```

### 3. متغيرات البيئة (Environment Variables)
إنشاء ملف `.env` في المجلد الرئيسي:
```env
OKX_API_KEY=your_okx_api_key_here
OKX_SECRET_KEY=your_okx_secret_key_here
OKX_PASSPHRASE=your_okx_passphrase_here
```

## الملفات الأساسية للتشغيل المحلي

### 1. تشغيل لوحة التحكم الرئيسية
```bash
python elite_dashboard_fixed.py
```
- الوصول: http://localhost:3005
- لوحة تحكم احترافية مع بيانات OKX الحقيقية

### 2. تشغيل مراقب المحفظة المباشر
```bash
python live_position_monitor.py
```
- مراقبة المراكز المفتوحة في الوقت الفعلي
- تتبع الأرباح والخسائر

### 3. تشغيل منفذ الإشارات المتقدم
```bash
python advanced_signal_executor.py
```
- تنفيذ الصفقات تلقائياً بناء على الإشارات عالية الجودة

### 4. تشغيل مدير المراكز المتقدم
```bash
python advanced_position_manager.py
```
- إدارة ذكية للمراكز مع استراتيجيات الخروج

### 5. تشغيل محسن الأرباح الذكي
```bash
python intelligent_profit_optimizer.py
```
- تحسين الأرباح باستخدام التوقيت الذكي

## تشغيل النظام الكامل

### الطريقة 1: تشغيل منفصل
```bash
# Terminal 1 - لوحة التحكم
python elite_dashboard_fixed.py

# Terminal 2 - مراقب المراكز
python live_position_monitor.py

# Terminal 3 - منفذ الإشارات
python advanced_signal_executor.py

# Terminal 4 - مدير المراكز
python advanced_position_manager.py

# Terminal 5 - محسن الأرباح
python intelligent_profit_optimizer.py
```

### الطريقة 2: تشغيل مجمع
```bash
python local_system_launcher.py
```

## إعدادات الحماية

### 1. إدارة المخاطر
- الحد الأقصى للمخاطرة: 2% من المحفظة لكل صفقة
- وقف الخسارة التلقائي: مفعل
- جني الأرباح التلقائي: مفعل

### 2. حدود التداول
- الحد الأدنى للرصيد: $50 USDT
- الحد الأقصى للمراكز المفتوحة: 3 مراكز
- الحد الأقصى للرافعة المالية: 10x

## مراقبة الأداء

### واجهات المراقبة:
1. **لوحة التحكم الرئيسية**: http://localhost:3005
2. **تحليلات المحفظة**: http://localhost:5000
3. **مراقب النظام**: سجلات الكونسول

### المقاييس المهمة:
- رصيد المحفظة الإجمالي
- عدد المراكز المفتوحة
- الأرباح/الخسائر غير المحققة
- معدل نجاح الإشارات
- كفاءة النظام

## استكشاف الأخطاء

### 1. مشاكل الاتصال بـ OKX
```bash
# التحقق من صحة مفاتيح API
python test_okx_connection.py
```

### 2. مشاكل قاعدة البيانات
```bash
# إعادة إنشاء قواعد البيانات
python reset_databases.py
```

### 3. مشاكل الذاكرة
```bash
# مراقبة استخدام الذاكرة
python system_monitor.py
```

## الأمان والنسخ الاحتياطي

### 1. نسخ احتياطي من قواعد البيانات
```bash
python backup_system.py
```

### 2. تشفير المفاتيح
- استخدم متغيرات البيئة فقط
- لا تحفظ المفاتيح في الكود مباشرة

### 3. مراقبة الأمان
- فحص دوري للاتصالات
- تسجيل جميع العمليات
- تنبيهات الأمان التلقائية

## التحديثات والصيانة

### تحديث النظام:
```bash
git pull origin main
pip install -r requirements.txt
python update_system.py
```

### صيانة دورية:
- تنظيف قواعد البيانات أسبوعياً
- فحص الأداء شهرياً  
- تحديث النماذج كل 3 أشهر

## الدعم الفني

### ملفات السجل:
- `logs/system.log` - سجل النظام العام
- `logs/trading.log` - سجل التداول
- `logs/errors.log` - سجل الأخطاء

### معلومات الاتصال:
- التوثيق: README.md
- الأمثلة: examples/
- الاختبارات: tests/

---

**تحذير**: هذا نظام تداول حقيقي يستخدم أموال فعلية. تأكد من فهم المخاطر قبل التشغيل.