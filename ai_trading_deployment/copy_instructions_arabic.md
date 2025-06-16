# تعليمات النسخ للتشغيل المحلي

## الخطوة 1: إنشاء مجلد المشروع
```bash
mkdir ai-trading-system
cd ai-trading-system
```

## الخطوة 2: نسخ الملفات الأساسية
انسخ الملفات التالية من Replit إلى مجلد المشروع:

### أ) ملفات النظام الأساسية:
- `elite_dashboard_fixed.py`
- `live_position_monitor.py`
- `advanced_signal_executor.py`
- `advanced_position_manager.py`
- `intelligent_profit_optimizer.py`
- `comprehensive_system_monitor.py`
- `master_portfolio_dashboard.py`

### ب) ملفات التحقق والبيانات:
- `okx_data_validator.py`
- `advanced_portfolio_analytics.py`

### ج) ملفات الإعداد:
- `setup_local_trading.py`
- `local_system_launcher.py`
- `test_okx_connection.py`
- `local_requirements.txt`
- `.env.template`

### د) الأدلة:
- `LOCAL_DEPLOYMENT_GUIDE_AR.md`
- `QUICK_START_ARABIC.md`

### هـ) ملفات الواجهة:
```bash
mkdir templates
```
انسخ: `templates/elite_dashboard_production.html`

## الخطوة 3: إعداد البيئة
```bash
# إنشاء ملف البيئة
cp .env.template .env

# تحرير ملف .env وإضافة مفاتيح OKX الحقيقية
nano .env
```

أضف مفاتيحك:
```env
OKX_API_KEY=your_real_api_key_here
OKX_SECRET_KEY=your_real_secret_key_here  
OKX_PASSPHRASE=your_real_passphrase_here
```

## الخطوة 4: تثبيت المكتبات
```bash
# تثبيت Python 3.11+ إذا لم يكن مثبتاً
python3 --version

# تثبيت المكتبات
pip install -r local_requirements.txt
```

## الخطوة 5: اختبار الإعداد
```bash
python test_okx_connection.py
```

## الخطوة 6: تشغيل النظام
```bash
# تشغيل تلقائي
python local_system_launcher.py

# أو تشغيل يدوي
python setup_local_trading.py
```

## الوصول للواجهات:
- لوحة التحكم: http://localhost:3005
- تحليلات المحفظة: http://localhost:5000

## ملاحظات مهمة:
- تأكد من وجود اتصال إنترنت مستقر
- استخدم مفاتيح OKX الحقيقية فقط
- احتفظ بنسخة احتياطية من ملف .env
- راقب سجلات النظام للتأكد من عدم وجود أخطاء

## استكشاف الأخطاء:
- إذا فشل الاتصال: تحقق من مفاتيح OKX
- إذا فشل تثبيت المكتبات: استخدم pip3 بدلاً من pip
- إذا لم تعمل الواجهة: تحقق من أن المنافذ 3005 و 5000 متاحة