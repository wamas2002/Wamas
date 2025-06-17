# Copy Instructions for Local Deployment

## Step 1: Create Project Folder
```bash
mkdir ai-trading-system
cd ai-trading-system
```

## Step 2: Copy Essential Files
Copy the following files from Replit to your project folder:

### A) Core System Files:
- `elite_dashboard_fixed.py`
- `live_position_monitor.py`
- `advanced_signal_executor.py`
- `advanced_position_manager.py`
- `intelligent_profit_optimizer.py`
- `comprehensive_system_monitor.py`
- `master_portfolio_dashboard.py`

### B) Data Validation Files:
- `okx_data_validator.py`
- `advanced_portfolio_analytics.py`

### C) Setup Files:
- `setup_local_trading.py`
- `local_system_launcher.py`
- `test_okx_connection.py`
- `local_requirements.txt`
- `.env.template`

### D) Documentation:
- `DEPLOYMENT_GUIDE_ENGLISH.md`
- `QUICK_START_ENGLISH.md`

### E) Interface Files:
```bash
mkdir templates
```
Copy: `templates/elite_dashboard_production.html`

## Step 3: Environment Setup
```bash
# Create environment file
cp .env.template .env

# Edit .env file and add real OKX keys
nano .env
```

Add your keys:
```env
OKX_API_KEY=your_real_api_key_here
OKX_SECRET_KEY=your_real_secret_key_here  
OKX_PASSPHRASE=your_real_passphrase_here
```

## Step 4: Install Libraries
```bash
# Check Python version (3.11+ required)
python3 --version

# Install libraries
pip install -r local_requirements.txt
```

## Step 5: Test Setup
```bash
python test_okx_connection.py
```

## Step 6: Run System
```bash
# Automatic startup
python local_system_launcher.py

# Or manual setup
python setup_local_trading.py
```

## Access Interfaces:
- Main Dashboard: http://localhost:3005
- Portfolio Analytics: http://localhost:5000

## Important Notes:
- Ensure stable internet connection
- Use only real OKX keys
- Keep backup of .env file
- Monitor system logs for errors

## Troubleshooting:
- Connection fails: Check OKX keys
- Library install fails: Use pip3 instead of pip
- Interface not working: Verify ports 3005 and 5000 are available

## Current System Performance:
- Balance: $192.15 USDT
- Recent profit: +$0.14 on ATOM/USDT trade
- Position successfully closed automatically
- All components operational