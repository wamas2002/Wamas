import os
import ccxt
from okx_data_validator import OKXDataValidator

# Test OKX connection
exchange = ccxt.okx({
    'apiKey': os.environ.get('OKX_API_KEY'),
    'secret': os.environ.get('OKX_SECRET_KEY'),
    'password': os.environ.get('OKX_PASSPHRASE'),
    'sandbox': False,
})

balance = exchange.fetch_balance()
print('Direct OKX Balance:', balance['USDT']['total'])

# Test validator
validator = OKXDataValidator()
portfolio = validator.get_authentic_portfolio()
print('Validator Balance:', portfolio['balance'])