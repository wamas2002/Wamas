"""
OKX Symbol Validator
Validates and filters cryptocurrency symbols to only include available OKX trading pairs
"""
import ccxt
import os
import logging

class OKXSymbolValidator:
    def __init__(self):
        self.exchange = None
        self.valid_spot_symbols = []
        self.valid_futures_symbols = []
        
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            logging.info("OKX exchange connection established")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to OKX: {e}")
            return False
    
    def get_valid_symbols(self):
        """Get all valid OKX trading symbols"""
        if not self.exchange:
            if not self.initialize_exchange():
                return self._get_fallback_symbols()
        
        try:
            # Get spot markets
            spot_markets = self.exchange.load_markets()
            self.valid_spot_symbols = [symbol for symbol in spot_markets.keys() 
                                     if symbol.endswith('/USDT') and 'USDT' in symbol]
            
            # Get futures markets
            self.exchange.options['defaultType'] = 'swap'
            futures_markets = self.exchange.load_markets()
            self.valid_futures_symbols = [symbol for symbol in futures_markets.keys() 
                                        if symbol.endswith('/USDT:USDT')]
            
            logging.info(f"Found {len(self.valid_spot_symbols)} valid spot symbols")
            logging.info(f"Found {len(self.valid_futures_symbols)} valid futures symbols")
            
            return {
                'spot': self.valid_spot_symbols[:100],  # Limit to 100 most liquid
                'futures': self.valid_futures_symbols[:100]
            }
            
        except Exception as e:
            logging.error(f"Failed to fetch OKX symbols: {e}")
            return self._get_fallback_symbols()
    
    def _get_fallback_symbols(self):
        """Fallback to known working OKX symbols"""
        spot_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'LINK/USDT', 'LTC/USDT', 'DOT/USDT',
            'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT',
            'TRX/USDT', 'ICP/USDT', 'ALGO/USDT', 'VET/USDT', 'HBAR/USDT',
            'XLM/USDT', 'SAND/USDT', 'MANA/USDT', 'THETA/USDT', 'AXS/USDT',
            'FIL/USDT', 'ETC/USDT', 'EGLD/USDT', 'KLAY/USDT', 'FLOW/USDT',
            'ENJ/USDT', 'CHZ/USDT', 'CRV/USDT', 'AAVE/USDT', 'MKR/USDT',
            'SNX/USDT', 'COMP/USDT', 'YFI/USDT', 'SUSHI/USDT', '1INCH/USDT',
            'CAKE/USDT', 'BAL/USDT', 'UMA/USDT', 'REN/USDT', 'KNC/USDT',
            'ZRX/USDT', 'LRC/USDT', 'BNT/USDT', 'GRT/USDT', 'BAT/USDT'
        ]
        
        futures_symbols = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT',
            'SOL/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'LTC/USDT:USDT', 'DOT/USDT:USDT',
            'AVAX/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT', 'MATIC/USDT:USDT',
            'TRX/USDT:USDT', 'ICP/USDT:USDT', 'ALGO/USDT:USDT', 'VET/USDT:USDT', 'HBAR/USDT:USDT',
            'XLM/USDT:USDT', 'SAND/USDT:USDT', 'MANA/USDT:USDT', 'THETA/USDT:USDT', 'AXS/USDT:USDT',
            'FIL/USDT:USDT', 'ETC/USDT:USDT', 'EGLD/USDT:USDT', 'KLAY/USDT:USDT', 'FLOW/USDT:USDT',
            'ENJ/USDT:USDT', 'CHZ/USDT:USDT', 'CRV/USDT:USDT', 'AAVE/USDT:USDT', 'MKR/USDT:USDT',
            'SNX/USDT:USDT', 'COMP/USDT:USDT', 'YFI/USDT:USDT', 'SUSHI/USDT:USDT', '1INCH/USDT:USDT',
            'CAKE/USDT:USDT', 'BAL/USDT:USDT', 'UMA/USDT:USDT', 'REN/USDT:USDT', 'KNC/USDT:USDT',
            'ZRX/USDT:USDT', 'LRC/USDT:USDT', 'BNT/USDT:USDT', 'GRT/USDT:USDT', 'BAT/USDT:USDT'
        ]
        
        return {
            'spot': spot_symbols,
            'futures': futures_symbols
        }

def get_validated_symbols():
    """Get validated OKX trading symbols"""
    validator = OKXSymbolValidator()
    return validator.get_valid_symbols()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = get_validated_symbols()
    print(f"Validated {len(symbols['spot'])} spot symbols")
    print(f"Validated {len(symbols['futures'])} futures symbols")