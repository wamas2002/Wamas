#!/usr/bin/env python3
"""
Verify AI Signal Authenticity
"""
from okx_data_validator import OKXDataValidator
import json

def verify_signal_authenticity():
    validator = OKXDataValidator()
    signals = validator.get_authentic_signals()

    print("=== AI SIGNAL AUTHENTICITY VERIFICATION ===")
    print(f"Total signals generated: {len(signals)}")
    if signals:
        print(f"Data source: {signals[0]['source']}")
        print(f"Validation status: {signals[0]['validated']}")
        print()

        for i, signal in enumerate(signals[:5]):
            print(f"Signal {i+1}:")
            print(f"  Symbol: {signal['symbol']}")
            print(f"  Action: {signal['action']}")
            print(f"  Confidence: {signal['confidence']}%")
            print(f"  Price: ${signal['price']:,.2f}")
            print(f"  Volume Surge: {signal['volume_surge']}x")
            print(f"  Source: {signal['source']}")
            print()
    else:
        print("No signals generated")

if __name__ == "__main__":
    verify_signal_authenticity()