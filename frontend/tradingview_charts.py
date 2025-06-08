"""
TradingView Chart Integration for Real-time Price Visualization
"""
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, Optional, List
import json

class TradingViewCharts:
    """TradingView chart integration with real-time data and technical indicators"""
    
    def __init__(self):
        self.supported_symbols = {
            'BTCUSDT': 'BINANCE:BTCUSDT',
            'ETHUSDT': 'BINANCE:ETHUSDT', 
            'ADAUSDT': 'BINANCE:ADAUSDT',
            'BNBUSDT': 'BINANCE:BNBUSDT',
            'DOTUSDT': 'BINANCE:DOTUSDT',
            'LINKUSDT': 'BINANCE:LINKUSDT',
            'LTCUSDT': 'BINANCE:LTCUSDT',
            'XRPUSDT': 'BINANCE:XRPUSDT'
        }
        
        self.default_indicators = [
            'RSI@tv-basicstudies',
            'MACD@tv-basicstudies', 
            'BB@tv-basicstudies',
            'MA@tv-basicstudies',
            'Volume@tv-basicstudies'
        ]
    
    def render_advanced_chart(self, 
                            symbol: str = 'BTCUSDT',
                            theme: str = 'dark',
                            interval: str = '1H',
                            height: int = 600,
                            indicators: Optional[List[str]] = None,
                            hide_side_toolbar: bool = False,
                            allow_symbol_change: bool = True,
                            studies: Optional[List[str]] = None) -> None:
        """
        Render advanced TradingView chart with customizable features
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            theme: Chart theme ('light' or 'dark')
            interval: Time interval ('1m', '5m', '15m', '1H', '4H', '1D')
            height: Chart height in pixels
            indicators: List of technical indicators to display
            hide_side_toolbar: Whether to hide the side toolbar
            allow_symbol_change: Allow users to change symbols
            studies: Additional studies to apply
        """
        
        # Convert symbol to TradingView format
        tv_symbol = self.supported_symbols.get(symbol, f'BINANCE:{symbol}')
        
        # Use default indicators if none provided
        if indicators is None:
            indicators = self.default_indicators
        
        # Chart configuration
        chart_config = {
            "symbol": tv_symbol,
            "interval": interval,
            "timezone": "Etc/UTC",
            "theme": theme,
            "style": "1",  # Candlestick style
            "locale": "en",
            "toolbar_bg": "#f1f3f6" if theme == 'light' else "#131722",
            "enable_publishing": False,
            "hide_top_toolbar": False,
            "hide_legend": False,
            "save_image": True,
            "container_id": "tradingview_chart",
            "height": height,
            "width": "100%",
            "allow_symbol_change": allow_symbol_change,
            "hide_side_toolbar": hide_side_toolbar,
            "details": True,
            "hotlist": True,
            "calendar": True,
            "studies": indicators if studies is None else studies
        }
        
        # TradingView widget HTML
        tradingview_html = f"""
        <div class="tradingview-widget-container" style="height:{height}px;width:100%">
            <div id="tradingview_chart" style="height:{height-50}px;width:100%"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/symbols/{tv_symbol}/" rel="noopener" target="_blank">
                    <span class="blue-text">{symbol} Chart</span>
                </a> by TradingView
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
            {{
                "autosize": false,
                "symbol": "{tv_symbol}",
                "interval": "{interval}",
                "timezone": "Etc/UTC",
                "theme": "{theme}",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "{chart_config['toolbar_bg']}",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": true,
                "container_id": "tradingview_chart",
                "height": {height-50},
                "width": "100%",
                "allow_symbol_change": {str(allow_symbol_change).lower()},
                "hide_side_toolbar": {str(hide_side_toolbar).lower()},
                "details": true,
                "hotlist": true,
                "calendar": true,
                "studies": {json.dumps(indicators)}
            }}
            </script>
        </div>
        """
        
        components.html(tradingview_html, height=height)
    
    def render_mini_chart(self, 
                         symbol: str = 'BTCUSDT',
                         theme: str = 'dark',
                         height: int = 350) -> None:
        """
        Render mini TradingView chart for quick price overview
        """
        tv_symbol = self.supported_symbols.get(symbol, f'BINANCE:{symbol}')
        
        mini_chart_html = f"""
        <div class="tradingview-widget-container" style="height:{height}px;width:100%">
            <div class="tradingview-widget-container__widget" style="height:{height-30}px;width:100%"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/symbols/{tv_symbol}/" rel="noopener" target="_blank">
                    <span class="blue-text">{symbol}</span>
                </a> by TradingView
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
            {{
                "symbol": "{tv_symbol}",
                "width": "100%",
                "height": {height-30},
                "locale": "en",
                "dateRange": "12M",
                "colorTheme": "{theme}",
                "trendLineColor": "rgba(41, 98, 255, 1)",
                "underLineColor": "rgba(41, 98, 255, 0.3)",
                "underLineBottomColor": "rgba(41, 98, 255, 0)",
                "isTransparent": false,
                "autosize": false,
                "largeChartUrl": ""
            }}
            </script>
        </div>
        """
        
        components.html(mini_chart_html, height=height)
    
    def render_technical_analysis_widget(self, 
                                       symbol: str = 'BTCUSDT',
                                       theme: str = 'dark',
                                       height: int = 400) -> None:
        """
        Render TradingView technical analysis widget showing indicator summaries
        """
        tv_symbol = self.supported_symbols.get(symbol, f'BINANCE:{symbol}')
        
        ta_widget_html = f"""
        <div class="tradingview-widget-container" style="height:{height}px;width:100%">
            <div class="tradingview-widget-container__widget" style="height:{height-30}px;width:100%"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/symbols/{tv_symbol}/technicals/" rel="noopener" target="_blank">
                    <span class="blue-text">Technical Analysis for {symbol}</span>
                </a> by TradingView
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
            {{
                "interval": "1h",
                "width": "100%",
                "isTransparent": false,
                "height": {height-30},
                "symbol": "{tv_symbol}",
                "showIntervalTabs": true,
                "locale": "en",
                "colorTheme": "{theme}"
            }}
            </script>
        </div>
        """
        
        components.html(ta_widget_html, height=height)
    
    def render_market_overview(self, 
                             theme: str = 'dark',
                             height: int = 400,
                             symbols: Optional[List[str]] = None) -> None:
        """
        Render market overview widget with multiple cryptocurrencies
        """
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
        
        # Convert symbols to TradingView format
        tv_symbols = []
        for symbol in symbols:
            tv_symbol = self.supported_symbols.get(symbol, f'BINANCE:{symbol}')
            tv_symbols.append([tv_symbol, symbol])
        
        market_overview_html = f"""
        <div class="tradingview-widget-container" style="height:{height}px;width:100%">
            <div class="tradingview-widget-container__widget" style="height:{height-30}px;width:100%"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener" target="_blank">
                    <span class="blue-text">Market Overview</span>
                </a> by TradingView
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
            {{
                "colorTheme": "{theme}",
                "dateRange": "12M",
                "showChart": true,
                "locale": "en",
                "width": "100%",
                "height": {height-30},
                "largeChartUrl": "",
                "isTransparent": false,
                "showSymbolLogo": true,
                "showFloatingTooltip": false,
                "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
                "plotLineColorFalling": "rgba(41, 98, 255, 1)",
                "gridLineColor": "rgba(240, 243, 250, 0)",
                "scaleFontColor": "rgba(120, 123, 134, 1)",
                "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
                "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
                "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
                "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
                "symbolActiveColor": "rgba(41, 98, 255, 0.12)",
                "tabs": {json.dumps(tv_symbols)}
            }}
            </script>
        </div>
        """
        
        components.html(market_overview_html, height=height)
    
    def render_crypto_screener(self, 
                             theme: str = 'dark',
                             height: int = 500) -> None:
        """
        Render cryptocurrency screener widget
        """
        screener_html = f"""
        <div class="tradingview-widget-container" style="height:{height}px;width:100%">
            <div class="tradingview-widget-container__widget" style="height:{height-30}px;width:100%"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener" target="_blank">
                    <span class="blue-text">Cryptocurrency Screener</span>
                </a> by TradingView
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
            {{
                "width": "100%",
                "height": {height-30},
                "defaultColumn": "overview",
                "screener_type": "crypto_mkt",
                "displayCurrency": "USD",
                "colorTheme": "{theme}",
                "locale": "en"
            }}
            </script>
        </div>
        """
        
        components.html(screener_html, height=height)
    
    def render_economic_calendar(self, 
                               theme: str = 'dark',
                               height: int = 500) -> None:
        """
        Render economic calendar widget for fundamental analysis
        """
        calendar_html = f"""
        <div class="tradingview-widget-container" style="height:{height}px;width:100%">
            <div class="tradingview-widget-container__widget" style="height:{height-30}px;width:100%"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://www.tradingview.com/" rel="noopener" target="_blank">
                    <span class="blue-text">Economic Calendar</span>
                </a> by TradingView
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
            {{
                "colorTheme": "{theme}",
                "isTransparent": false,
                "width": "100%",
                "height": {height-30},
                "locale": "en",
                "importanceFilter": "-1,0,1",
                "currencyFilter": "USD,EUR,JPY,GBP,CHF,AUD,CAD,NZD,CNY"
            }}
            </script>
        </div>
        """
        
        components.html(calendar_html, height=height)
    
    def get_available_intervals(self) -> List[str]:
        """Get list of available time intervals"""
        return ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '8H', '12H', '1D', '3D', '1W', '1M']
    
    def get_available_indicators(self) -> Dict[str, str]:
        """Get dictionary of available technical indicators"""
        return {
            'Volume': 'Volume@tv-basicstudies',
            'Moving Average': 'MA@tv-basicstudies',
            'Exponential Moving Average': 'EMA@tv-basicstudies',
            'RSI': 'RSI@tv-basicstudies',
            'MACD': 'MACD@tv-basicstudies',
            'Bollinger Bands': 'BB@tv-basicstudies',
            'Stochastic': 'Stoch@tv-basicstudies',
            'Williams %R': 'WilliamsR@tv-basicstudies',
            'ATR': 'ATR@tv-basicstudies',
            'CCI': 'CCI@tv-basicstudies',
            'ADX': 'ADX@tv-basicstudies',
            'Awesome Oscillator': 'AwesomeOscillator@tv-basicstudies',
            'Momentum': 'Mom@tv-basicstudies',
            'Money Flow Index': 'MFI@tv-basicstudies',
            'OBV': 'OBV@tv-basicstudies',
            'TRIX': 'Trix@tv-basicstudies',
            'Ultimate Oscillator': 'UO@tv-basicstudies',
            'VWAP': 'VWAP@tv-basicstudies'
        }
    
    def get_supported_symbols(self) -> Dict[str, str]:
        """Get dictionary of supported trading symbols"""
        return self.supported_symbols.copy()