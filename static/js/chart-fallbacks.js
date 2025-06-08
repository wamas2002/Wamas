/**
 * Chart Fallbacks and Error Handling
 * Provides backup chart solutions when TradingView widgets fail to load
 */

// Chart.js fallback implementation
class ChartFallback {
    constructor() {
        this.colors = {
            primary: '#3498db',
            success: '#27ae60',
            warning: '#f39c12',
            danger: '#e74c3c',
            info: '#17a2b8'
        };
    }

    createMarketOverview(containerId, symbols) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Create fallback chart with current prices
        container.innerHTML = `
            <div class="fallback-chart-container">
                <h6 class="text-center mb-3">Market Overview</h6>
                <div class="row" id="${containerId}_grid">
                    ${symbols.map(([name, symbol]) => `
                        <div class="col-md-4 col-sm-6 mb-3">
                            <div class="market-item p-3 border rounded">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong>${name}</strong>
                                        <div class="text-muted small">${symbol.replace('BINANCE:', '').replace('|1D', '')}</div>
                                    </div>
                                    <div class="text-end">
                                        <div class="price" data-symbol="${symbol}">Loading...</div>
                                        <div class="change text-success small">+0.00%</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        // Load real prices via API
        this.loadMarketPrices(containerId);
    }

    async loadMarketPrices(containerId) {
        try {
            const response = await fetch('/api/prices');
            const prices = await response.json();
            
            const container = document.getElementById(containerId);
            if (!container) return;

            // Update prices in the fallback chart
            Object.entries(prices).forEach(([symbol, price]) => {
                const priceElement = container.querySelector(`[data-symbol*="${symbol}"]`);
                if (priceElement && price > 0) {
                    priceElement.textContent = `$${price.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    })}`;
                }
            });
        } catch (error) {
            console.log('Using static price display');
        }
    }

    createPortfolioChart(containerId) {
        const canvas = document.createElement('canvas');
        canvas.id = `${containerId}_canvas`;
        canvas.width = 400;
        canvas.height = 200;
        
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        container.appendChild(canvas);

        // Create simple portfolio performance chart
        const ctx = canvas.getContext('2d');
        
        // Generate sample portfolio data
        const dates = [];
        const values = [];
        const baseValue = 156.92;
        
        for (let i = 29; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            dates.push(date.toLocaleDateString());
            
            // Simulate realistic portfolio movement
            const variation = (Math.sin(i * 0.2) + Math.random() * 0.4 - 0.2) * 0.05;
            values.push(baseValue * (1 + variation));
        }

        if (typeof Chart !== 'undefined') {
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        label: 'Portfolio Value',
                        data: values,
                        borderColor: this.colors.primary,
                        backgroundColor: this.colors.primary + '20',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        },
                        x: {
                            display: false
                        }
                    }
                }
            });
        }
    }

    createTradingChart(containerId, symbol = 'BTCUSDT') {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Create simple price chart placeholder
        container.innerHTML = `
            <div class="trading-chart-fallback p-4 text-center">
                <h5>${symbol} Price Chart</h5>
                <div class="chart-placeholder bg-secondary rounded" style="height: 300px; display: flex; align-items: center; justify-content: center;">
                    <div>
                        <div class="spinner-border text-primary mb-3" role="status"></div>
                        <div>Loading live market data...</div>
                        <small class="text-muted">Real-time price: <span id="live-price-${symbol}">--</span></small>
                    </div>
                </div>
            </div>
        `;

        // Load current price
        this.loadCurrentPrice(symbol);
    }

    async loadCurrentPrice(symbol) {
        try {
            const response = await fetch('/api/prices');
            const prices = await response.json();
            
            const priceElement = document.getElementById(`live-price-${symbol}`);
            if (priceElement && prices[symbol]) {
                priceElement.textContent = `$${prices[symbol].toLocaleString()}`;
            }
        } catch (error) {
            console.log('Price loading fallback active');
        }
    }
}

// Initialize fallback system
const chartFallback = new ChartFallback();

// TradingView widget error handler
function initializeTradingViewWithFallback(widgetConfig, containerId) {
    try {
        // Attempt to load TradingView widget
        if (typeof TradingView !== 'undefined' && TradingView.MiniWidget) {
            new TradingView.MiniWidget(widgetConfig);
        } else {
            throw new Error('TradingView not available');
        }
    } catch (error) {
        console.log('Using chart fallback for', containerId);
        
        // Use appropriate fallback based on widget type
        if (widgetConfig.symbols && widgetConfig.symbols.length > 1) {
            chartFallback.createMarketOverview(containerId, widgetConfig.symbols);
        } else {
            chartFallback.createTradingChart(containerId);
        }
    }
}

// Auto-retry TradingView loading
function retryTradingViewLoad() {
    const retryAttempts = 3;
    let attempts = 0;
    
    function attempt() {
        attempts++;
        
        if (typeof TradingView !== 'undefined') {
            console.log('TradingView loaded successfully');
            return;
        }
        
        if (attempts < retryAttempts) {
            setTimeout(attempt, 2000);
        } else {
            console.log('TradingView loading failed, using fallbacks');
            // Activate all fallbacks
            document.querySelectorAll('[id*="tradingview"]').forEach(container => {
                if (container.innerHTML.trim() === '') {
                    chartFallback.createTradingChart(container.id);
                }
            });
        }
    }
    
    attempt();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    retryTradingViewLoad();
    
    // Update prices every 30 seconds
    setInterval(() => {
        document.querySelectorAll('.fallback-chart-container').forEach(container => {
            chartFallback.loadMarketPrices(container.parentElement.id);
        });
    }, 30000);
});

// Export for global use
window.chartFallback = chartFallback;
window.initializeTradingViewWithFallback = initializeTradingViewWithFallback;