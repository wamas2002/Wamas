// Portfolio data loading with proper error handling
function loadPortfolioData() {
    fetch('/api/portfolio')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data && typeof data === 'object') {
                updatePortfolioDisplay(data);
            } else {
                console.warn('Invalid portfolio data received:', data);
                showPortfolioFallback();
            }
        })
        .catch(error => {
            console.warn('Portfolio data loading failed:', error.message);
            showPortfolioFallback();
        });
}

function updatePortfolioDisplay(data) {
    try {
        // Update portfolio values safely
        const totalBalance = data.total_balance || 0;
        const cashBalance = data.cash_balance || 0;
        const positions = data.positions || [];
        
        // Update DOM elements if they exist
        const totalElement = document.getElementById('portfolio-total');
        if (totalElement) {
            totalElement.textContent = `$${totalBalance.toFixed(2)}`;
        }
        
        const cashElement = document.getElementById('portfolio-cash');
        if (cashElement) {
            cashElement.textContent = `$${cashBalance.toFixed(2)}`;
        }
        
        const positionsElement = document.getElementById('portfolio-positions');
        if (positionsElement) {
            positionsElement.textContent = positions.length;
        }
        
        console.log('Portfolio updated successfully');
    } catch (error) {
        console.warn('Error updating portfolio display:', error.message);
        showPortfolioFallback();
    }
}

function showPortfolioFallback() {
    // Show loading state instead of error
    const elements = ['portfolio-total', 'portfolio-cash', 'portfolio-positions'];
    elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = 'Loading...';
        }
    });
}

// Initialize portfolio loading with retry
function initPortfolioLoader() {
    // Load immediately
    loadPortfolioData();
    
    // Set up periodic refresh
    setInterval(loadPortfolioData, 10000); // Every 10 seconds
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPortfolioLoader);
} else {
    initPortfolioLoader();
}