
// Enhanced Fetch Error Handling
class FetchManager {
    constructor() {
        this.retryDelay = 1000;
        this.maxRetries = 3;
    }

    async safeFetch(url, options = {}, retries = 0) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout
            
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return { success: true, data };
            
        } catch (error) {
            console.log(`Fetch error for ${url}:`, error.message);
            
            if (retries < this.maxRetries && !error.name === 'AbortError') {
                console.log(`Retrying ${url} (attempt ${retries + 1}/${this.maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * (retries + 1)));
                return this.safeFetch(url, options, retries + 1);
            }
            
            return { 
                success: false, 
                error: error.message,
                fallback: this.getFallbackData(url)
            };
        }
    }

    getFallbackData(url) {
        // Return appropriate fallback data based on endpoint
        if (url.includes('/api/signals')) {
            return { signals: [], message: 'Using cached data' };
        } else if (url.includes('/api/portfolio')) {
            return { 
                total_value: 0, 
                positions: [], 
                pnl: 0,
                message: 'Portfolio data temporarily unavailable'
            };
        } else if (url.includes('/api/trading/active-positions')) {
            return { positions: [], message: 'Position data temporarily unavailable' };
        }
        return { message: 'Data temporarily unavailable' };
    }
}

// Global fetch manager
window.fetchManager = new FetchManager();

// Enhanced promise rejection handler
window.addEventListener('unhandledrejection', event => {
    console.log('Promise rejection handled:', event.reason);
    event.preventDefault(); // Prevent console spam
    
    // Show user-friendly error message
    if (event.reason && event.reason.message) {
        showNotification('Connection issue detected, retrying...', 'warning');
    }
});

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 4px;
        color: white;
        z-index: 9999;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    switch (type) {
        case 'warning':
            notification.style.backgroundColor = '#ff9800';
            break;
        case 'error':
            notification.style.backgroundColor = '#f44336';
            break;
        default:
            notification.style.backgroundColor = '#2196f3';
    }
    
    document.body.appendChild(notification);
    
    // Fade in
    setTimeout(() => notification.style.opacity = '1', 100);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => document.body.removeChild(notification), 300);
    }, 3000);
}

// Export for global use
window.showNotification = showNotification;
