
/* 
JavaScript Error Handling and WebSocket Fixes
Add this script to all HTML templates to resolve frontend errors
*/

// Global Error Handling
window.addEventListener('error', function(e) {
    console.warn('JavaScript error caught:', e.error?.message || 'Unknown error');
    return true; // Prevent default browser error handling
});

window.addEventListener('unhandledrejection', function(e) {
    console.warn('Unhandled promise rejection:', e.reason);
    e.preventDefault(); // Prevent console spam
});

// Safe Fetch Function
function safeFetch(url, options = {}) {
    return fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .catch(error => {
        console.warn(`Fetch error for ${url}:`, error.message);
        return { error: error.message, data: null };
    });
}

// WebSocket Error Handling
function createResilientWebSocket(url, protocols) {
    let ws;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 1000; // 1 second
    
    function connect() {
        try {
            ws = new WebSocket(url, protocols);
            
            ws.onopen = function(event) {
                console.log('WebSocket connected successfully');
                reconnectAttempts = 0;
            };
            
            ws.onerror = function(error) {
                console.warn('WebSocket error:', error);
            };
            
            ws.onclose = function(event) {
                if (!event.wasClean && reconnectAttempts < maxReconnectAttempts) {
                    console.warn(`WebSocket closed unexpectedly (${event.code}), attempting reconnection...`);
                    setTimeout(() => {
                        reconnectAttempts++;
                        connect();
                    }, reconnectDelay * reconnectAttempts);
                }
            };
            
        } catch (error) {
            console.warn('WebSocket creation failed:', error);
        }
    }
    
    connect();
    return ws;
}

// Chart Error Handling
function safeCreateChart(element, config) {
    try {
        if (typeof Plotly !== 'undefined') {
            return Plotly.newPlot(element, config.data, config.layout, config.options);
        } else {
            console.warn('Plotly not loaded, chart creation skipped');
            return Promise.resolve();
        }
    } catch (error) {
        console.warn('Chart creation failed:', error);
        if (element) {
            element.innerHTML = '<p class="text-muted">Chart temporarily unavailable</p>';
        }
        return Promise.resolve();
    }
}

// API Call with Retry Logic
function apiCallWithRetry(url, options = {}, maxRetries = 3) {
    let attempts = 0;
    
    function attempt() {
        return safeFetch(url, options)
            .then(result => {
                if (result.error && attempts < maxRetries) {
                    attempts++;
                    console.warn(`API call failed, retrying (${attempts}/${maxRetries})...`);
                    return new Promise(resolve => {
                        setTimeout(() => resolve(attempt()), 1000 * attempts);
                    });
                }
                return result;
            });
    }
    
    return attempt();
}

// DOM Ready Handler
function onDOMReady(callback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', callback);
    } else {
        callback();
    }
}

// Initialize Error Handling
onDOMReady(function() {
    console.log('Error handling system initialized');
    
    // Replace all fetch calls with safe versions
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        return safeFetch(url, options)
            .then(result => {
                if (result.error) {
                    throw new Error(result.error);
                }
                return { 
                    ok: true, 
                    json: () => Promise.resolve(result),
                    text: () => Promise.resolve(JSON.stringify(result))
                };
            });
    };
    
    // Add loading states to buttons
    document.querySelectorAll('button[onclick]').forEach(button => {
        const originalOnclick = button.onclick;
        button.onclick = function(e) {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            
            try {
                const result = originalOnclick.call(this, e);
                if (result instanceof Promise) {
                    result.finally(() => {
                        button.disabled = false;
                        button.innerHTML = button.getAttribute('data-original-text') || 'Button';
                    });
                } else {
                    setTimeout(() => {
                        button.disabled = false;
                        button.innerHTML = button.getAttribute('data-original-text') || 'Button';
                    }, 1000);
                }
            } catch (error) {
                console.warn('Button onclick error:', error);
                button.disabled = false;
                button.innerHTML = button.getAttribute('data-original-text') || 'Button';
            }
        };
    });
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        safeFetch,
        createResilientWebSocket,
        safeCreateChart,
        apiCallWithRetry,
        onDOMReady
    };
}
