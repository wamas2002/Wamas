
// Global Error Handling
window.addEventListener('error', function(e) {
    console.warn('JavaScript error caught:', e.error?.message || 'Unknown error');
});

window.addEventListener('unhandledrejection', function(e) {
    console.warn('Unhandled promise rejection:', e.reason);
    e.preventDefault(); // Prevent default browser behavior
});

// Fetch with error handling
function safeFetch(url, options = {}) {
    return fetch(url, options)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response;
        })
        .catch(error => {
            console.warn(`Fetch error for ${url}:`, error.message);
            return null;
        });
}

// WebSocket with error handling
function createSafeWebSocket(url) {
    const ws = new WebSocket(url);
    
    ws.onerror = function(error) {
        console.warn('WebSocket error:', error);
    };
    
    ws.onclose = function(event) {
        if (!event.wasClean) {
            console.warn('WebSocket closed unexpectedly, code:', event.code);
        }
    };
    
    return ws;
}
