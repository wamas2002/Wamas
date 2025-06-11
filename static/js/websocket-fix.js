
// WebSocket Error Handling and Recovery
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isConnecting = false;
        this.init();
    }

    init() {
        this.connect();
    }

    connect() {
        if (this.isConnecting) return;
        
        this.isConnecting = true;
        
        try {
            // Use relative WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected successfully');
                this.reconnectAttempts = 0;
                this.isConnecting = false;
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.log('Non-JSON WebSocket message:', event.data);
                }
            };
            
            this.ws.onerror = (error) => {
                console.log('WebSocket error handled gracefully');
                this.isConnecting = false;
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket connection closed');
                this.isConnecting = false;
                this.scheduleReconnect();
            };
            
        } catch (error) {
            console.log('WebSocket connection attempt failed, continuing without real-time updates');
            this.isConnecting = false;
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Attempting WebSocket reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.log('WebSocket reconnection attempts exhausted, continuing in polling mode');
        }
    }

    handleMessage(data) {
        // Handle incoming WebSocket messages
        if (data.type === 'signal_update') {
            window.dispatchEvent(new CustomEvent('signalUpdate', { detail: data }));
        } else if (data.type === 'portfolio_update') {
            window.dispatchEvent(new CustomEvent('portfolioUpdate', { detail: data }));
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
}

// Initialize WebSocket manager globally
window.wsManager = new WebSocketManager();
