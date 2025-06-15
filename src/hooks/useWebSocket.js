import { useEffect, useRef } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { toast } from 'react-toastify';
import io from 'socket.io-client';

export const useWebSocket = () => {
  const socketRef = useRef(null);
  const { 
    setPortfolioData, 
    setSignals, 
    setTrades, 
    setConfidence,
    settings,
    addSignal,
    addTrade
  } = useTradingStore();

  useEffect(() => {
    // Connect to existing Flask-SocketIO endpoints
    const connectSocket = () => {
      try {
        socketRef.current = io('http://localhost:3000', {
          transports: ['websocket', 'polling']
        });

        socketRef.current.on('connect', () => {
          console.log('WebSocket connected to trading backend');
        });

        socketRef.current.on('portfolio_update', (data) => {
          setPortfolioData(data);
        });

        socketRef.current.on('new_signal', (signal) => {
          addSignal(signal);
          if (settings.soundEnabled && signal.confidence >= 80) {
            // Play notification sound for high-confidence signals
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSsFJHfH8N2QQAoUXrTp66hVFApGn+DyvmgfCi6Gyu3deSEFJI');
            audio.volume = 0.3;
            audio.play().catch(() => {}); // Ignore errors if audio fails
          }
          
          if (settings.vibrationEnabled && 'vibrate' in navigator) {
            navigator.vibrate([100, 50, 100]);
          }

          toast.success(`New ${signal.action} signal: ${signal.symbol} (${signal.confidence}%)`, {
            position: "top-right",
            autoClose: 5000,
          });
        });

        socketRef.current.on('trade_executed', (trade) => {
          addTrade(trade);
          toast.info(`Trade executed: ${trade.side} ${trade.symbol}`, {
            position: "top-right",
            autoClose: 3000,
          });
        });

        socketRef.current.on('confidence_update', (data) => {
          setConfidence(data.confidence);
        });

        socketRef.current.on('disconnect', () => {
          console.log('WebSocket disconnected');
        });

        socketRef.current.on('connect_error', (error) => {
          console.log('WebSocket connection error:', error);
        });

      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
      }
    };

    connectSocket();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [setPortfolioData, setSignals, setTrades, setConfidence, addSignal, addTrade, settings]);

  return socketRef.current;
};