import { useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { useTradingStore } from '../store/tradingStore';
import { toast } from 'react-toastify';

export const useWebSocket = () => {
  const socketRef = useRef(null);
  const { settings, setSignals, setTrades, setPortfolioData, setConfidence } = useTradingStore();

  useEffect(() => {
    // Connect to existing Flask-SocketIO backend
    socketRef.current = io('http://localhost:3000', {
      transports: ['websocket', 'polling']
    });

    socketRef.current.on('connect', () => {
      console.log('Connected to trading backend');
    });

    socketRef.current.on('signal_update', (data) => {
      setSignals(data.signals || []);
      
      if (data.signals && data.signals.length > 0) {
        const latestSignal = data.signals[0];
        if (settings.soundEnabled) {
          playNotificationSound();
        }
        toast.success(`New ${latestSignal.action} signal: ${latestSignal.symbol} (${latestSignal.confidence}%)`);
      }
    });

    socketRef.current.on('trade_executed', (data) => {
      setTrades(prev => [data, ...prev.slice(0, 49)]); // Keep last 50 trades
      if (settings.soundEnabled) {
        playTradeSound();
      }
      toast.info(`Trade executed: ${data.symbol} ${data.side} - $${data.amount}`);
    });

    socketRef.current.on('portfolio_update', (data) => {
      setPortfolioData(data);
    });

    socketRef.current.on('confidence_update', (data) => {
      setConfidence(data.confidence);
    });

    socketRef.current.on('profit_hit', (data) => {
      toast.success(`Profit target hit: ${data.symbol} +$${data.profit}`);
      if (settings.soundEnabled) {
        playProfitSound();
      }
    });

    socketRef.current.on('stop_loss_triggered', (data) => {
      toast.error(`Stop loss triggered: ${data.symbol} -$${data.loss}`);
      if (settings.soundEnabled) {
        playLossSound();
      }
    });

    socketRef.current.on('drawdown_warning', (data) => {
      toast.warn(`Drawdown warning: ${data.percentage}% - Consider reducing position sizes`);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [settings.soundEnabled, setSignals, setTrades, setPortfolioData, setConfidence]);

  const playNotificationSound = () => {
    try {
      const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRgoRYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRg==');
      audio.volume = 0.3;
      audio.play();
    } catch (e) {
      console.log('Audio playback not supported');
    }
  };

  const playTradeSound = () => {
    try {
      const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRgoRYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRg==');
      audio.volume = 0.5;
      audio.play();
    } catch (e) {
      console.log('Audio playback not supported');
    }
  };

  const playProfitSound = () => {
    try {
      const audio = new Audio('data:audio/wav;base64,UklGRmYGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRgoRYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRg==');
      audio.volume = 0.6;
      audio.play();
    } catch (e) {
      console.log('Audio playback not supported');
    }
  };

  const playLossSound = () => {
    try {
      const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRgoRYrTp66hVFApGn+DyvmQdByaN0/PWfBgMJG/E8OOXRg==');
      audio.volume = 0.4;
      audio.play();
    } catch (e) {
      console.log('Audio playback not supported');
    }
  };

  return {
    socket: socketRef.current,
    isConnected: socketRef.current?.connected || false
  };
};