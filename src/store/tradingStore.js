import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

export const useTradingStore = create(
  subscribeWithSelector((set, get) => ({
    // Dashboard Data
    portfolioData: null,
    signals: [],
    trades: [],
    confidence: 88,
    layout: null,
    
    // Settings
    settings: {
      soundEnabled: true,
      vibrationEnabled: true,
      autoRefresh: true,
      refreshInterval: 3000,
      chartTimeframe: '1h',
      selectedSymbols: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
      developerMode: false,
    },
    
    // Strategy Testing
    strategyTests: [],
    activeStrategy: null,
    
    // Filters
    filters: {
      symbol: '',
      minConfidence: 70,
      signalType: 'all', // 'buy', 'sell', 'all'
      strategyTag: '',
    },
    
    // Multi-chart view
    chartMode: 'single', // 'single', 'dual', 'triple'
    
    // Actions
    setPortfolioData: (data) => set({ portfolioData: data }),
    setSignals: (signals) => set({ signals }),
    setTrades: (trades) => set({ trades }),
    setConfidence: (confidence) => set({ confidence }),
    setLayout: (layout) => set({ layout }),
    
    updateSettings: (newSettings) => 
      set((state) => ({ 
        settings: { ...state.settings, ...newSettings } 
      })),
    
    setFilters: (filters) => 
      set((state) => ({ 
        filters: { ...state.filters, ...filters } 
      })),
    
    setChartMode: (mode) => set({ chartMode: mode }),
    
    addStrategyTest: (test) => 
      set((state) => ({ 
        strategyTests: [...state.strategyTests, test] 
      })),
    
    setActiveStrategy: (strategy) => set({ activeStrategy: strategy }),
    
    // Computed values
    getFilteredSignals: () => {
      const { signals, filters } = get();
      return signals.filter(signal => {
        const symbolMatch = !filters.symbol || 
          signal.symbol.toLowerCase().includes(filters.symbol.toLowerCase());
        const confidenceMatch = signal.confidence >= filters.minConfidence;
        const typeMatch = filters.signalType === 'all' || 
          signal.action.toLowerCase() === filters.signalType.toLowerCase();
        return symbolMatch && confidenceMatch && typeMatch;
      });
    },
    
    getPortfolioSummary: () => {
      const { portfolioData } = get();
      if (!portfolioData) return null;
      
      return {
        totalValue: portfolioData.balance || 0,
        dayChange: portfolioData.dayChange || 0,
        dayChangePercent: portfolioData.dayChangePercent || 0,
        positions: portfolioData.positions || [],
      };
    },
  }))
);