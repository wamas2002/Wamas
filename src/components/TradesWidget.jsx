import React, { useState } from 'react';
import { useTradingStore } from '../store/tradingStore';

const TradesWidget = () => {
  const { trades } = useTradingStore();
  const [filter, setFilter] = useState('all'); // 'all', 'profitable', 'loss'

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(value);
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getFilteredTrades = () => {
    if (!trades || trades.length === 0) return [];
    
    return trades.filter(trade => {
      if (filter === 'profitable') return trade.pnl > 0;
      if (filter === 'loss') return trade.pnl < 0;
      return true;
    });
  };

  const filteredTrades = getFilteredTrades();

  const getTradeIcon = (side) => {
    return side === 'BUY' ? '↗' : '↘';
  };

  const getTradeColor = (side) => {
    return side === 'BUY' ? 'text-green-400' : 'text-red-400';
  };

  const getPnlColor = (pnl) => {
    return pnl >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-400';
      case 'pending': return 'text-yellow-400';
      case 'cancelled': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const calculateStats = () => {
    if (filteredTrades.length === 0) return { totalPnl: 0, winRate: 0, totalTrades: 0 };
    
    const totalPnl = filteredTrades.reduce((sum, trade) => sum + (trade.pnl || 0), 0);
    const profitableTrades = filteredTrades.filter(trade => trade.pnl > 0).length;
    const winRate = (profitableTrades / filteredTrades.length) * 100;
    
    return { totalPnl, winRate, totalTrades: filteredTrades.length };
  };

  const stats = calculateStats();

  return (
    <div key="trades" className="bg-gray-900 border border-gray-700 rounded-xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Recent Trades</h3>
        <div className="text-xs text-gray-400">
          {filteredTrades.length} trades
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-3 gap-3 mb-4 p-3 bg-gray-800 rounded-lg">
        <div className="text-center">
          <div className={`text-lg font-bold ${getPnlColor(stats.totalPnl)}`}>
            {formatCurrency(stats.totalPnl)}
          </div>
          <div className="text-xs text-gray-400">Total P&L</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-400">
            {stats.winRate.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400">Win Rate</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-white">
            {stats.totalTrades}
          </div>
          <div className="text-xs text-gray-400">Trades</div>
        </div>
      </div>

      {/* Filter */}
      <div className="flex space-x-1 mb-4">
        {[
          { key: 'all', label: 'All' },
          { key: 'profitable', label: 'Profitable' },
          { key: 'loss', label: 'Loss' }
        ].map(f => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              filter === f.key
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Trades List */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {filteredTrades.length > 0 ? (
          filteredTrades.slice(0, 20).map((trade, index) => (
            <div
              key={trade.id || `trade-${index}`}
              className="bg-gray-800 border border-gray-700 rounded-lg p-3 hover:bg-gray-750 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <span className={`text-lg ${getTradeColor(trade.side)}`}>
                    {getTradeIcon(trade.side)}
                  </span>
                  <span className="font-medium text-white">{trade.symbol}</span>
                  <span className={`text-sm font-bold ${getTradeColor(trade.side)}`}>
                    {trade.side}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-xs text-gray-400">
                    {formatTime(trade.timestamp)}
                  </div>
                  <div className={`text-xs ${getStatusColor(trade.status)}`}>
                    {trade.status || 'completed'}
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-4 gap-2 text-sm">
                <div>
                  <div className="text-xs text-gray-400">Price</div>
                  <div className="text-white">
                    {formatCurrency(trade.price)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Amount</div>
                  <div className="text-white">
                    {formatCurrency(trade.amount || trade.size)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Fee</div>
                  <div className="text-gray-300">
                    {formatCurrency(trade.fee || 0)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">P&L</div>
                  <div className={`font-bold ${getPnlColor(trade.pnl || 0)}`}>
                    {formatCurrency(trade.pnl || 0)}
                  </div>
                </div>
              </div>

              {trade.confidence && (
                <div className="mt-2 flex items-center space-x-2">
                  <div className="text-xs text-gray-400">Confidence:</div>
                  <div className="text-xs text-yellow-400 font-medium">
                    {trade.confidence}%
                  </div>
                  {trade.strategy && (
                    <>
                      <div className="text-xs text-gray-400">•</div>
                      <div className="text-xs text-blue-400">
                        {trade.strategy}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          ))
        ) : (
          <div className="flex items-center justify-center h-32 text-gray-400">
            <div className="text-center">
              <div className="text-2xl mb-2">📈</div>
              <div>No trades found</div>
              <div className="text-xs mt-1">
                {filter !== 'all' ? 'Try adjusting your filter' : 'Trades will appear here once executed'}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradesWidget;