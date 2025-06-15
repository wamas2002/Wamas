import React, { useState } from 'react';
import { useTradingStore } from '../store/tradingStore';

const SignalsWidget = () => {
  const { signals, filters, setFilters, getFilteredSignals } = useTradingStore();
  const [localFilters, setLocalFilters] = useState(filters);

  const filteredSignals = getFilteredSignals();

  const handleFilterChange = (key, value) => {
    const newFilters = { ...localFilters, [key]: value };
    setLocalFilters(newFilters);
    setFilters(newFilters);
  };

  const getSignalIcon = (action) => {
    return action === 'BUY' ? '↗' : '↘';
  };

  const getSignalColor = (action) => {
    return action === 'BUY' ? 'text-green-400' : 'text-red-400';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 85) return 'text-green-400';
    if (confidence >= 75) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div key="signals" className="bg-gray-900 border border-gray-700 rounded-xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Live Signals</h3>
        <div className="text-xs text-gray-400">
          {filteredSignals.length} signals
        </div>
      </div>

      {/* Filters */}
      <div className="space-y-3 mb-4 p-3 bg-gray-800 rounded-lg">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Symbol</label>
            <select
              value={localFilters.symbol}
              onChange={(e) => handleFilterChange('symbol', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white text-sm"
            >
              <option value="">All</option>
              <option value="BTC">BTC</option>
              <option value="ETH">ETH</option>
              <option value="SOL">SOL</option>
              <option value="ADA">ADA</option>
              <option value="DOT">DOT</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Type</label>
            <select
              value={localFilters.signalType}
              onChange={(e) => handleFilterChange('signalType', e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white text-sm"
            >
              <option value="all">All</option>
              <option value="buy">BUY</option>
              <option value="sell">SELL</option>
            </select>
          </div>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">
            Min Confidence: {localFilters.minConfidence}%
          </label>
          <input
            type="range"
            min="50"
            max="100"
            value={localFilters.minConfidence}
            onChange={(e) => handleFilterChange('minConfidence', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
          />
        </div>
      </div>

      {/* Signals List */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {filteredSignals.length > 0 ? (
          filteredSignals.map((signal, index) => (
            <div
              key={`${signal.symbol}-${signal.timestamp || index}`}
              className="bg-gray-800 border border-gray-700 rounded-lg p-3 hover:bg-gray-750 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <span className={`text-lg ${getSignalColor(signal.action)}`}>
                    {getSignalIcon(signal.action)}
                  </span>
                  <span className="font-medium text-white">{signal.symbol}</span>
                  <span className={`text-sm font-bold ${getSignalColor(signal.action)}`}>
                    {signal.action}
                  </span>
                </div>
                <span className="text-xs text-gray-400">
                  {formatTime(signal.timestamp)}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div>
                    <div className="text-xs text-gray-400">Confidence</div>
                    <div className={`text-sm font-bold ${getConfidenceColor(signal.confidence)}`}>
                      {signal.confidence}%
                    </div>
                  </div>
                  {signal.price && (
                    <div>
                      <div className="text-xs text-gray-400">Price</div>
                      <div className="text-sm text-white">
                        ${signal.price.toFixed(4)}
                      </div>
                    </div>
                  )}
                </div>
                
                {signal.risk && (
                  <div className="text-right">
                    <div className="text-xs text-gray-400">Risk</div>
                    <div className={`text-xs font-medium ${
                      signal.risk === 'low' ? 'text-green-400' :
                      signal.risk === 'medium' ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {signal.risk.toUpperCase()}
                    </div>
                  </div>
                )}
              </div>
              
              {signal.targets && (
                <div className="mt-2 text-xs text-gray-400">
                  <span>Target: ${signal.targets.take_profit?.toFixed(4)}</span>
                  {signal.targets.stop_loss && (
                    <span className="ml-3">SL: ${signal.targets.stop_loss?.toFixed(4)}</span>
                  )}
                </div>
              )}
            </div>
          ))
        ) : (
          <div className="flex items-center justify-center h-32 text-gray-400">
            <div className="text-center">
              <div className="text-2xl mb-2">📊</div>
              <div>No signals match your filters</div>
              <div className="text-xs mt-1">Adjust filters to see more signals</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SignalsWidget;