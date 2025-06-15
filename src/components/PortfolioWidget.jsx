import React from 'react';
import { useTradingStore } from '../store/tradingStore';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const PortfolioWidget = () => {
  const { portfolioData, confidence } = useTradingStore();

  const summary = portfolioData ? {
    totalValue: portfolioData.balance || 0,
    dayChange: portfolioData.dayChange || 0,
    dayChangePercent: portfolioData.dayChangePercent || 0,
    positions: portfolioData.positions || [],
  } : {
    totalValue: 0,
    dayChange: 0,
    dayChangePercent: 0,
    positions: [],
  };

  const COLORS = ['#00ff88', '#3742fa', '#ffa726', '#ef4444', '#9c27b0'];

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  return (
    <div key="portfolio" className="bg-gray-900 border border-gray-700 rounded-xl p-6 h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Portfolio Overview</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-xs text-gray-400">Live OKX Data</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <div className="text-sm text-gray-400">Total Value</div>
          <div className="text-2xl font-bold text-white">
            {formatCurrency(summary.totalValue)}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-400">24h Change</div>
          <div className={`text-xl font-bold ${summary.dayChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatPercent(summary.dayChangePercent)}
          </div>
          <div className={`text-sm ${summary.dayChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatCurrency(summary.dayChange)}
          </div>
        </div>
      </div>

      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-2">AI Confidence</div>
        <div className="relative">
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${confidence}%` }}
            ></div>
          </div>
          <div className="text-right mt-1">
            <span className="text-lg font-bold text-green-400">{confidence}%</span>
          </div>
        </div>
      </div>

      {summary.positions.length > 0 && (
        <div className="h-40">
          <div className="text-sm text-gray-400 mb-2">Asset Allocation</div>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={summary.positions}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={60}
                paddingAngle={2}
                dataKey="value"
                animationBegin={0}
                animationDuration={800}
              >
                {summary.positions.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value, name) => [formatCurrency(value), name]}
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#fff'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default PortfolioWidget;