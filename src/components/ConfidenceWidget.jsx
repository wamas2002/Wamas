import React from 'react';
import { useTradingStore } from '../store/tradingStore';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

const ConfidenceWidget = () => {
  const { confidence, signals } = useTradingStore();

  const confidenceData = [
    { name: 'Confidence', value: confidence },
    { name: 'Remaining', value: 100 - confidence }
  ];

  const COLORS = ['#00ff88', '#374151'];

  const getConfidenceLevel = (value) => {
    if (value >= 85) return { level: 'Excellent', color: 'text-green-400' };
    if (value >= 75) return { level: 'Good', color: 'text-yellow-400' };
    if (value >= 65) return { level: 'Fair', color: 'text-orange-400' };
    return { level: 'Low', color: 'text-red-400' };
  };

  const confidenceLevel = getConfidenceLevel(confidence);

  const getRecentSignalStats = () => {
    if (!signals || signals.length === 0) return { avgConfidence: 0, count: 0 };
    
    const recent = signals.slice(0, 10);
    const avgConfidence = recent.reduce((sum, signal) => sum + signal.confidence, 0) / recent.length;
    
    return { avgConfidence, count: recent.length };
  };

  const stats = getRecentSignalStats();

  return (
    <div key="confidence" className="bg-gray-900 border border-gray-700 rounded-xl p-6 h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">AI Confidence</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-xs text-gray-400">8 Models Active</span>
        </div>
      </div>

      <div className="flex items-center justify-center mb-6">
        <div className="relative w-32 h-32">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={confidenceData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                startAngle={90}
                endAngle={450}
                paddingAngle={2}
                dataKey="value"
                animationBegin={0}
                animationDuration={1000}
              >
                {confidenceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">{confidence}%</div>
              <div className={`text-xs ${confidenceLevel.color}`}>
                {confidenceLevel.level}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
          <div>
            <div className="text-sm text-gray-400">Model Accuracy</div>
            <div className="text-lg font-semibold text-white">{confidence}%</div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">Trend</div>
            <div className="text-lg font-semibold text-green-400">↗ +2.4%</div>
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
          <div>
            <div className="text-sm text-gray-400">Recent Signals</div>
            <div className="text-lg font-semibold text-white">{stats.count}</div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">Avg Confidence</div>
            <div className="text-lg font-semibold text-blue-400">
              {stats.avgConfidence.toFixed(1)}%
            </div>
          </div>
        </div>

        <div className="p-3 bg-gray-800 rounded-lg">
          <div className="text-sm text-gray-400 mb-2">Model Status</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-gray-300">Pure Local</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-gray-300">Dynamic</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-gray-300">Enhanced</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-gray-300">Professional</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-gray-300">Futures</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-gray-300">Direct Auto</span>
            </div>
          </div>
        </div>

        <div className="text-center text-xs text-gray-400 mt-4">
          Next model retrain in 2h 34m
        </div>
      </div>
    </div>
  );
};

export default ConfidenceWidget;