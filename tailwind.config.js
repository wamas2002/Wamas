/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/**/*.js",
    "./**/*.py"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        'inter': ['Inter', 'sans-serif'],
      },
      colors: {
        'trading-green': '#10b981',
        'trading-blue': '#3b82f6',
        'trading-red': '#ef4444',
        'trading-orange': '#f59e0b',
        'dark-primary': '#0f172a',
        'dark-secondary': '#1e293b',
        'dark-tertiary': '#334155',
      },
      animation: {
        'pulse-trading': 'pulse 2s infinite',
        'spin-trading': 'spin 1s linear infinite',
      },
      backgroundImage: {
        'gradient-trading': 'linear-gradient(135deg, #10b981, #3b82f6)',
        'gradient-success': 'linear-gradient(135deg, #10b981, #059669)',
      },
      boxShadow: {
        'trading-card': '0 10px 25px rgba(0,0,0,0.2)',
        'trading-hover': '0 20px 40px rgba(16, 185, 129, 0.15)',
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}