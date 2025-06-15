import React from 'react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

const DraggableWidget = ({ children, layouts, onLayoutChange, isDraggable = true }) => {
  const defaultLayouts = {
    lg: [
      { i: 'portfolio', x: 0, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'confidence', x: 6, y: 0, w: 6, h: 4, minW: 4, minH: 3 },
      { i: 'chart', x: 0, y: 4, w: 8, h: 8, minW: 6, minH: 6 },
      { i: 'signals', x: 8, y: 4, w: 4, h: 8, minW: 3, minH: 6 },
      { i: 'trades', x: 0, y: 12, w: 6, h: 6, minW: 4, minH: 4 },
      { i: 'strategy', x: 6, y: 12, w: 6, h: 6, minW: 4, minH: 4 },
    ],
    md: [
      { i: 'portfolio', x: 0, y: 0, w: 5, h: 4, minW: 3, minH: 3 },
      { i: 'confidence', x: 5, y: 0, w: 5, h: 4, minW: 3, minH: 3 },
      { i: 'chart', x: 0, y: 4, w: 10, h: 8, minW: 6, minH: 6 },
      { i: 'signals', x: 0, y: 12, w: 5, h: 8, minW: 3, minH: 6 },
      { i: 'trades', x: 5, y: 12, w: 5, h: 8, minW: 3, minH: 6 },
      { i: 'strategy', x: 0, y: 20, w: 10, h: 6, minW: 6, minH: 4 },
    ],
    sm: [
      { i: 'portfolio', x: 0, y: 0, w: 6, h: 4, minW: 6, minH: 3 },
      { i: 'confidence', x: 0, y: 4, w: 6, h: 4, minW: 6, minH: 3 },
      { i: 'chart', x: 0, y: 8, w: 6, h: 8, minW: 6, minH: 6 },
      { i: 'signals', x: 0, y: 16, w: 6, h: 8, minW: 6, minH: 6 },
      { i: 'trades', x: 0, y: 24, w: 6, h: 8, minW: 6, minH: 6 },
      { i: 'strategy', x: 0, y: 32, w: 6, h: 6, minW: 6, minH: 4 },
    ]
  };

  const breakpoints = { lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 };
  const cols = { lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 };

  const handleLayoutChange = (layout, layouts) => {
    // Save to localStorage
    localStorage.setItem('dashboardLayouts', JSON.stringify(layouts));
    if (onLayoutChange) {
      onLayoutChange(layout, layouts);
    }
  };

  const currentLayouts = layouts || 
    JSON.parse(localStorage.getItem('dashboardLayouts')) || 
    defaultLayouts;

  return (
    <ResponsiveGridLayout
      className="layout"
      layouts={currentLayouts}
      breakpoints={breakpoints}
      cols={cols}
      rowHeight={60}
      isDraggable={isDraggable}
      isResizable={isDraggable}
      onLayoutChange={handleLayoutChange}
      margin={[16, 16]}
      containerPadding={[16, 16]}
      useCSSTransforms={true}
      preventCollision={false}
      compactType="vertical"
    >
      {children}
    </ResponsiveGridLayout>
  );
};

export default DraggableWidget;