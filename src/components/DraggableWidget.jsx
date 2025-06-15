import React, { useState, useEffect } from 'react';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const DraggableWidget = ({ children, isDraggable = true }) => {
  const [layouts, setLayouts] = useState(() => {
    const saved = localStorage.getItem('dashboard-layout');
    if (saved) {
      return JSON.parse(saved);
    }
    
    // Default layout configuration
    return [
      { i: 'portfolio', x: 0, y: 0, w: 6, h: 8, minW: 4, minH: 6 },
      { i: 'confidence', x: 6, y: 0, w: 6, h: 8, minW: 4, minH: 6 },
      { i: 'chart', x: 0, y: 8, w: 12, h: 10, minW: 8, minH: 8 },
      { i: 'signals', x: 0, y: 18, w: 6, h: 12, minW: 4, minH: 8 },
      { i: 'trades', x: 6, y: 18, w: 6, h: 12, minW: 4, minH: 8 },
      { i: 'strategy', x: 0, y: 30, w: 12, h: 14, minW: 8, minH: 10 }
    ];
  });

  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    localStorage.setItem('dashboard-layout', JSON.stringify(layouts));
  }, [layouts]);

  const onLayoutChange = (newLayout) => {
    setLayouts(newLayout);
  };

  if (!mounted) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {children}
      </div>
    );
  }

  return (
    <GridLayout
      className="layout"
      layout={layouts}
      cols={12}
      rowHeight={30}
      width={1200}
      isDraggable={isDraggable}
      isResizable={isDraggable}
      onLayoutChange={onLayoutChange}
      margin={[16, 16]}
      containerPadding={[0, 0]}
      useCSSTransforms={true}
      preventCollision={false}
      compactType="vertical"
    >
      {children}
    </GridLayout>
  );
};

export default DraggableWidget;