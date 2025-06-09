"""
Modular Plugin Framework for Trading Platform
TrendSpider-style smart tools and AI-assisted strategy evaluation
"""

import importlib
import os
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class PluginManager:
    """Manages dynamic loading and registration of trading plugins"""
    
    def __init__(self):
        self.plugins = {}
        self.active_plugins = []
        
    def register_plugin(self, name: str, plugin_class):
        """Register a plugin with the system"""
        try:
            self.plugins[name] = plugin_class()
            self.active_plugins.append(name)
            logger.info(f"Plugin '{name}' registered successfully")
        except Exception as e:
            logger.error(f"Failed to register plugin '{name}': {e}")
    
    def load_plugins(self):
        """Dynamically load all plugins from the plugins directory"""
        plugin_dir = os.path.dirname(__file__)
        
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f'plugins.{module_name}')
                    
                    # Look for plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (hasattr(attr, '__bases__') and 
                            hasattr(attr, 'plugin_name') and 
                            callable(attr)):
                            self.register_plugin(attr.plugin_name, attr)
                            
                except Exception as e:
                    logger.error(f"Failed to load plugin from {filename}: {e}")
    
    def get_plugin(self, name: str):
        """Get a specific plugin by name"""
        return self.plugins.get(name)
    
    def get_all_plugins(self) -> Dict:
        """Get all registered plugins"""
        return self.plugins
    
    def execute_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs):
        """Execute a method on a specific plugin"""
        plugin = self.get_plugin(plugin_name)
        if plugin and hasattr(plugin, method_name):
            return getattr(plugin, method_name)(*args, **kwargs)
        return None

# Global plugin manager instance
plugin_manager = PluginManager()