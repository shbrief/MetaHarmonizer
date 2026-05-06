"""
Configuration settings for MetaHarmonizer Enhanced Logging
"""

import os
from typing import Dict, Any

# Default logging configuration
DEFAULT_LOG_CONFIG = {
    # Console output settings
    'enable_colors': True,
    'enable_emojis': True,
    'console_level': 'INFO',
    
    # File output settings
    'json_logs': True,
    'detailed_log_file': 'metaharmonizer_detailed.log',
    'simple_log_file': 'out.log',
    'file_level': 'DEBUG',
    
    # Performance tracking
    'track_performance': True,
    'performance_threshold_ms': 100,  # Log performance if slower than this
    
    # Formatting options
    'timestamp_format': '%H:%M:%S.%f',
    'include_line_numbers': True,
    'include_function_names': True,
}

# Environment-specific overrides
ENV_CONFIGS = {
    'development': {
        'console_level': 'DEBUG',
        'enable_colors': True,
        'track_performance': True,
    },
    'production': {
        'console_level': 'INFO',
        'enable_colors': False,  # Disable colors for production logs
        'json_logs': True,
        'track_performance': False,
    },
    'testing': {
        'console_level': 'WARNING',
        'json_logs': False,
        'detailed_log_file': 'test.log',
    }
}

def get_log_config(environment: str = None) -> Dict[str, Any]:
    """
    Get logging configuration for the specified environment
    
    Args:
        environment: Environment name ('development', 'production', 'testing')
                    If None, uses LOG_ENV environment variable or defaults to 'development'
    
    Returns:
        Dictionary with logging configuration
    """
    if environment is None:
        environment = os.getenv('LOG_ENV', 'development')
    
    config = DEFAULT_LOG_CONFIG.copy()
    
    # Apply environment-specific overrides
    if environment in ENV_CONFIGS:
        config.update(ENV_CONFIGS[environment])
    
    return config

# Log level mappings
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
}

# Emoji mappings for different log types
LOG_EMOJIS = {
    'start': '🚀',
    'complete': '✅',
    'error': '💥',
    'warning': '⚠️',
    'info': '✨',
    'debug': '🔍',
    'performance': '⚡',
    'user': '👤',
    'system': '🔧',
    'api': '🌐',
    'database': '🗄️',
    'file': '📁',
    'network': '🌐',
    'security': '🔒',
    'cache': '💾',
    'process': '⚙️',
}

# Color codes for different components
COLOR_SCHEME = {
    'timestamp': '\033[90m',    # Dark gray
    'level': {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    },
    'logger_name': '\033[94m',   # Blue
    'message': '\033[0m',        # Default
    'performance': '\033[93m',   # Bright yellow
    'data': '\033[90m',          # Dark gray
    'reset': '\033[0m',          # Reset
}
