"""
Logging utilities and decorators for MetaHarmonizer
"""

import functools
import time
from typing import Any, Callable, Dict, Optional
from .custom_logger import CustomLogger, timer_decorator

# Global logger instance for convenience
_global_logger = None

def get_logger(name: str = None):
    """Get a spiced-up logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = CustomLogger()
    return _global_logger.custlogger()

def log_function_calls(include_args: bool = False, include_result: bool = False):
    """Decorator to log function calls with optional arguments and results"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            # Prepare log message
            msg = f"📞 Calling {func.__name__}"
            extra_data = {}
            
            if include_args and (args or kwargs):
                extra_data['args'] = args
                extra_data['kwargs'] = kwargs
            
            logger.debug(msg, extra={'extra_data': extra_data} if extra_data else {})
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    logger.debug(f"✅ {func.__name__} completed", 
                               extra={'extra_data': {'result': str(result)[:200]}})
                else:
                    logger.debug(f"✅ {func.__name__} completed successfully")
                
                return result
            except Exception as e:
                logger.error(f"❌ {func.__name__} failed: {str(e)}")
                raise
        return wrapper
    return decorator

def log_data_processing(operation: str):
    """Decorator specifically for data processing operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            start_time = time.time()
            logger.info(f"🔄 Starting {operation}...")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Try to get some metrics about the result
                result_info = {}
                if hasattr(result, '__len__'):
                    result_info['count'] = len(result)
                elif isinstance(result, dict):
                    result_info['keys'] = list(result.keys())[:5]  # First 5 keys
                
                logger.performance_log(operation, duration, **result_info)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"💥 {operation} failed after {duration:.2f}s: {str(e)}")
                raise
                
        return wrapper
    return decorator

def log_api_call(service: str):
    """Decorator for API calls with retry and error logging"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            start_time = time.time()
            logger.info(f"🌐 Making API call to {service}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.performance_log(f"{service} API call", duration, 
                                     service=service, status='success')
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"🔥 {service} API call failed after {duration:.2f}s: {str(e)}")
                logger.performance_log(f"{service} API call", duration,
                                     service=service, status='error', error=str(e))
                raise
                
        return wrapper
    return decorator

class LogContext:
    """Context manager for logging with automatic cleanup"""
    
    def __init__(self, operation: str, logger=None):
        self.operation = operation
        self.logger = logger or get_logger()
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"🚀 Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.performance_log(self.operation, duration, status='success')
        else:
            self.logger.error(f"💥 {self.operation} failed: {str(exc_val)}")
            self.logger.performance_log(self.operation, duration, 
                                      status='error', error=str(exc_val))
        
        return False  # Don't suppress exceptions

def batch_progress_logger(total_items: int, batch_size: int = 100):
    """Helper for logging progress in batch operations"""
    def log_progress(current: int, logger=None):
        if logger is None:
            logger = get_logger()
            
        if current % batch_size == 0 or current == total_items:
            percentage = (current / total_items) * 100
            progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
            logger.info(f"📊 Progress: [{progress_bar}] {percentage:.1f}% ({current}/{total_items})")
    
    return log_progress
