import logging
import inspect
import sys
import json
import functools
import time
from datetime import datetime
from typing import Any, Dict, Optional
import os


class ColoredFormatter(logging.Formatter):
    """Colorful formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
        'BOLD': '\033[1m',       # Bold
        'DIM': '\033[2m',        # Dim
    }
    
    # Fun emoji indicators
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '✨',
        'WARNING': '⚠️',
        'ERROR': '💥',
        'CRITICAL': '🚨',
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        emoji = self.EMOJIS.get(record.levelname, '📝')
        
        # Add some visual flair
        if hasattr(record, 'performance_data'):
            emoji = '⚡'  # Performance logs get lightning bolt
        elif hasattr(record, 'user_action'):
            emoji = '👤'  # User actions get person
        elif hasattr(record, 'system_event'):
            emoji = '🔧'  # System events get wrench
        
        # Format the timestamp with style
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Create a styled log line
        formatted_msg = (
            f"{log_color}{self.COLORS['BOLD']}{emoji} "
            f"[{timestamp}] {record.levelname:<8}{self.COLORS['RESET']} "
            f"{log_color}│{self.COLORS['RESET']} "
            f"{self.COLORS['DIM']}{record.name}{self.COLORS['RESET']} "
            f"{log_color}│{self.COLORS['RESET']} "
            f"{record.getMessage()}"
        )
        
        # Add extra context if available
        if hasattr(record, 'extra_data'):
            formatted_msg += f"\n    {self.COLORS['DIM']}📊 Data: {json.dumps(record.extra_data, indent=2)}{self.COLORS['RESET']}"
        
        return formatted_msg


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for file output"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add performance data if available
        if hasattr(record, 'performance_data'):
            log_entry['performance'] = record.performance_data
            
        # Add extra structured data
        if hasattr(record, 'extra_data'):
            log_entry['data'] = record.extra_data
            
        # Add context tags
        if hasattr(record, 'tags'):
            log_entry['tags'] = record.tags
        
        return json.dumps(log_entry)


class SpicyLoggerAdapter(logging.LoggerAdapter):
    """Enhanced logger adapter with extra spice! 🌶️"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
        self._start_times = {}
    
    def info_with_data(self, msg: str, data: Dict[str, Any] = None, **kwargs):
        """Log info with structured data"""
        extra = {'extra_data': data} if data else {}
        self.info(msg, extra=extra, **kwargs)
    
    def performance_log(self, operation: str, duration: float, **metrics):
        """Log performance metrics with style"""
        perf_data = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            **metrics
        }
        extra = {'performance_data': perf_data}
        self.info(f"⚡ {operation} completed in {duration:.3f}s", extra=extra)
    
    def user_action(self, action: str, **context):
        """Log user actions"""
        extra = {'user_action': action, 'extra_data': context}
        self.info(f"👤 User action: {action}", extra=extra)
    
    def system_event(self, event: str, **context):
        """Log system events"""
        extra = {'system_event': event, 'extra_data': context}
        self.info(f"🔧 System event: {event}", extra=extra)
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self._start_times[operation] = time.time()
        self.debug(f"🏁 Starting {operation}...")
    
    def end_timer(self, operation: str, **metrics):
        """End timing and log performance"""
        if operation in self._start_times:
            duration = time.time() - self._start_times[operation]
            del self._start_times[operation]
            self.performance_log(operation, duration, **metrics)
        else:
            self.warning(f"⏰ No start time found for operation: {operation}")


def timer_decorator(operation_name: str = None):
    """Decorator to automatically time and log function execution"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get logger from self if it's a method
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            elif args and hasattr(args[0], '_logger'):
                logger = args[0]._logger
            
            op_name = operation_name or f"{func.__name__}"
            
            if logger and hasattr(logger, 'start_timer'):
                logger.start_timer(op_name)
                try:
                    result = func(*args, **kwargs)
                    logger.end_timer(op_name, status='success')
                    return result
                except Exception as e:
                    logger.end_timer(op_name, status='error', error=str(e))
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


## Enhanced CustomLogger with more flexibility and spice! 🌶️
class CustomLogger:

    def custlogger(self, loglevel=logging.DEBUG, enable_colors=True, json_logs=True):
        """Create a spicy logger with enhanced features"""
        # Set class name from where logger is called
        stack = inspect.stack()
        the_class = stack[1][0].f_locals.get("self", None)
        logger_name = the_class.__class__.__name__ if the_class else "DefaultLogger"

        # Create or get logger
        logger = logging.getLogger(logger_name)
        
        # Convert string log levels to constants
        if isinstance(loglevel, str):
            loglevel = getattr(logging, loglevel.upper(), logging.INFO)
        
        logger.setLevel(loglevel)

        # Add handlers only if they are not already added
        if not logger.handlers:
            # Enhanced file handler with JSON formatting
            if json_logs:
                fh = logging.FileHandler("metaharmonizer_detailed.log")
                fh.setFormatter(JsonFormatter())
                logger.addHandler(fh)
            
            # Simple file handler for backward compatibility
            simple_fh = logging.FileHandler("out.log")
            simple_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s: %(message)s",
                datefmt="%d/%m/%Y %I:%M:%S %p",
            )
            simple_fh.setFormatter(simple_formatter)
            logger.addHandler(simple_fh)
            
            # Enhanced console handler with colors
            stdout = logging.StreamHandler(stream=sys.stdout)
            if enable_colors and os.getenv('TERM', '').lower() != 'dumb':
                stdout.setFormatter(ColoredFormatter())
            else:
                stdout.setFormatter(simple_formatter)
            logger.addHandler(stdout)

        # Return enhanced adapter
        return SpicyLoggerAdapter(logger)

    def get_performance_logger(self):
        """Get a specialized logger for performance metrics"""
        return self.custlogger(loglevel=logging.INFO, enable_colors=True)

    # Create a base class

    def log(self):
        stack = inspect.stack()
        try:
            print("Whole stack is:")
            print("\n".join([str(x[4]) for x in stack]))
            print("-" * 20)
            print("Caller was %s" % (str(stack[2][4])))
        finally:
            del stack


## An easy logger to include class name
## Does not work with my setup currently
## Might want to figure out the exact handling in logger module
class LoggingHandler:

    def __init__(self, *args, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
