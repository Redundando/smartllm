"""Colored logging configuration for LLM wrapper"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for different log levels
    
    Adds color coding to log levels and special highlighting for:
    - Cache hits (bright green)
    - Cache misses (bright yellow)
    - API calls (bright blue)
    - Errors (bright red)
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    HIGHLIGHTS = {
        'cache_hit': '\033[92m',   # Bright green
        'cache_miss': '\033[93m',  # Bright yellow
        'api_call': '\033[94m',    # Bright blue
        'error': '\033[91m',       # Bright red
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Add color to log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Add color to specific message types
        msg = str(record.msg)
        if 'cache hit' in msg.lower():
            record.msg = f"{self.HIGHLIGHTS['cache_hit']}✓ {msg}{self.RESET}"
        elif 'cache miss' in msg.lower() or 'no cache' in msg.lower():
            record.msg = f"{self.HIGHLIGHTS['cache_miss']}○ {msg}{self.RESET}"
        elif 'api call' in msg.lower() or 'calling' in msg.lower():
            record.msg = f"{self.HIGHLIGHTS['api_call']}→ {msg}{self.RESET}"
        elif 'error' in msg.lower() or 'failed' in msg.lower():
            record.msg = f"{self.HIGHLIGHTS['error']}✗ {msg}{self.RESET}"
        
        return super().format(record)


def setup_logging(level=logging.INFO):
    """Setup colored logging for the LLM wrapper
    
    Args:
        level: Logging level (default: INFO)
    """
    logger = logging.getLogger('smartllm')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Format: [TIME] LEVEL - message
    formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
