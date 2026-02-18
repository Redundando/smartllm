"""Retry logic with exponential backoff for LLM API calls"""

import asyncio
import random
import logging
from typing import Callable, TypeVar
from functools import wraps

logger = logging.getLogger('aws_llm_wrapper')

T = TypeVar('T')

RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "TooManyRequestsException",
    "ModelTimeoutException",
    "InternalServerException",
}


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable
    
    Retries on:
    - AWS throttling and server errors
    - HTTP 5xx errors
    - Timeout and rate limit errors
    
    Args:
        error: Exception to check
        
    Returns:
        True if error should be retried
    """
    try:
        from botocore.exceptions import ClientError
        
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            status_code = error.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 0)
            
            # Retry on specific error codes or 5xx server errors
            return error_code in RETRYABLE_ERROR_CODES or status_code >= 500
    except ImportError:
        pass
    
    # Generic retry for common HTTP errors
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in ['timeout', 'rate limit', '429', '500', '502', '503', '504'])


def calculate_backoff(attempt: int, base_delay: float, max_delay: float) -> float:
    """Calculate exponential backoff with jitter
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds with added jitter
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, 0.1 * delay)
    return delay + jitter


def retry_on_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """Decorator for retrying async functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # Don't retry if not retryable or last attempt
                    if not is_retryable_error(e) or attempt == max_retries:
                        raise
                    
                    # Calculate backoff delay
                    delay = calculate_backoff(attempt, base_delay, max_delay)
                    
                    # Log retry attempt
                    error_name = type(e).__name__
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} after {error_name}, "
                        f"waiting {delay:.1f}s..."
                    )
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_error
        
        return wrapper
    return decorator
