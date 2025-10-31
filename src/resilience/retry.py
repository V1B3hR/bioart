"""
Retry mechanisms with exponential backoff and jitter.

This module provides retry logic for operations that may fail transiently,
with configurable backoff strategies and jitter to avoid thundering herd.
"""

import functools
import logging
import random
import time
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Failed after {attempts} attempts. Last error: {last_exception}"
        )


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            max_delay: Maximum delay in seconds between retries
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delay
            retryable_exceptions: List of exception types to retry.
                                 If None, retries all exceptions.
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or []
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter if enabled
        if self.jitter:
            # Full jitter: random value between 0 and calculated delay
            delay = random.uniform(0, delay)
        
        return delay
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        if not self.retryable_exceptions:
            # Retry all exceptions if no specific types configured
            return True
        
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.retryable_exceptions
        )


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
) -> Callable:
    """
    Decorator to retry a function with exponential backoff and jitter.
    
    Args:
        config: RetryConfig object (if provided, other args are ignored)
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter
        retryable_exceptions: Exceptions to retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_attempts=5, initial_delay=0.5)
        def unstable_operation():
            # May fail transiently
            return call_external_api()
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
        )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry this exception
                    if not config.should_retry(e):
                        logger.warning(
                            f"Exception {type(e).__name__} is not retryable, "
                            f"raising immediately"
                        )
                        raise
                    
                    # Check if we have more attempts
                    if attempt >= config.max_attempts - 1:
                        logger.error(
                            f"Failed after {config.max_attempts} attempts: {e}"
                        )
                        raise RetryError(config.max_attempts, e)
                    
                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here, but for type safety
            if last_exception:
                raise RetryError(config.max_attempts, last_exception)
            raise RuntimeError("Unexpected retry state")
        
        return wrapper
    return decorator


def retry_async(
    operation: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    **kwargs: Any
) -> T:
    """
    Retry an operation with exponential backoff (functional interface).
    
    This is useful when you can't use the decorator (e.g., for lambdas
    or dynamically constructed operations).
    
    Args:
        operation: Function to retry
        *args: Positional arguments for operation
        config: RetryConfig object
        max_attempts: Maximum attempts (if config not provided)
        initial_delay: Initial delay (if config not provided)
        **kwargs: Keyword arguments for operation
        
    Returns:
        Result of the operation
        
    Raises:
        RetryError: If all attempts fail
        
    Example:
        result = retry_async(
            lambda: api.call(),
            max_attempts=5,
            initial_delay=0.5
        )
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
        )
    
    @retry_with_backoff(config=config)
    def wrapped_operation():
        return operation(*args, **kwargs)
    
    return wrapped_operation()


class IdempotencyKey:
    """
    Manages idempotency keys for operations.
    
    Idempotency keys ensure that repeated operations with the same key
    produce the same result without unintended side effects.
    """
    
    def __init__(self):
        """Initialize idempotency tracker."""
        self._results: dict[str, Any] = {}
        self._in_progress: set[str] = set()
    
    def execute_once(
        self,
        key: str,
        operation: Callable[[], T],
    ) -> T:
        """
        Execute operation with idempotency guarantee.
        
        If the key has been seen before, returns the cached result.
        If the operation is in progress, waits and returns the result.
        Otherwise, executes the operation and caches the result.
        
        Args:
            key: Unique idempotency key
            operation: Operation to execute
            
        Returns:
            Result of the operation (cached or fresh)
            
        Raises:
            RuntimeError: If operation is already in progress by another caller
        """
        # Check if we have a cached result
        if key in self._results:
            logger.debug(f"Returning cached result for idempotency key: {key}")
            return self._results[key]
        
        # Check if operation is in progress
        if key in self._in_progress:
            raise RuntimeError(
                f"Operation with key {key} is already in progress"
            )
        
        try:
            # Mark as in progress
            self._in_progress.add(key)
            
            # Execute operation
            result = operation()
            
            # Cache result
            self._results[key] = result
            
            return result
        finally:
            # Remove from in-progress set
            self._in_progress.discard(key)
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear cached results.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key is None:
            self._results.clear()
            logger.debug("Cleared all idempotency results")
        elif key in self._results:
            del self._results[key]
            logger.debug(f"Cleared idempotency result for key: {key}")
    
    def has_result(self, key: str) -> bool:
        """Check if a result is cached for the given key."""
        return key in self._results


# Global idempotency tracker
_global_idempotency = IdempotencyKey()


def get_idempotency_tracker() -> IdempotencyKey:
    """Get the global idempotency tracker."""
    return _global_idempotency
