"""
Concurrency control and rate limiting.

This module provides mechanisms to control concurrent operations and
enforce rate limits to prevent resource exhaustion.
"""

import functools
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConcurrencyLimiter:
    """
    Limits the number of concurrent operations.
    
    Uses a semaphore to ensure that no more than max_concurrent
    operations are executing simultaneously.
    """
    
    def __init__(self, max_concurrent: int):
        """
        Initialize concurrency limiter.
        
        Args:
            max_concurrent: Maximum number of concurrent operations
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to execute an operation.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._active_count += 1
        return acquired
    
    def release(self) -> None:
        """Release permission after operation completes."""
        with self._lock:
            self._active_count -= 1
        self._semaphore.release()
    
    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError("Failed to acquire concurrency slot")
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.release()
    
    def get_active_count(self) -> int:
        """Get number of currently active operations."""
        with self._lock:
            return self._active_count
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with concurrency control.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of func
        """
        with self:
            return func(*args, **kwargs)


def concurrency_limit(max_concurrent: int) -> Callable:
    """
    Decorator to limit concurrent executions of a function.
    
    Args:
        max_concurrent: Maximum concurrent executions
        
    Returns:
        Decorated function
        
    Example:
        @concurrency_limit(max_concurrent=5)
        def process_item(item):
            # Only 5 concurrent executions allowed
            return expensive_operation(item)
    """
    limiter = ConcurrencyLimiter(max_concurrent)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return limiter.call(func, *args, **kwargs)
        
        # Attach limiter for inspection
        wrapper._concurrency_limiter = limiter  # type: ignore
        
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    
    Allows bursts up to capacity, then enforces a steady rate.
    """
    
    def __init__(
        self,
        rate: float,
        capacity: Optional[int] = None,
        time_unit: float = 1.0,
    ):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of operations allowed per time_unit
            capacity: Bucket capacity (defaults to rate)
            time_unit: Time unit in seconds (default 1.0 = per second)
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        
        self.rate = rate
        self.capacity = capacity or int(rate)
        self.time_unit = time_unit
        
        self._tokens = float(self.capacity)
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        
        # Calculate tokens to add based on rate and elapsed time
        tokens_to_add = (elapsed / self.time_unit) * self.rate
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_update = now
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            True if acquired, False if timeout
        """
        if tokens < 1:
            raise ValueError("tokens must be at least 1")
        if tokens > self.capacity:
            raise ValueError(f"tokens ({tokens}) exceeds capacity ({self.capacity})")
        
        start_time = time.time()
        
        while True:
            with self._lock:
                self._refill_tokens()
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
            
            # Wait a bit before retrying
            time.sleep(0.01)
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False otherwise
        """
        return self.acquire(tokens=tokens, timeout=0)
    
    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill_tokens()
            return self._tokens


def rate_limit(
    rate: float,
    capacity: Optional[int] = None,
    time_unit: float = 1.0,
) -> Callable:
    """
    Decorator to rate limit function calls.
    
    Args:
        rate: Operations per time_unit
        capacity: Burst capacity
        time_unit: Time unit in seconds
        
    Returns:
        Decorated function
        
    Example:
        @rate_limit(rate=10, time_unit=1.0)  # 10 calls per second
        def api_call():
            return requests.get("https://api.example.com")
    """
    limiter = RateLimiter(rate=rate, capacity=capacity, time_unit=time_unit)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not limiter.acquire():
                raise RuntimeError("Rate limit exceeded")
            return func(*args, **kwargs)
        
        # Attach limiter for inspection
        wrapper._rate_limiter = limiter  # type: ignore
        
        return wrapper
    return decorator


class GracefulShutdown:
    """
    Manages graceful shutdown of operations.
    
    Allows operations to complete before shutdown, with a timeout
    for forceful termination.
    """
    
    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize graceful shutdown manager.
        
        Args:
            timeout_seconds: Maximum time to wait for operations
        """
        self.timeout_seconds = timeout_seconds
        self._shutdown_requested = False
        self._active_operations = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        with self._lock:
            return self._shutdown_requested
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        with self._lock:
            logger.info("Graceful shutdown requested")
            self._shutdown_requested = True
            self._condition.notify_all()
    
    def register_operation(self) -> bool:
        """
        Register start of an operation.
        
        Returns:
            True if operation can proceed, False if shutdown in progress
        """
        with self._lock:
            if self._shutdown_requested:
                return False
            self._active_operations += 1
            return True
    
    def unregister_operation(self) -> None:
        """Unregister completion of an operation."""
        with self._lock:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._condition.notify_all()
    
    def wait_for_completion(self) -> bool:
        """
        Wait for all operations to complete.
        
        Returns:
            True if all operations completed, False if timeout
        """
        start_time = time.time()
        
        with self._lock:
            while self._active_operations > 0:
                remaining = self.timeout_seconds - (time.time() - start_time)
                if remaining <= 0:
                    logger.warning(
                        f"Shutdown timeout: {self._active_operations} "
                        f"operations still active"
                    )
                    return False
                
                self._condition.wait(timeout=remaining)
        
        logger.info("All operations completed successfully")
        return True
    
    def __enter__(self):
        """Context manager entry."""
        if not self.register_operation():
            raise RuntimeError("Shutdown in progress")
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.unregister_operation()


# Global graceful shutdown manager
_global_shutdown = GracefulShutdown()


def get_shutdown_manager() -> GracefulShutdown:
    """Get the global shutdown manager."""
    return _global_shutdown
