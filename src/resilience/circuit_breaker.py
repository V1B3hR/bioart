"""
Circuit breaker pattern for resilient external calls.

This module implements the circuit breaker pattern to prevent cascading
failures when external systems are unavailable or degraded.
"""

import functools
import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """States of a circuit breaker."""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, circuit_name: str):
        self.circuit_name = circuit_name
        super().__init__(
            f"Circuit breaker '{circuit_name}' is OPEN. "
            f"Service appears to be unavailable."
        )


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    The circuit breaker monitors failures and can "open" the circuit
    to fail fast when a service appears to be down, preventing
    resource exhaustion and cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        expected_exceptions: Optional[tuple[type[Exception], ...]] = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in HALF_OPEN before closing
            timeout_seconds: Time to wait before moving from OPEN to HALF_OPEN
            expected_exceptions: Exceptions that count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        expected_exceptions: Optional[tuple[Type[Exception], ...]] = None,
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._update_state()
            return self._state
    
    def _update_state(self) -> None:
        """Update circuit state based on timeout."""
        if self._state == CircuitState.OPEN:
            if (self._last_failure_time and
                time.time() - self._last_failure_time >= self.timeout_seconds):
                logger.info(
                    f"Circuit breaker '{self.name}' moving from OPEN to HALF_OPEN "
                    f"after {self.timeout_seconds}s timeout"
                )
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function protected by the circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from func
        """
        with self._lock:
            self._update_state()
            
            if self._state == CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker '{self.name}' is OPEN, "
                    f"failing fast"
                )
                raise CircuitBreakerError(self.name)
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._failure_count = 0
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker '{self.name}' success count: "
                    f"{self._success_count}/{self.success_threshold}"
                )
                
                if self._success_count >= self.success_threshold:
                    logger.info(
                        f"Circuit breaker '{self.name}' closing after "
                        f"{self._success_count} successes"
                    )
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}' failure count: "
                f"{self._failure_count}/{self.failure_threshold}"
            )
            
            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit breaker '{self.name}' opening after failure "
                    f"in HALF_OPEN state"
                )
                self._state = CircuitState.OPEN
                self._success_count = 0
            elif self._failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}' opening after "
                    f"{self._failure_count} failures"
                )
                self._state = CircuitState.OPEN
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Manually resetting circuit breaker '{self.name}'")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
    
    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            self._update_state()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "timeout_seconds": self.timeout_seconds,
            }


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout_seconds: float = 60.0,
    expected_exceptions: Optional[tuple[type[Exception], ...]] = None,
) -> Callable:
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        name: Name of circuit breaker (defaults to function name)
        failure_threshold: Failures before opening circuit
        success_threshold: Successes needed to close circuit
        timeout_seconds: Time before trying HALF_OPEN
        expected_exceptions: Exceptions to catch
        
    Returns:
        Decorated function
        
    Example:
        @circuit_breaker(failure_threshold=3, timeout_seconds=30)
        def call_external_api():
            return requests.get("https://api.example.com")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker_name = name or func.__name__
        breaker = CircuitBreaker(
            name=breaker_name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            expected_exceptions=expected_exceptions,
        )
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker to wrapper for inspection
        wrapper._circuit_breaker = breaker  # type: ignore
        
        return wrapper
    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        expected_exceptions: Optional[tuple[type[Exception], ...]] = None,
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening
            success_threshold: Successes to close
            timeout_seconds: Timeout before HALF_OPEN
            expected_exceptions: Exceptions to catch
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout_seconds=timeout_seconds,
                    expected_exceptions=expected_exceptions,
                )
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_all_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return [breaker.get_stats() for breaker in self._breakers.values()]


# Global circuit breaker registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _global_registry
