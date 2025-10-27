from typing import Any, Callable, Optional, TypeVar, Type, Tuple, Dict, List
import threading
import time

T = TypeVar("T")

class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and cannot be called."""
    pass

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.lock = threading.Lock()
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        # store expected_exceptions and ensure we always have a tuple of types
"""Circuit Breaker pattern implementation for resilience."""

import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar


class CircuitState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


F = TypeVar("F", bound=Callable[..., Any])


class CircuitBreaker:
    """Circuit breaker for handling failures gracefully."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exceptions: Tuple of exception types to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions: Tuple[Type[Exception], ...] = (
            expected_exceptions if expected_exceptions is not None else (Exception,)
        )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self.lock:
            if self.state == "open":
                now = time.time()
                if self.last_failure_time and (now - self.last_failure_time) > self.timeout_seconds:
                    # Move to half-open
                    self.state = "half-open"
                else:
                    raise CircuitBreakerOpen("Circuit breaker is open.")
        try:
            result = func(*args, **kwargs)
        except self.expected_exceptions as e:
            with self.lock:
                self.failure_count += 1
                self.success_count = 0
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
            raise
        else:
            with self.lock:
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0
                        self.success_count = 0
                elif self.state == "closed":
                    self.failure_count = 0
                    self.success_count = 0
            return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "timeout_seconds": self.timeout_seconds,
        }

def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout_seconds: float = 60.0,
    expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            expected_exceptions=expected_exceptions,
        )
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper
    return decorator

class CircuitBreakerRegistry:
    def __init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.success_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with current statistics
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
        }


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout_seconds=timeout_seconds,
        recovery_timeout: float = 60.0,
        expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exceptions: Tuple of exception types to catch

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exceptions=expected_exceptions,
            )
        return self._breakers[name]

    def get_all_stats(self) -> List[Dict[str, Any]]:
        return [breaker.get_stats() for breaker in self._breakers.values()]
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all circuit breakers.

        Returns:
            List of dictionaries with statistics
        """
        return [breaker.get_stats() for breaker in self._breakers.values()]


# Global registry instance
_registry = CircuitBreakerRegistry()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """Decorator to apply circuit breaker pattern.

    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        expected_exceptions: Tuple of exception types to catch

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        breaker = _registry.get_or_create(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator
