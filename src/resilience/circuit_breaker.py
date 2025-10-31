from typing import Any, Callable, Optional, TypeVar, Type, Tuple, Dict, List
from enum import Enum
import threading
import time

T = TypeVar("T")

class CircuitBreakerOpen(Exception):
class CircuitState(Enum):
    """Enum representing circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerError(Exception):
    """Raised when the circuit breaker is open and cannot be called."""
    pass

class CircuitBreaker:
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        success_threshold: int = 1,
        recovery_timeout: float = 60.0,
        expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout
        self.lock = threading.Lock()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
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
            if self.state == CircuitState.OPEN:
                now = time.time()
                if self.last_failure_time and (now - self.last_failure_time) > self.recovery_timeout:
                    # Move to half-open
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN.")
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
                    self.state = CircuitState.OPEN
            raise
        else:
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        self.success_count = 0
                elif self.state == CircuitState.CLOSED:
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
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout_seconds: float = 60.0,
            "recovery_timeout": self.recovery_timeout,
        }

def circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    success_threshold: int = 1,
    recovery_timeout: float = 60.0,
    expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions,
        )
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper
    return decorator

class CircuitBreakerRegistry:
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        success_threshold: int = 1,
        recovery_timeout: float = 60.0,
        expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout_seconds=timeout_seconds,
                recovery_timeout=recovery_timeout,
                expected_exceptions=expected_exceptions,
            )
        return self._breakers[name]

    def get_all_stats(self) -> List[Dict[str, Any]]:
        return [breaker.get_stats() for breaker in self._breakers.values()]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
