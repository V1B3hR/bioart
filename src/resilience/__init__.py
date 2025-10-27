"""
Resilience module for reliable operations.

This module provides patterns and utilities for building resilient systems:
- Retry with exponential backoff and jitter
- Circuit breaker for failing fast
- Concurrency control and rate limiting
- Graceful shutdown management
"""

from .retry import (
    RetryConfig,
    RetryError,
    retry_with_backoff,
    retry_async,
    IdempotencyKey,
    get_idempotency_tracker,
)
"""Resilience patterns for robust applications."""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
)

from .concurrency import (
    ConcurrencyLimiter,
    concurrency_limit,
    RateLimiter,
    rate_limit,
    GracefulShutdown,
    get_shutdown_manager,
)

__all__ = [
    # Retry
    "RetryConfig",
    "RetryError",
    "retry_with_backoff",
    "retry_async",
    "IdempotencyKey",
    "get_idempotency_tracker",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "circuit_breaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    # Concurrency
    "ConcurrencyLimiter",
    "concurrency_limit",
    "RateLimiter",
    "rate_limit",
    "GracefulShutdown",
    "get_shutdown_manager",
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    "circuit_breaker",
]
