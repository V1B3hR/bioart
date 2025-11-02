"""
"""Resilience patterns for robust applications."""

This module provides patterns and utilities for building resilient systems:
- Retry with exponential backoff and jitter
- Circuit breaker for failing fast
- Concurrency control and rate limiting
- Graceful shutdown management
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
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
