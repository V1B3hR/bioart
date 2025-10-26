"""
Tests for resilience module (retry, circuit breaker, concurrency control).
"""

import pytest
import time
import threading

from src.resilience import (
    RetryConfig,
    RetryError,
    retry_with_backoff,
    retry_async,
    IdempotencyKey,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
    ConcurrencyLimiter,
    concurrency_limit,
    RateLimiter,
    rate_limit,
    GracefulShutdown,
)


class TestRetryLogic:
    """Test retry mechanisms."""
    
    def test_retry_success_on_first_attempt(self):
        """Test successful operation on first try."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = always_succeeds()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test retry after transient failures."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"
        
        result = fails_twice()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def always_fails():
            raise ValueError("Permanent error")
        
        with pytest.raises(RetryError) as exc_info:
            always_fails()
        
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ValueError)
    
    def test_retry_with_specific_exceptions(self):
        """Test retry only for specific exceptions."""
        @retry_with_backoff(
            max_attempts=3,
            retryable_exceptions=[ConnectionError],
            initial_delay=0.01
        )
        def fails_with_value_error():
            raise ValueError("Not retryable")
        
        # Should raise immediately without retry
        with pytest.raises(ValueError):
            fails_with_value_error()
    
    def test_retry_backoff_calculation(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False  # Disable jitter for predictable testing
        )
        
        assert config.calculate_delay(0) == 1.0  # 1 * 2^0
        assert config.calculate_delay(1) == 2.0  # 1 * 2^1
        assert config.calculate_delay(2) == 4.0  # 1 * 2^2
        assert config.calculate_delay(3) == 8.0  # 1 * 2^3
        assert config.calculate_delay(4) == 10.0  # capped at max_delay
    
    def test_retry_with_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(
            initial_delay=1.0,
            jitter=True
        )
        
        # With jitter, delay should be random between 0 and calculated delay
        delays = [config.calculate_delay(0) for _ in range(10)]
        
        # Should have variation
        assert len(set(delays)) > 1
        # All should be within bounds
        assert all(0 <= d <= 1.0 for d in delays)
    
    def test_retry_async_function(self):
        """Test retry_async with functional interface."""
        call_count = 0
        
        def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Transient")
            return "success"
        
        result = retry_async(fails_once, max_attempts=3, initial_delay=0.01)
        assert result == "success"
        assert call_count == 2


class TestIdempotency:
    """Test idempotency mechanisms."""
    
    def test_idempotent_operation(self):
        """Test idempotent execution."""
        tracker = IdempotencyKey()
        call_count = 0
        
        def operation():
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"
        
        # First call executes
        result1 = tracker.execute_once("key1", operation)
        assert result1 == "result-1"
        assert call_count == 1
        
        # Second call with same key returns cached result
        result2 = tracker.execute_once("key1", operation)
        assert result2 == "result-1"  # Same as first
        assert call_count == 1  # Not executed again
    
    def test_idempotency_different_keys(self):
        """Test different keys execute independently."""
        tracker = IdempotencyKey()
        
        result1 = tracker.execute_once("key1", lambda: "value1")
        result2 = tracker.execute_once("key2", lambda: "value2")
        
        assert result1 == "value1"
        assert result2 == "value2"
    
    def test_idempotency_clear(self):
        """Test clearing idempotency cache."""
        tracker = IdempotencyKey()
        call_count = 0
        
        def operation():
            nonlocal call_count
            call_count += 1
            return call_count
        
        result1 = tracker.execute_once("key1", operation)
        assert result1 == 1
        
        # Clear and execute again
        tracker.clear("key1")
        result2 = tracker.execute_once("key1", operation)
        assert result2 == 2
    
    def test_idempotency_has_result(self):
        """Test checking if result exists."""
        tracker = IdempotencyKey()
        
        assert not tracker.has_result("key1")
        
        tracker.execute_once("key1", lambda: "value")
        assert tracker.has_result("key1")


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal state."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        assert breaker.state == CircuitState.CLOSED
        
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        # Cause failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: 1 / 0 if False else (_ for _ in ()).throw(ValueError("Error")))
        
        assert breaker.state == CircuitState.OPEN
    
    def test_circuit_breaker_fails_fast_when_open(self):
        """Test circuit breaker fails fast when open."""
        breaker = CircuitBreaker("test", failure_threshold=2)
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("Error")))
        
        # Should fail immediately
        with pytest.raises(CircuitBreakerError):
            breaker.call(lambda: "should not execute")
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through HALF_OPEN."""
        breaker = CircuitBreaker(
            "test",
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.1
        )
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("Error")))
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should be HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Successful calls should close it
        breaker.call(lambda: "success")
        breaker.call(lambda: "success")
        
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        call_count = 0
        
        @circuit_breaker(failure_threshold=2, timeout_seconds=0.1)
        def unstable_function(should_fail):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Error")
            return "success"
        
        # Normal operation
        result = unstable_function(False)
        assert result == "success"
        
        # Cause failures to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                unstable_function(True)
        
        # Should fail fast
        with pytest.raises(CircuitBreakerError):
            unstable_function(False)
    
    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        breaker = CircuitBreaker("test", failure_threshold=2)
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("Error")))
        
        assert breaker.state == CircuitState.OPEN
        
        # Reset manually
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        
        # Should work normally now
        result = breaker.call(lambda: "success")
        assert result == "success"


class TestConcurrency:
    """Test concurrency control."""
    
    def test_concurrency_limiter(self):
        """Test basic concurrency limiting."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        
        assert limiter.get_active_count() == 0
        
        # Acquire slots
        assert limiter.acquire()
        assert limiter.get_active_count() == 1
        
        assert limiter.acquire()
        assert limiter.get_active_count() == 2
        
        # Release slots
        limiter.release()
        assert limiter.get_active_count() == 1
        
        limiter.release()
        assert limiter.get_active_count() == 0
    
    def test_concurrency_limiter_context_manager(self):
        """Test concurrency limiter as context manager."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        
        with limiter:
            assert limiter.get_active_count() == 1
            with limiter:
                assert limiter.get_active_count() == 2
        
        assert limiter.get_active_count() == 0
    
    def test_concurrency_limit_decorator(self):
        """Test concurrency limit decorator."""
        results = []
        
        @concurrency_limit(max_concurrent=2)
        def slow_operation(value):
            time.sleep(0.1)
            results.append(value)
            return value
        
        # Run operations in threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=slow_operation, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 5
        assert sorted(results) == [0, 1, 2, 3, 4]


class TestRateLimiting:
    """Test rate limiting."""
    
    def test_rate_limiter_burst(self):
        """Test rate limiter allows bursts."""
        limiter = RateLimiter(rate=10, capacity=10)
        
        # Should allow burst of capacity
        for _ in range(10):
            assert limiter.try_acquire()
        
        # Should be depleted
        assert not limiter.try_acquire()
    
    def test_rate_limiter_refill(self):
        """Test rate limiter refills over time."""
        limiter = RateLimiter(rate=10, time_unit=0.1)  # 10 per 0.1s = 100/s
        
        # Deplete tokens
        for _ in range(10):
            limiter.try_acquire()
        
        # Wait for refill
        time.sleep(0.2)
        
        # Should have refilled
        assert limiter.get_available_tokens() >= 10
    
    def test_rate_limit_decorator(self):
        """Test rate limit decorator."""
        @rate_limit(rate=5, time_unit=1.0)
        def limited_function():
            return "success"
        
        # Should allow rate number of calls
        for _ in range(5):
            result = limited_function()
            assert result == "success"


class TestGracefulShutdown:
    """Test graceful shutdown."""
    
    def test_shutdown_with_no_operations(self):
        """Test shutdown with no active operations."""
        shutdown = GracefulShutdown(timeout_seconds=1.0)
        
        shutdown.request_shutdown()
        assert shutdown.is_shutdown_requested()
        
        completed = shutdown.wait_for_completion()
        assert completed is True
    
    def test_shutdown_waits_for_operations(self):
        """Test shutdown waits for operations to complete."""
        shutdown = GracefulShutdown(timeout_seconds=2.0)
        completed = []
        
        def slow_operation():
            if shutdown.register_operation():
                try:
                    time.sleep(0.2)
                    completed.append(True)
                finally:
                    shutdown.unregister_operation()
        
        # Start operation
        thread = threading.Thread(target=slow_operation)
        thread.start()
        
        time.sleep(0.1)  # Let operation start
        
        # Request shutdown
        shutdown.request_shutdown()
        
        # Wait for completion
        success = shutdown.wait_for_completion()
        thread.join()
        
        assert success is True
        assert len(completed) == 1
    
    def test_shutdown_blocks_new_operations(self):
        """Test shutdown prevents new operations."""
        shutdown = GracefulShutdown()
        
        shutdown.request_shutdown()
        
        # Should not allow new operations
        assert not shutdown.register_operation()
    
    def test_shutdown_context_manager(self):
        """Test shutdown with context manager."""
        shutdown = GracefulShutdown()
        
        with shutdown:
            # Operation is active
            pass
        
        # Operation completed
        assert shutdown.wait_for_completion()
