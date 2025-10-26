"""
Mock adapter implementations for testing.

This module provides mock implementations of the port interfaces
for unit and contract testing without external dependencies.
"""

import time
from typing import Any, Dict, List, Optional

from .ports import (
    JobResult,
    JobStatus,
    SequencingPort,
    StoragePort,
    SynthesisPort,
)


class MockSynthesisAdapter(SynthesisPort):
    """
    Mock implementation of SynthesisPort for testing.
    
    Simulates synthesis operations in-memory without external calls.
    Useful for unit tests and contract validation.
    """
    
    def __init__(self, fail_rate: float = 0.0):
        """
        Initialize mock synthesis adapter.
        
        Args:
            fail_rate: Probability (0.0-1.0) of simulating failures
        """
        self.jobs: Dict[str, JobResult] = {}
        self.fail_rate = fail_rate
        self._validate_sequences = True
    
    def submit_synthesis(
        self,
        sequence: str,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JobResult:
        """Submit a mock synthesis job."""
        if self._validate_sequences and not self._is_valid_sequence(sequence):
            raise ValueError(f"Invalid DNA sequence: {sequence}")
        
        # Simulate immediate completion in mock
        status = JobStatus.FAILED if self._should_fail() else JobStatus.COMPLETED
        
        result = JobResult(
            job_id=job_id,
            status=status,
            output={"sequence": sequence, "synthesized": status == JobStatus.COMPLETED},
            error="Simulated failure" if status == JobStatus.FAILED else None,
            metadata=metadata or {},
        )
        self.jobs[job_id] = result
        return result
    
    def get_job_status(self, job_id: str) -> JobResult:
        """Get status of a mock synthesis job."""
        if job_id not in self.jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self.jobs[job_id]
    
    def cancel_job(self, job_id: str) -> JobResult:
        """Cancel a mock synthesis job."""
        if job_id not in self.jobs:
            raise KeyError(f"Job not found: {job_id}")
        
        result = self.jobs[job_id]
        if result.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.IN_PROGRESS):
            result.status = JobStatus.CANCELLED
        return result
    
    def _is_valid_sequence(self, sequence: str) -> bool:
        """Validate DNA sequence contains only ATCG characters."""
        return all(c in "ATCG" for c in sequence.upper())
    
    def _should_fail(self) -> bool:
        """Determine if this operation should fail based on fail_rate."""
        import random
        return random.random() < self.fail_rate


class MockSequencingAdapter(SequencingPort):
    """
    Mock implementation of SequencingPort for testing.
    
    Simulates sequencing operations in-memory without external calls.
    """
    
    def __init__(self, fail_rate: float = 0.0):
        """
        Initialize mock sequencing adapter.
        
        Args:
            fail_rate: Probability (0.0-1.0) of simulating failures
        """
        self.jobs: Dict[str, JobResult] = {}
        self.samples: Dict[str, str] = {}  # sample_id -> sequence
        self.fail_rate = fail_rate
    
    def register_sample(self, sample_id: str, sequence: str) -> None:
        """Register a sample for later sequencing (test helper)."""
        self.samples[sample_id] = sequence
    
    def submit_sequencing(
        self,
        sample_id: str,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JobResult:
        """Submit a mock sequencing job."""
        if sample_id not in self.samples:
            raise ValueError(f"Sample not found: {sample_id}")
        
        # Simulate immediate completion in mock
        status = JobStatus.FAILED if self._should_fail() else JobStatus.COMPLETED
        
        result = JobResult(
            job_id=job_id,
            status=status,
            output={"sample_id": sample_id, "sequence": self.samples[sample_id]} if status == JobStatus.COMPLETED else None,
            error="Simulated failure" if status == JobStatus.FAILED else None,
            metadata=metadata or {},
        )
        self.jobs[job_id] = result
        return result
    
    def get_job_status(self, job_id: str) -> JobResult:
        """Get status of a mock sequencing job."""
        if job_id not in self.jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self.jobs[job_id]
    
    def retrieve_sequence(self, job_id: str) -> str:
        """Retrieve sequenced data from a completed job."""
        if job_id not in self.jobs:
            raise KeyError(f"Job not found: {job_id}")
        
        result = self.jobs[job_id]
        if result.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Job not completed: {result.status.value}")
        
        if not result.output or "sequence" not in result.output:
            raise RuntimeError("No sequence data available")
        
        return result.output["sequence"]
    
    def _should_fail(self) -> bool:
        """Determine if this operation should fail based on fail_rate."""
        import random
        return random.random() < self.fail_rate


class MockStorageAdapter(StoragePort):
    """
    Mock implementation of StoragePort for testing.
    
    Simulates storage operations using in-memory dictionary.
    """
    
    def __init__(self):
        """Initialize mock storage adapter."""
        self.storage: Dict[str, Dict[str, Any]] = {}
    
    def store(
        self,
        key: str,
        sequence: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a sequence in memory."""
        if not key:
            raise ValueError("Key cannot be empty")
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        
        self.storage[key] = {
            "sequence": sequence,
            "metadata": metadata or {},
            "stored_at": time.time(),
        }
        return True
    
    def retrieve(self, key: str) -> Dict[str, Any]:
        """Retrieve a stored sequence."""
        if key not in self.storage:
            raise KeyError(f"Key not found: {key}")
        
        return {
            "sequence": self.storage[key]["sequence"],
            "metadata": self.storage[key]["metadata"],
        }
    
    def delete(self, key: str) -> bool:
        """Delete a stored sequence."""
        if key not in self.storage:
            raise KeyError(f"Key not found: {key}")
        
        del self.storage[key]
        return True
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List stored keys, optionally filtered by prefix."""
        if prefix is None:
            return list(self.storage.keys())
        return [k for k in self.storage.keys() if k.startswith(prefix)]
    
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self.storage


class MockAdapter:
    """
    Unified mock adapter providing all port implementations.
    
    Convenient for testing full workflows with all adapters.
    """
    
    def __init__(self, fail_rate: float = 0.0):
        """
        Initialize all mock adapters.
        
        Args:
            fail_rate: Probability (0.0-1.0) of simulating failures
        """
        self.synthesis = MockSynthesisAdapter(fail_rate=fail_rate)
        self.sequencing = MockSequencingAdapter(fail_rate=fail_rate)
        self.storage = MockStorageAdapter()
