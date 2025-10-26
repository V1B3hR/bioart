"""
Sandbox adapter for safe end-to-end testing.

This module provides a sandbox environment that simulates external system
interactions without side effects. It logs all operations for auditing
and provides a safe environment for E2E pipeline testing.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .ports import (
    JobResult,
    JobStatus,
    SequencingPort,
    StoragePort,
    SynthesisPort,
)


class SandboxSynthesisAdapter(SynthesisPort):
    """
    Sandbox implementation of SynthesisPort.
    
    Simulates synthesis with realistic delays and comprehensive logging.
    All operations are traced for audit purposes.
    """
    
    def __init__(self, delay_seconds: float = 0.1):
        """
        Initialize sandbox synthesis adapter.
        
        Args:
            delay_seconds: Simulated processing delay
        """
        self.jobs: Dict[str, JobResult] = {}
        self.trace: List[Dict[str, Any]] = []
        self.delay_seconds = delay_seconds
    
    def submit_synthesis(
        self,
        sequence: str,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JobResult:
        """Submit a synthesis job in sandbox."""
        timestamp = time.time()
        
        # Validate sequence
        if not self._is_valid_sequence(sequence):
            self._log_trace("submit_synthesis", job_id, "INVALID", {
                "error": "Invalid DNA sequence",
                "sequence_length": len(sequence),
            })
            raise ValueError(f"Invalid DNA sequence: contains non-ATCG characters")
        
        # Simulate processing delay
        time.sleep(self.delay_seconds)
        
        # Create job result
        result = JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            output={
                "sequence": sequence,
                "length": len(sequence),
                "gc_content": self._calculate_gc_content(sequence),
                "synthesized": True,
            },
            metadata=metadata or {},
        )
        
        self.jobs[job_id] = result
        
        # Log trace
        self._log_trace("submit_synthesis", job_id, "SUCCESS", {
            "sequence_length": len(sequence),
            "gc_content": self._calculate_gc_content(sequence),
            "delay_seconds": self.delay_seconds,
        })
        
        return result
    
    def get_job_status(self, job_id: str) -> JobResult:
        """Get status of a synthesis job."""
        if job_id not in self.jobs:
            self._log_trace("get_job_status", job_id, "NOT_FOUND", {})
            raise KeyError(f"Job not found: {job_id}")
        
        self._log_trace("get_job_status", job_id, "SUCCESS", {
            "status": self.jobs[job_id].status.value,
        })
        
        return self.jobs[job_id]
    
    def cancel_job(self, job_id: str) -> JobResult:
        """Cancel a synthesis job."""
        if job_id not in self.jobs:
            self._log_trace("cancel_job", job_id, "NOT_FOUND", {})
            raise KeyError(f"Job not found: {job_id}")
        
        result = self.jobs[job_id]
        if result.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.IN_PROGRESS):
            result.status = JobStatus.CANCELLED
            self._log_trace("cancel_job", job_id, "CANCELLED", {})
        else:
            self._log_trace("cancel_job", job_id, "NOT_CANCELLABLE", {
                "status": result.status.value,
            })
        
        return result
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full audit trace of all operations."""
        return self.trace.copy()
    
    def clear_trace(self) -> None:
        """Clear the audit trace."""
        self.trace.clear()
    
    def _log_trace(self, operation: str, job_id: str, status: str, details: Dict[str, Any]) -> None:
        """Log an operation to the audit trace."""
        self.trace.append({
            "timestamp": time.time(),
            "operation": operation,
            "job_id": job_id,
            "status": status,
            "details": details,
        })
    
    def _is_valid_sequence(self, sequence: str) -> bool:
        """Validate DNA sequence."""
        return all(c in "ATCG" for c in sequence.upper())
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content percentage."""
        if not sequence:
            return 0.0
        gc_count = sum(1 for c in sequence.upper() if c in "GC")
        return (gc_count / len(sequence)) * 100.0


class SandboxSequencingAdapter(SequencingPort):
    """
    Sandbox implementation of SequencingPort.
    
    Simulates sequencing with realistic delays and comprehensive logging.
    """
    
    def __init__(self, delay_seconds: float = 0.1):
        """
        Initialize sandbox sequencing adapter.
        
        Args:
            delay_seconds: Simulated processing delay
        """
        self.jobs: Dict[str, JobResult] = {}
        self.samples: Dict[str, str] = {}
        self.trace: List[Dict[str, Any]] = []
        self.delay_seconds = delay_seconds
    
    def register_sample(self, sample_id: str, sequence: str) -> None:
        """Register a sample for sequencing (test helper)."""
        self.samples[sample_id] = sequence
        self._log_trace("register_sample", sample_id, "SUCCESS", {
            "sequence_length": len(sequence),
        })
    
    def submit_sequencing(
        self,
        sample_id: str,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JobResult:
        """Submit a sequencing job in sandbox."""
        if sample_id not in self.samples:
            self._log_trace("submit_sequencing", job_id, "SAMPLE_NOT_FOUND", {
                "sample_id": sample_id,
            })
            raise ValueError(f"Sample not found: {sample_id}")
        
        # Simulate processing delay
        time.sleep(self.delay_seconds)
        
        sequence = self.samples[sample_id]
        result = JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            output={
                "sample_id": sample_id,
                "sequence": sequence,
                "length": len(sequence),
                "quality_score": 0.95,  # Simulated quality
            },
            metadata=metadata or {},
        )
        
        self.jobs[job_id] = result
        
        self._log_trace("submit_sequencing", job_id, "SUCCESS", {
            "sample_id": sample_id,
            "sequence_length": len(sequence),
        })
        
        return result
    
    def get_job_status(self, job_id: str) -> JobResult:
        """Get status of a sequencing job."""
        if job_id not in self.jobs:
            self._log_trace("get_job_status", job_id, "NOT_FOUND", {})
            raise KeyError(f"Job not found: {job_id}")
        
        self._log_trace("get_job_status", job_id, "SUCCESS", {
            "status": self.jobs[job_id].status.value,
        })
        
        return self.jobs[job_id]
    
    def retrieve_sequence(self, job_id: str) -> str:
        """Retrieve sequenced data."""
        if job_id not in self.jobs:
            self._log_trace("retrieve_sequence", job_id, "NOT_FOUND", {})
            raise KeyError(f"Job not found: {job_id}")
        
        result = self.jobs[job_id]
        if result.status != JobStatus.COMPLETED:
            self._log_trace("retrieve_sequence", job_id, "NOT_COMPLETED", {
                "status": result.status.value,
            })
            raise RuntimeError(f"Job not completed: {result.status.value}")
        
        if not result.output or "sequence" not in result.output:
            raise RuntimeError("No sequence data available")
        
        sequence = result.output["sequence"]
        self._log_trace("retrieve_sequence", job_id, "SUCCESS", {
            "sequence_length": len(sequence),
        })
        
        return sequence
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full audit trace of all operations."""
        return self.trace.copy()
    
    def clear_trace(self) -> None:
        """Clear the audit trace."""
        self.trace.clear()
    
    def _log_trace(self, operation: str, ref_id: str, status: str, details: Dict[str, Any]) -> None:
        """Log an operation to the audit trace."""
        self.trace.append({
            "timestamp": time.time(),
            "operation": operation,
            "ref_id": ref_id,
            "status": status,
            "details": details,
        })


class SandboxStorageAdapter(StoragePort):
    """
    Sandbox implementation of StoragePort.
    
    Provides in-memory storage with full audit trail.
    """
    
    def __init__(self):
        """Initialize sandbox storage adapter."""
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.trace: List[Dict[str, Any]] = []
    
    def store(
        self,
        key: str,
        sequence: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a sequence with full tracing."""
        if not key:
            self._log_trace("store", key, "INVALID_KEY", {})
            raise ValueError("Key cannot be empty")
        
        if not sequence:
            self._log_trace("store", key, "INVALID_SEQUENCE", {})
            raise ValueError("Sequence cannot be empty")
        
        self.storage[key] = {
            "sequence": sequence,
            "metadata": metadata or {},
            "stored_at": time.time(),
        }
        
        self._log_trace("store", key, "SUCCESS", {
            "sequence_length": len(sequence),
            "has_metadata": bool(metadata),
        })
        
        return True
    
    def retrieve(self, key: str) -> Dict[str, Any]:
        """Retrieve a stored sequence."""
        if key not in self.storage:
            self._log_trace("retrieve", key, "NOT_FOUND", {})
            raise KeyError(f"Key not found: {key}")
        
        self._log_trace("retrieve", key, "SUCCESS", {
            "sequence_length": len(self.storage[key]["sequence"]),
        })
        
        return {
            "sequence": self.storage[key]["sequence"],
            "metadata": self.storage[key]["metadata"],
        }
    
    def delete(self, key: str) -> bool:
        """Delete a stored sequence."""
        if key not in self.storage:
            self._log_trace("delete", key, "NOT_FOUND", {})
            raise KeyError(f"Key not found: {key}")
        
        del self.storage[key]
        self._log_trace("delete", key, "SUCCESS", {})
        return True
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List stored keys."""
        keys = list(self.storage.keys())
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        
        self._log_trace("list_keys", prefix or "*", "SUCCESS", {
            "count": len(keys),
        })
        
        return keys
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        exists = key in self.storage
        self._log_trace("exists", key, "SUCCESS", {"exists": exists})
        return exists
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full audit trace of all operations."""
        return self.trace.copy()
    
    def clear_trace(self) -> None:
        """Clear the audit trace."""
        self.trace.clear()
    
    def _log_trace(self, operation: str, key: str, status: str, details: Dict[str, Any]) -> None:
        """Log an operation to the audit trace."""
        self.trace.append({
            "timestamp": time.time(),
            "operation": operation,
            "key": key,
            "status": status,
            "details": details,
        })


class SandboxAdapter:
    """
    Unified sandbox adapter providing all port implementations.
    
    Provides a complete sandbox environment for E2E testing with
    full audit trails across all operations.
    """
    
    def __init__(self, delay_seconds: float = 0.1):
        """
        Initialize all sandbox adapters.
        
        Args:
            delay_seconds: Simulated processing delay for async operations
        """
        self.synthesis = SandboxSynthesisAdapter(delay_seconds=delay_seconds)
        self.sequencing = SandboxSequencingAdapter(delay_seconds=delay_seconds)
        self.storage = SandboxStorageAdapter()
    
    def get_full_trace(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get combined audit trace from all adapters.
        
        Returns:
            Dictionary with traces for synthesis, sequencing, and storage
        """
        return {
            "synthesis": self.synthesis.get_trace(),
            "sequencing": self.sequencing.get_trace(),
            "storage": self.storage.get_trace(),
        }
    
    def clear_all_traces(self) -> None:
        """Clear audit traces from all adapters."""
        self.synthesis.clear_trace()
        self.sequencing.clear_trace()
        self.storage.clear_trace()
    
    def export_trace_json(self) -> str:
        """
        Export full audit trace as JSON.
        
        Returns:
            JSON string with complete trace
        """
        return json.dumps(self.get_full_trace(), indent=2)
