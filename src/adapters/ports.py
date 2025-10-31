"""
Port interfaces for external system integrations.

This module defines the abstract interfaces (ports) that external systems
must implement. Ports provide a stable contract for synthesis, sequencing,
and storage operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(Enum):
    """Status of a job in an external system."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of a job execution."""
    job_id: str
    status: JobStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata or {},
        }


class SynthesisPort(ABC):
    """
    Abstract interface for DNA synthesis platforms.
    
    Synthesis platforms take DNA sequences and produce physical DNA molecules.
    This port abstracts operations like submitting synthesis jobs, monitoring
    their progress, and retrieving results.
    """
    
    @abstractmethod
    def submit_synthesis(
        self,
        sequence: str,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JobResult:
        """
        Submit a DNA sequence for synthesis.
        
        Args:
            sequence: DNA sequence to synthesize (e.g., "ATCG...")
            job_id: Unique identifier for this job
            metadata: Optional metadata (length, GC content, etc.)
            
        Returns:
            JobResult with initial job status
            
        Raises:
            ValueError: If sequence is invalid
        """
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobResult:
        """
        Check the status of a synthesis job.
        
        Args:
            job_id: Unique identifier of the job
            
        Returns:
            JobResult with current job status
            
        Raises:
            KeyError: If job_id is not found
        """
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> JobResult:
        """
        Cancel a pending or in-progress synthesis job.
        
        Args:
            job_id: Unique identifier of the job
            
        Returns:
            JobResult with updated status
            
        Raises:
            KeyError: If job_id is not found
        """
        pass


class SequencingPort(ABC):
    """
    Abstract interface for DNA sequencing platforms.
    
    Sequencing platforms read physical DNA molecules and produce sequence data.
    This port abstracts operations like submitting sequencing jobs and
    retrieving the resulting sequences.
    """
    
    @abstractmethod
    def submit_sequencing(
        self,
        sample_id: str,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JobResult:
        """
        Submit a DNA sample for sequencing.
        
        Args:
            sample_id: Identifier for the physical sample
            job_id: Unique identifier for this job
            metadata: Optional metadata (quality settings, etc.)
            
        Returns:
            JobResult with initial job status
            
        Raises:
            ValueError: If sample_id is invalid
        """
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobResult:
        """
        Check the status of a sequencing job.
        
        Args:
            job_id: Unique identifier of the job
            
        Returns:
            JobResult with current job status and sequence data if complete
            
        Raises:
            KeyError: If job_id is not found
        """
        pass
    
    @abstractmethod
    def retrieve_sequence(self, job_id: str) -> str:
        """
        Retrieve the sequenced DNA data.
        
        Args:
            job_id: Unique identifier of the completed job
            
        Returns:
            DNA sequence string
            
        Raises:
            KeyError: If job_id is not found
            RuntimeError: If job is not completed
        """
        pass


class StoragePort(ABC):
    """
    Abstract interface for storage backends.
    
    Storage backends persist DNA sequences and associated metadata.
    This port abstracts operations like storing, retrieving, and listing
    stored sequences.
    """
    
    @abstractmethod
    def store(
        self,
        key: str,
        sequence: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a DNA sequence with metadata.
        
        Args:
            key: Unique key for storage
            sequence: DNA sequence to store
            metadata: Optional metadata
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If key or sequence is invalid
        """
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Dict[str, Any]:
        """
        Retrieve a stored DNA sequence and its metadata.
        
        Args:
            key: Unique key for retrieval
            
        Returns:
            Dictionary with 'sequence' and 'metadata' keys
            
        Raises:
            KeyError: If key is not found
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a stored sequence.
        
        Args:
            key: Unique key to delete
            
        Returns:
            True if successful
            
        Raises:
            KeyError: If key is not found
        """
        pass
    
    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        List stored keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of matching keys
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in storage.
        
        Args:
            key: Key to check
            
        Returns:
            True if key exists
        """
        pass
