"""
Contract tests for adapter implementations.

These tests verify that all adapter implementations conform to the
port interface contracts and behave correctly according to the
specifications.
"""

import pytest
import time

from src.adapters import (
    MockAdapter,
    SandboxAdapter,
    JobStatus,
)


class TestMockAdapterContracts:
    """Test that MockAdapter conforms to all port contracts."""
    
    def test_synthesis_submit_and_status(self):
        """Test synthesis job submission and status checking."""
        adapter = MockAdapter()
        
        # Submit a synthesis job
        result = adapter.synthesis.submit_synthesis(
            sequence="ATCGATCG",
            job_id="job-001",
            metadata={"priority": "high"}
        )
        
        assert result.job_id == "job-001"
        assert result.status == JobStatus.COMPLETED
        assert result.output is not None
        assert result.output["sequence"] == "ATCGATCG"
        
        # Check status
        status = adapter.synthesis.get_job_status("job-001")
        assert status.job_id == "job-001"
        assert status.status == JobStatus.COMPLETED
    
    def test_synthesis_invalid_sequence(self):
        """Test that invalid sequences are rejected."""
        adapter = MockAdapter()
        
        with pytest.raises(ValueError, match="Invalid DNA sequence"):
            adapter.synthesis.submit_synthesis(
                sequence="ATCGXYZ",
                job_id="job-002",
            )
    
    def test_synthesis_job_not_found(self):
        """Test error handling for non-existent jobs."""
        adapter = MockAdapter()
        
        with pytest.raises(KeyError, match="Job not found"):
            adapter.synthesis.get_job_status("nonexistent-job")
    
    def test_synthesis_cancel_job(self):
        """Test job cancellation."""
        adapter = MockAdapter()
        
        # For mock, job completes immediately, so cancel won't change status
        result = adapter.synthesis.submit_synthesis(
            sequence="ATCG",
            job_id="job-003",
        )
        
        cancel_result = adapter.synthesis.cancel_job("job-003")
        assert cancel_result.job_id == "job-003"
        # Status may remain COMPLETED since job finished immediately
    
    def test_sequencing_submit_and_retrieve(self):
        """Test sequencing job submission and retrieval."""
        adapter = MockAdapter()
        
        # Register a sample
        adapter.sequencing.register_sample("sample-001", "ATCGATCG")
        
        # Submit sequencing job
        result = adapter.sequencing.submit_sequencing(
            sample_id="sample-001",
            job_id="seq-001",
            metadata={"quality": "high"}
        )
        
        assert result.job_id == "seq-001"
        assert result.status == JobStatus.COMPLETED
        
        # Retrieve sequence
        sequence = adapter.sequencing.retrieve_sequence("seq-001")
        assert sequence == "ATCGATCG"
    
    def test_sequencing_sample_not_found(self):
        """Test error handling for missing samples."""
        adapter = MockAdapter()
        
        with pytest.raises(ValueError, match="Sample not found"):
            adapter.sequencing.submit_sequencing(
                sample_id="nonexistent",
                job_id="seq-002",
            )
    
    def test_sequencing_job_not_found(self):
        """Test error handling for non-existent sequencing jobs."""
        adapter = MockAdapter()
        
        with pytest.raises(KeyError, match="Job not found"):
            adapter.sequencing.get_job_status("nonexistent-job")
        
        with pytest.raises(KeyError, match="Job not found"):
            adapter.sequencing.retrieve_sequence("nonexistent-job")
    
    def test_storage_store_and_retrieve(self):
        """Test storing and retrieving sequences."""
        adapter = MockAdapter()
        
        # Store a sequence
        success = adapter.storage.store(
            key="seq-001",
            sequence="ATCGATCG",
            metadata={"source": "test"}
        )
        assert success is True
        
        # Retrieve the sequence
        data = adapter.storage.retrieve("seq-001")
        assert data["sequence"] == "ATCGATCG"
        assert data["metadata"]["source"] == "test"
    
    def test_storage_key_not_found(self):
        """Test error handling for missing keys."""
        adapter = MockAdapter()
        
        with pytest.raises(KeyError, match="Key not found"):
            adapter.storage.retrieve("nonexistent")
        
        with pytest.raises(KeyError, match="Key not found"):
            adapter.storage.delete("nonexistent")
    
    def test_storage_delete(self):
        """Test deleting stored sequences."""
        adapter = MockAdapter()
        
        # Store and then delete
        adapter.storage.store("seq-001", "ATCG")
        assert adapter.storage.exists("seq-001")
        
        success = adapter.storage.delete("seq-001")
        assert success is True
        assert not adapter.storage.exists("seq-001")
    
    def test_storage_list_keys(self):
        """Test listing stored keys."""
        adapter = MockAdapter()
        
        # Store multiple sequences
        adapter.storage.store("seq-001", "ATCG")
        adapter.storage.store("seq-002", "GCTA")
        adapter.storage.store("data-001", "AAAA")
        
        # List all keys
        all_keys = adapter.storage.list_keys()
        assert len(all_keys) == 3
        assert "seq-001" in all_keys
        
        # List with prefix
        seq_keys = adapter.storage.list_keys(prefix="seq-")
        assert len(seq_keys) == 2
        assert "seq-001" in seq_keys
        assert "seq-002" in seq_keys
        assert "data-001" not in seq_keys
    
    def test_storage_exists(self):
        """Test checking key existence."""
        adapter = MockAdapter()
        
        assert not adapter.storage.exists("seq-001")
        
        adapter.storage.store("seq-001", "ATCG")
        assert adapter.storage.exists("seq-001")
    
    def test_storage_invalid_inputs(self):
        """Test validation of storage inputs."""
        adapter = MockAdapter()
        
        with pytest.raises(ValueError, match="Key cannot be empty"):
            adapter.storage.store("", "ATCG")
        
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            adapter.storage.store("key", "")


class TestSandboxAdapterContracts:
    """Test that SandboxAdapter conforms to all port contracts."""
    
    def test_synthesis_with_tracing(self):
        """Test synthesis with audit trail."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # Submit job
        result = adapter.synthesis.submit_synthesis(
            sequence="ATCGATCG",
            job_id="job-001",
        )
        
        assert result.status == JobStatus.COMPLETED
        assert result.output["gc_content"] > 0
        
        # Check trace
        trace = adapter.synthesis.get_trace()
        assert len(trace) > 0
        assert trace[0]["operation"] == "submit_synthesis"
        assert trace[0]["job_id"] == "job-001"
        assert trace[0]["status"] == "SUCCESS"
    
    def test_sequencing_with_tracing(self):
        """Test sequencing with audit trail."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # Register sample
        adapter.sequencing.register_sample("sample-001", "ATCGATCG")
        
        # Submit job
        result = adapter.sequencing.submit_sequencing(
            sample_id="sample-001",
            job_id="seq-001",
        )
        
        assert result.status == JobStatus.COMPLETED
        
        # Check trace
        trace = adapter.sequencing.get_trace()
        assert len(trace) >= 2  # register + submit
        assert any(t["operation"] == "register_sample" for t in trace)
        assert any(t["operation"] == "submit_sequencing" for t in trace)
    
    def test_storage_with_tracing(self):
        """Test storage with audit trail."""
        adapter = SandboxAdapter()
        
        # Store sequence
        adapter.storage.store("key-001", "ATCG")
        
        # Retrieve sequence
        data = adapter.storage.retrieve("key-001")
        assert data["sequence"] == "ATCG"
        
        # Check trace
        trace = adapter.storage.get_trace()
        assert len(trace) >= 2  # store + retrieve
        assert trace[0]["operation"] == "store"
        assert trace[1]["operation"] == "retrieve"
    
    def test_full_trace_export(self):
        """Test exporting complete audit trail."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # Perform operations
        adapter.synthesis.submit_synthesis("ATCG", "job-001")
        adapter.sequencing.register_sample("sample-001", "GCTA")
        adapter.storage.store("key-001", "TTTT")
        
        # Get full trace
        full_trace = adapter.get_full_trace()
        assert "synthesis" in full_trace
        assert "sequencing" in full_trace
        assert "storage" in full_trace
        
        assert len(full_trace["synthesis"]) > 0
        assert len(full_trace["sequencing"]) > 0
        assert len(full_trace["storage"]) > 0
        
        # Export as JSON
        json_trace = adapter.export_trace_json()
        assert isinstance(json_trace, str)
        assert "synthesis" in json_trace
    
    def test_clear_traces(self):
        """Test clearing audit traces."""
        adapter = SandboxAdapter()
        
        # Perform operations
        adapter.synthesis.submit_synthesis("ATCG", "job-001")
        adapter.storage.store("key-001", "ATCG")
        
        # Clear traces
        adapter.clear_all_traces()
        
        full_trace = adapter.get_full_trace()
        assert len(full_trace["synthesis"]) == 0
        assert len(full_trace["sequencing"]) == 0
        assert len(full_trace["storage"]) == 0
    
    def test_gc_content_calculation(self):
        """Test GC content calculation in synthesis."""
        adapter = SandboxAdapter()
        
        # 50% GC content (2 G's, 2 C's in 4 bases)
        result = adapter.synthesis.submit_synthesis("GCGC", "job-001")
        assert result.output["gc_content"] == 100.0
        
        # 0% GC content
        result = adapter.synthesis.submit_synthesis("ATAT", "job-002")
        assert result.output["gc_content"] == 0.0
        
        # Mixed content
        result = adapter.synthesis.submit_synthesis("ATCG", "job-003")
        assert result.output["gc_content"] == 50.0


class TestJobResultSerialization:
    """Test JobResult serialization."""
    
    def test_job_result_to_dict(self):
        """Test converting JobResult to dictionary."""
        from src.adapters import JobResult, JobStatus
        
        result = JobResult(
            job_id="job-001",
            status=JobStatus.COMPLETED,
            output={"data": "test"},
            error=None,
            metadata={"key": "value"}
        )
        
        d = result.to_dict()
        assert d["job_id"] == "job-001"
        assert d["status"] == "completed"
        assert d["output"]["data"] == "test"
        assert d["error"] is None
        assert d["metadata"]["key"] == "value"


class TestFailureSimulation:
    """Test failure simulation in mock adapters."""
    
    def test_mock_synthesis_with_failures(self):
        """Test mock adapter with simulated failures."""
        adapter = MockAdapter(fail_rate=1.0)  # 100% failure rate
        
        # All jobs should fail
        result = adapter.synthesis.submit_synthesis("ATCG", "job-001")
        assert result.status == JobStatus.FAILED
        assert result.error is not None
    
    def test_mock_sequencing_with_failures(self):
        """Test mock sequencing with failures."""
        adapter = MockAdapter(fail_rate=1.0)
        
        adapter.sequencing.register_sample("sample-001", "ATCG")
        result = adapter.sequencing.submit_sequencing("sample-001", "seq-001")
        
        assert result.status == JobStatus.FAILED
        
        # Should not be able to retrieve from failed job
        with pytest.raises(RuntimeError, match="Job not completed"):
            adapter.sequencing.retrieve_sequence("seq-001")
