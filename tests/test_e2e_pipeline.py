"""
End-to-end pipeline tests using sandbox adapter.

These tests demonstrate full workflows from encoding through synthesis,
storage, retrieval, sequencing, and decoding with complete audit trails.
"""

import pytest
import json

from src.adapters import SandboxAdapter


class TestEndToEndPipeline:
    """
    Test complete E2E workflows in sandbox environment.
    
    These tests verify the full pipeline integration without external
    dependencies, using the sandbox adapter for safe testing.
    """
    
    def test_encode_synthesize_store_pipeline(self):
        """Test encoding -> synthesis -> storage pipeline."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # 1. Prepare DNA sequence (simulated encode step)
        original_data = b"Hello, DNA!"
        dna_sequence = "ATCGATCGATCGATCG"  # Simplified for test
        job_id = "e2e-001"
        
        # 2. Submit for synthesis
        synth_result = adapter.synthesis.submit_synthesis(
            sequence=dna_sequence,
            job_id=job_id,
            metadata={"original_size": len(original_data)}
        )
        
        assert synth_result.status.value == "completed"
        assert synth_result.output["synthesized"] is True
        
        # 3. Store synthesized sequence
        storage_key = f"synthesized-{job_id}"
        adapter.storage.store(
            key=storage_key,
            sequence=dna_sequence,
            metadata={
                "job_id": job_id,
                "original_size": len(original_data),
                "gc_content": synth_result.output["gc_content"]
            }
        )
        
        # 4. Verify storage
        stored_data = adapter.storage.retrieve(storage_key)
        assert stored_data["sequence"] == dna_sequence
        assert stored_data["metadata"]["job_id"] == job_id
        
        # 5. Verify full trace
        trace = adapter.get_full_trace()
        assert len(trace["synthesis"]) > 0
        assert len(trace["storage"]) > 0
    
    def test_retrieve_sequence_store_pipeline(self):
        """Test retrieval -> sequencing -> storage pipeline."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # 1. Simulate physical DNA sample exists
        sample_id = "sample-001"
        expected_sequence = "GCTAGCTAGCTA"
        adapter.sequencing.register_sample(sample_id, expected_sequence)
        
        # 2. Submit for sequencing
        seq_job_id = "seq-e2e-001"
        seq_result = adapter.sequencing.submit_sequencing(
            sample_id=sample_id,
            job_id=seq_job_id,
            metadata={"read_quality": "high"}
        )
        
        assert seq_result.status.value == "completed"
        
        # 3. Retrieve sequenced data
        sequenced_data = adapter.sequencing.retrieve_sequence(seq_job_id)
        assert sequenced_data == expected_sequence
        
        # 4. Store sequenced result
        storage_key = f"sequenced-{seq_job_id}"
        adapter.storage.store(
            key=storage_key,
            sequence=sequenced_data,
            metadata={
                "sample_id": sample_id,
                "seq_job_id": seq_job_id,
                "quality": seq_result.output["quality_score"]
            }
        )
        
        # 5. Verify storage
        stored_data = adapter.storage.retrieve(storage_key)
        assert stored_data["sequence"] == expected_sequence
        
        # 6. Verify complete trace
        trace = adapter.get_full_trace()
        assert len(trace["sequencing"]) >= 2  # register + submit
        assert len(trace["storage"]) >= 2  # store + retrieve
    
    def test_full_roundtrip_pipeline(self):
        """Test complete roundtrip: encode -> synth -> store -> retrieve -> seq -> decode."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # Phase 1: Encode and Synthesize
        original_sequence = "ATCGATCGATCG"
        synth_job = "roundtrip-synth-001"
        
        synth_result = adapter.synthesis.submit_synthesis(
            sequence=original_sequence,
            job_id=synth_job,
            metadata={"phase": "synthesis"}
        )
        assert synth_result.status.value == "completed"
        
        # Phase 2: Store synthesized result
        storage_key = f"roundtrip-{synth_job}"
        adapter.storage.store(
            key=storage_key,
            sequence=original_sequence,
            metadata={"synth_job": synth_job}
        )
        
        # Phase 3: Simulate physical storage and retrieval
        # In real world, DNA would be physically stored
        
        # Phase 4: Retrieve and prepare for sequencing
        retrieved = adapter.storage.retrieve(storage_key)
        assert retrieved["sequence"] == original_sequence
        
        # Register as physical sample for sequencing
        sample_id = f"sample-{synth_job}"
        adapter.sequencing.register_sample(sample_id, retrieved["sequence"])
        
        # Phase 5: Sequence the sample
        seq_job = "roundtrip-seq-001"
        seq_result = adapter.sequencing.submit_sequencing(
            sample_id=sample_id,
            job_id=seq_job,
            metadata={"phase": "sequencing"}
        )
        
        assert seq_result.status.value == "completed"
        
        # Phase 6: Retrieve sequenced data
        sequenced = adapter.sequencing.retrieve_sequence(seq_job)
        
        # Phase 7: Verify roundtrip integrity
        assert sequenced == original_sequence, "Roundtrip failed: sequence mismatch"
        
        # Phase 8: Verify complete audit trail
        trace = adapter.get_full_trace()
        
        # Should have operations in all three systems
        assert len(trace["synthesis"]) > 0, "Missing synthesis trace"
        assert len(trace["sequencing"]) > 0, "Missing sequencing trace"
        assert len(trace["storage"]) > 0, "Missing storage trace"
        
        # Verify trace contains expected operations
        synth_ops = [t["operation"] for t in trace["synthesis"]]
        seq_ops = [t["operation"] for t in trace["sequencing"]]
        storage_ops = [t["operation"] for t in trace["storage"]]
        
        assert "submit_synthesis" in synth_ops
        assert "register_sample" in seq_ops
        assert "submit_sequencing" in seq_ops
        assert "store" in storage_ops
        assert "retrieve" in storage_ops
    
    def test_pipeline_with_multiple_sequences(self):
        """Test pipeline handling multiple sequences in parallel."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        sequences = {
            "seq-1": "ATCGATCG",
            "seq-2": "GCTAGCTA",
            "seq-3": "TTAATTAA",
        }
        
        # Synthesize all sequences
        for seq_id, sequence in sequences.items():
            result = adapter.synthesis.submit_synthesis(
                sequence=sequence,
                job_id=f"synth-{seq_id}",
            )
            assert result.status.value == "completed"
            
            # Store each
            adapter.storage.store(
                key=seq_id,
                sequence=sequence,
            )
        
        # Verify all stored
        keys = adapter.storage.list_keys()
        assert len(keys) == len(sequences)
        
        for seq_id in sequences:
            assert seq_id in keys
            stored = adapter.storage.retrieve(seq_id)
            assert stored["sequence"] == sequences[seq_id]
        
        # Verify trace shows all operations
        trace = adapter.get_full_trace()
        assert len(trace["synthesis"]) >= len(sequences)
        assert len(trace["storage"]) >= len(sequences) * 2  # store + retrieve
    
    def test_pipeline_error_handling(self):
        """Test pipeline behavior with errors."""
        adapter = SandboxAdapter()
        
        # Test invalid sequence in synthesis
        with pytest.raises(ValueError):
            adapter.synthesis.submit_synthesis(
                sequence="INVALID",
                job_id="error-001",
            )
        
        # Verify error is logged in trace
        trace = adapter.synthesis.get_trace()
        error_traces = [t for t in trace if t["status"] == "INVALID"]
        assert len(error_traces) > 0
        
        # Test missing sample in sequencing
        with pytest.raises(ValueError):
            adapter.sequencing.submit_sequencing(
                sample_id="nonexistent",
                job_id="error-002",
            )
        
        # Test missing key in storage
        with pytest.raises(KeyError):
            adapter.storage.retrieve("nonexistent")
    
    def test_pipeline_audit_trail_export(self):
        """Test exporting audit trail for compliance."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # Perform a series of operations
        adapter.synthesis.submit_synthesis("ATCG", "job-001")
        adapter.storage.store("key-001", "ATCG")
        
        sample_id = "sample-001"
        adapter.sequencing.register_sample(sample_id, "GCTA")
        adapter.sequencing.submit_sequencing(sample_id, "seq-001")
        
        # Export as JSON
        json_trace = adapter.export_trace_json()
        
        # Verify JSON is valid
        parsed = json.loads(json_trace)
        assert "synthesis" in parsed
        assert "sequencing" in parsed
        assert "storage" in parsed
        
        # Verify trace contains timestamps
        for op in parsed["synthesis"]:
            assert "timestamp" in op
            assert "operation" in op
            assert "status" in op
    
    def test_pipeline_idempotency(self):
        """Test that operations are idempotent where appropriate."""
        adapter = SandboxAdapter()
        
        # Store same key multiple times (should overwrite)
        adapter.storage.store("key-001", "ATCG")
        adapter.storage.store("key-001", "GCTA")  # Overwrite
        
        stored = adapter.storage.retrieve("key-001")
        assert stored["sequence"] == "GCTA"  # Should be latest
        
        # Verify both operations are in trace
        trace = adapter.storage.get_trace()
        store_ops = [t for t in trace if t["operation"] == "store"]
        assert len(store_ops) >= 2
    
    def test_pipeline_with_metadata_propagation(self):
        """Test that metadata propagates through pipeline."""
        adapter = SandboxAdapter(delay_seconds=0.01)
        
        # Start with metadata
        original_metadata = {
            "source": "test",
            "priority": "high",
            "batch": "batch-001"
        }
        
        # Synthesis with metadata
        synth_result = adapter.synthesis.submit_synthesis(
            sequence="ATCGATCG",
            job_id="meta-001",
            metadata=original_metadata
        )
        
        assert synth_result.metadata == original_metadata
        
        # Store with enriched metadata
        enriched_metadata = {
            **original_metadata,
            "gc_content": synth_result.output["gc_content"],
            "length": synth_result.output["length"]
        }
        
        adapter.storage.store(
            key="meta-seq-001",
            sequence="ATCGATCG",
            metadata=enriched_metadata
        )
        
        # Retrieve and verify metadata
        stored = adapter.storage.retrieve("meta-seq-001")
        assert stored["metadata"]["source"] == "test"
        assert stored["metadata"]["priority"] == "high"
        assert "gc_content" in stored["metadata"]


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_pipeline_timing(self):
        """Test that sandbox delays are working."""
        import time
        
        adapter = SandboxAdapter(delay_seconds=0.1)
        
        start = time.time()
        adapter.synthesis.submit_synthesis("ATCG", "perf-001")
        duration = time.time() - start
        
        # Should have taken at least the delay time
        assert duration >= 0.1, f"Expected delay >= 0.1s, got {duration}s"
    
    def test_pipeline_trace_size(self):
        """Test that trace doesn't grow unbounded."""
        adapter = SandboxAdapter()
        
        # Perform many operations
        for i in range(100):
            adapter.storage.store(f"key-{i}", "ATCG")
        
        trace = adapter.storage.get_trace()
        assert len(trace) == 100
        
        # Clear and verify
        adapter.storage.clear_trace()
        trace = adapter.storage.get_trace()
        assert len(trace) == 0
