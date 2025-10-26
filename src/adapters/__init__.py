"""
Adapters module for external system integrations.

This module provides port interfaces and adapter implementations for
safe integration with external systems like synthesis platforms,
sequencing services, and storage backends.
"""

from .ports import (
    SynthesisPort,
    SequencingPort,
    StoragePort,
    JobStatus,
    JobResult,
)
from .mock_adapter import (
    MockAdapter,
    MockSynthesisAdapter,
    MockSequencingAdapter,
    MockStorageAdapter,
)
from .sandbox_adapter import (
    SandboxAdapter,
    SandboxSynthesisAdapter,
    SandboxSequencingAdapter,
    SandboxStorageAdapter,
)

__all__ = [
    # Port interfaces
    "SynthesisPort",
    "SequencingPort",
    "StoragePort",
    "JobStatus",
    "JobResult",
    # Mock adapters
    "MockAdapter",
    "MockSynthesisAdapter",
    "MockSequencingAdapter",
    "MockStorageAdapter",
    # Sandbox adapters
    "SandboxAdapter",
    "SandboxSynthesisAdapter",
    "SandboxSequencingAdapter",
    "SandboxStorageAdapter",
]
