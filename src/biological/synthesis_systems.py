#!/usr/bin/env python3
"""
DNA Synthesis Systems Integration
Provides interfaces for biological DNA synthesis platforms and validation
"""

import time
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib

class SynthesisStatus(Enum):
    """DNA synthesis job status"""
    QUEUED = "queued"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"

@dataclass
class SynthesisJob:
    """DNA synthesis job specification"""
    job_id: str
    dna_sequence: str
    length: int
    priority: int = 5
    quality_threshold: float = 0.95
    created_time: float = 0.0
    status: SynthesisStatus = SynthesisStatus.QUEUED
    estimated_completion: float = 0.0
    actual_completion: Optional[float] = None
    error_rate: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_time == 0.0:
            self.created_time = time.time()
        if self.metadata is None:
            self.metadata = {}

class DNASynthesisManager:
    """
    DNA Synthesis Systems Manager
    Interfaces with biological synthesis platforms for real DNA production
    """
    
    def __init__(self):
        """Initialize synthesis manager"""
        self.synthesis_queue = []
        self.completed_jobs = []
        self.failed_jobs = []
        self.synthesis_platforms = {
            'platform_a': {'capacity': 1000, 'error_rate': 0.001, 'speed': 100},
            'platform_b': {'capacity': 2000, 'error_rate': 0.0005, 'speed': 150},
            'platform_c': {'capacity': 500, 'error_rate': 0.002, 'speed': 200}  # Fast but less accurate
        }
        self.job_counter = 0
        
        # Biological constraints
        self.max_sequence_length = 10000  # Maximum synthesizable length
        self.forbidden_sequences = [
            'AAAAAAAAAA',  # Poly-A sequences can be problematic
            'GGGGGGGGGG',  # Poly-G sequences
            'CCCCCCCCCC',  # Poly-C sequences
        ]
        
        # Quality control parameters
        self.min_gc_content = 0.3
        self.max_gc_content = 0.7
        self.min_complexity = 0.4
    
    def submit_synthesis_job(self, dna_sequence: str, priority: int = 5, 
                           platform: str = 'auto') -> str:
        """
        Submit DNA sequence for biological synthesis
        
        Args:
            dna_sequence: DNA sequence to synthesize
            priority: Job priority (1-10, higher is more urgent)
            platform: Target synthesis platform
            
        Returns:
            Job ID for tracking
        """
        # Validate sequence
        validation_result = self._validate_sequence(dna_sequence)
        if not validation_result['valid']:
            raise ValueError(f"Invalid sequence: {validation_result['errors']}")
        
        # Generate job ID
        self.job_counter += 1
        job_id = f"SYN_{self.job_counter:06d}_{int(time.time())}"
        
        # Select optimal platform
        if platform == 'auto':
            platform = self._select_optimal_platform(len(dna_sequence))
        
        # Create synthesis job
        job = SynthesisJob(
            job_id=job_id,
            dna_sequence=dna_sequence,
            length=len(dna_sequence),
            priority=priority,
            metadata={
                'platform': platform,
                'gc_content': validation_result['gc_content'],
                'complexity': validation_result['complexity']
            }
        )
        
        # Calculate estimated completion
        platform_info = self.synthesis_platforms[platform]
        job.estimated_completion = time.time() + (len(dna_sequence) / platform_info['speed']) * 60
        
        # Add to queue (sorted by priority)
        self.synthesis_queue.append(job)
        self.synthesis_queue.sort(key=lambda x: (-x.priority, x.created_time))
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get synthesis job status and details"""
        # Check all job lists
        all_jobs = self.synthesis_queue + self.completed_jobs + self.failed_jobs
        
        for job in all_jobs:
            if job.job_id == job_id:
                return {
                    'job_id': job.job_id,
                    'status': job.status.value,
                    'length': job.length,
                    'priority': job.priority,
                    'created_time': job.created_time,
                    'estimated_completion': job.estimated_completion,
                    'actual_completion': job.actual_completion,
                    'error_rate': job.error_rate,
                    'metadata': job.metadata
                }
        
        return None
    
    def process_synthesis_queue(self) -> List[Dict[str, Any]]:
        """
        Process synthesis queue (simulate biological synthesis)
        Returns list of completed jobs
        """
        completed_jobs = []
        current_time = time.time()
        
        for job in self.synthesis_queue[:]:
            if job.status == SynthesisStatus.QUEUED:
                # Start synthesis
                job.status = SynthesisStatus.SYNTHESIZING
                
            elif job.status == SynthesisStatus.SYNTHESIZING:
                # Check if synthesis should be complete
                if current_time >= job.estimated_completion:
                    # Simulate synthesis outcome
                    success = self._simulate_synthesis(job)
                    
                    if success:
                        job.status = SynthesisStatus.COMPLETED
                        job.actual_completion = current_time
                        self.completed_jobs.append(job)
                        completed_jobs.append({
                            'job_id': job.job_id,
                            'status': 'completed',
                            'dna_sequence': job.dna_sequence,
                            'error_rate': job.error_rate
                        })
                    else:
                        job.status = SynthesisStatus.FAILED
                        job.actual_completion = current_time
                        self.failed_jobs.append(job)
                    
                    self.synthesis_queue.remove(job)
        
        return completed_jobs
    
    def _validate_sequence(self, dna_sequence: str) -> Dict[str, Any]:
        """Validate DNA sequence for biological synthesis"""
        sequence = dna_sequence.upper()
        errors = []
        
        # Check length
        if len(sequence) > self.max_sequence_length:
            errors.append(f"Sequence too long: {len(sequence)} > {self.max_sequence_length}")
        
        # Check for forbidden sequences
        for forbidden in self.forbidden_sequences:
            if forbidden in sequence:
                errors.append(f"Contains forbidden sequence: {forbidden}")
        
        # Check nucleotide composition
        valid_nucleotides = set('AUCG')
        invalid_chars = set(sequence) - valid_nucleotides
        if invalid_chars:
            errors.append(f"Invalid nucleotides: {invalid_chars}")
        
        # Calculate GC content
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0
        
        if gc_content < self.min_gc_content or gc_content > self.max_gc_content:
            errors.append(f"GC content out of range: {gc_content:.2f} not in [{self.min_gc_content}, {self.max_gc_content}]")
        
        # Calculate sequence complexity (Shannon entropy)
        complexity = self._calculate_complexity(sequence)
        if complexity < self.min_complexity:
            errors.append(f"Sequence complexity too low: {complexity:.2f} < {self.min_complexity}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'gc_content': gc_content,
            'complexity': complexity
        }
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity using Shannon entropy"""
        if not sequence:
            return 0.0
        
        # Count nucleotide frequencies
        nucleotide_counts = {}
        for nucleotide in sequence:
            nucleotide_counts[nucleotide] = nucleotide_counts.get(nucleotide, 0) + 1
        
        # Calculate Shannon entropy
        length = len(sequence)
        entropy = 0.0
        for count in nucleotide_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)  # Approximation of log2
        
        # Normalize to 0-1 range (max entropy for 4 nucleotides is 2)
        return entropy / 2.0
    
    def _select_optimal_platform(self, sequence_length: int) -> str:
        """Select optimal synthesis platform based on sequence characteristics"""
        # Simple selection based on capacity and length
        suitable_platforms = []
        
        for platform, specs in self.synthesis_platforms.items():
            if sequence_length <= specs['capacity']:
                suitable_platforms.append((platform, specs))
        
        if not suitable_platforms:
            # Use platform with highest capacity
            return max(self.synthesis_platforms.keys(), 
                      key=lambda p: self.synthesis_platforms[p]['capacity'])
        
        # Select platform with best balance of speed and accuracy
        best_platform = min(suitable_platforms, 
                           key=lambda x: x[1]['error_rate'] - x[1]['speed'] / 1000)
        
        return best_platform[0]
    
    def _simulate_synthesis(self, job: SynthesisJob) -> bool:
        """Simulate biological synthesis process"""
        platform = job.metadata['platform']
        platform_specs = self.synthesis_platforms[platform]
        
        # Calculate success probability based on platform specs and sequence characteristics
        base_success_rate = 1.0 - platform_specs['error_rate']
        
        # Adjust for sequence complexity
        complexity_bonus = job.metadata['complexity'] * 0.1
        gc_penalty = abs(job.metadata['gc_content'] - 0.5) * 0.2
        length_penalty = (job.length / self.max_sequence_length) * 0.1
        
        success_rate = base_success_rate + complexity_bonus - gc_penalty - length_penalty
        success_rate = max(0.1, min(0.99, success_rate))  # Clamp to reasonable range
        
        # Simulate synthesis
        success = random.random() < success_rate
        
        # Set error rate
        if success:
            job.error_rate = platform_specs['error_rate'] * (1 + random.uniform(-0.5, 0.5))
        else:
            job.error_rate = random.uniform(0.1, 0.5)  # High error rate on failure
        
        return success
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get synthesis system statistics"""
        total_jobs = len(self.synthesis_queue) + len(self.completed_jobs) + len(self.failed_jobs)
        
        return {
            'total_jobs': total_jobs,
            'queued_jobs': len(self.synthesis_queue),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'success_rate': len(self.completed_jobs) / max(1, total_jobs),
            'platforms': list(self.synthesis_platforms.keys()),
            'queue_status': [
                {
                    'job_id': job.job_id,
                    'status': job.status.value,
                    'priority': job.priority,
                    'estimated_completion': job.estimated_completion
                }
                for job in self.synthesis_queue[:5]  # Top 5 jobs
            ]
        }
    
    def optimize_synthesis_parameters(self, dna_sequence: str) -> Dict[str, Any]:
        """Optimize synthesis parameters for given sequence"""
        validation = self._validate_sequence(dna_sequence)
        
        recommendations = {
            'platform': self._select_optimal_platform(len(dna_sequence)),
            'estimated_cost': len(dna_sequence) * 0.10,  # $0.10 per nucleotide
            'estimated_time': len(dna_sequence) / 100,   # minutes
            'quality_score': validation['complexity'] * (1 - abs(validation['gc_content'] - 0.5)),
        }
        
        # Add optimization suggestions
        suggestions = []
        if validation['gc_content'] < 0.4 or validation['gc_content'] > 0.6:
            suggestions.append("Consider adjusting GC content for better synthesis yield")
        
        if validation['complexity'] < 0.5:
            suggestions.append("Low sequence complexity may cause synthesis issues")
        
        if len(dna_sequence) > 5000:
            suggestions.append("Consider splitting long sequences into smaller fragments")
        
        recommendations['suggestions'] = suggestions
        
        return recommendations