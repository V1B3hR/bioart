#!/usr/bin/env python3
"""
Biological Error Correction
Advanced error correction coding for biological environments
"""

import math
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ErrorType(Enum):
    """Types of DNA errors"""
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"
    INVERSION = "inversion"
    TRANSPOSITION = "transposition"

@dataclass
class ErrorPattern:
    """DNA error pattern"""
    error_type: ErrorType
    position: int
    original: str
    corrected: str
    confidence: float

class BiologicalErrorCorrection:
    """
    Advanced Error Correction for Biological DNA Storage
    Implements Reed-Solomon, biological-specific corrections, and redundancy
    """
    
    def __init__(self):
        """Initialize error correction system"""
        self.nucleotides = ['A', 'U', 'C', 'G']
        
        # Reed-Solomon parameters for DNA
        self.rs_n = 255  # Total symbols
        self.rs_k = 223  # Data symbols  
        self.rs_t = 16   # Error correction capability
        
        # Biological error patterns (observed frequencies)
        self.biological_error_rates = {
            ErrorType.SUBSTITUTION: 0.001,   # Point mutations
            ErrorType.INSERTION: 0.0005,     # Insertions
            ErrorType.DELETION: 0.0007,      # Deletions  
            ErrorType.INVERSION: 0.0001,     # Local inversions
            ErrorType.TRANSPOSITION: 0.00005 # Transpositions
        }
        
        # Context-specific error patterns
        self.context_errors = {
            'homopolymer': 0.01,    # Errors in homopolymer runs
            'hairpin': 0.005,       # Secondary structure errors
            'gc_rich': 0.003,       # GC-rich region errors
            'at_rich': 0.002,       # AT-rich region errors
        }
        
        # Correction matrices for common errors
        self.correction_matrices = self._build_correction_matrices()
    
    def encode_with_error_correction(self, dna_sequence: str, redundancy_level: int = 3) -> str:
        """
        Encode DNA sequence with error correction
        
        Args:
            dna_sequence: Input DNA sequence
            redundancy_level: Level of redundancy (1-5)
            
        Returns:
            Error-correction encoded sequence
        """
        # Apply Reed-Solomon encoding
        rs_encoded = self._reed_solomon_encode(dna_sequence)
        
        # Add biological redundancy
        redundant_encoded = self._add_biological_redundancy(rs_encoded, redundancy_level)
        
        # Add checksum sequences
        checksum_encoded = self._add_checksum_sequences(redundant_encoded)
        
        # Add synchronization markers
        final_encoded = self._add_sync_markers(checksum_encoded)
        
        return final_encoded
    
    def decode_with_error_correction(self, encoded_sequence: str) -> Tuple[str, List[ErrorPattern]]:
        """
        Decode sequence and correct errors
        
        Args:
            encoded_sequence: Error-correction encoded sequence
            
        Returns:
            Tuple of (corrected_sequence, detected_errors)
        """
        detected_errors = []
        
        # Remove synchronization markers
        sequence_no_sync = self._remove_sync_markers(encoded_sequence)
        
        # Verify and remove checksums
        sequence_no_checksum, checksum_errors = self._verify_and_remove_checksums(sequence_no_sync)
        detected_errors.extend(checksum_errors)
        
        # Process biological redundancy
        sequence_no_redundancy, redundancy_errors = self._process_biological_redundancy(sequence_no_checksum)
        detected_errors.extend(redundancy_errors)
        
        # Apply Reed-Solomon decoding
        final_sequence, rs_errors = self._reed_solomon_decode(sequence_no_redundancy)
        detected_errors.extend(rs_errors)
        
        # Apply biological pattern corrections
        corrected_sequence, pattern_errors = self._apply_biological_corrections(final_sequence)
        detected_errors.extend(pattern_errors)
        
        return corrected_sequence, detected_errors
    
    def _reed_solomon_encode(self, dna_sequence: str) -> str:
        """Apply Reed-Solomon encoding to DNA sequence"""
        # Convert DNA to numeric representation
        numeric_data = self._dna_to_numeric(dna_sequence)
        
        # Pad to multiple of k
        while len(numeric_data) % self.rs_k != 0:
            numeric_data.append(0)
        
        encoded_blocks = []
        
        # Process in blocks
        for i in range(0, len(numeric_data), self.rs_k):
            block = numeric_data[i:i + self.rs_k]
            
            # Calculate parity symbols (simplified Reed-Solomon)
            parity_symbols = self._calculate_rs_parity(block)
            
            # Combine data and parity
            encoded_block = block + parity_symbols
            encoded_blocks.extend(encoded_block)
        
        # Convert back to DNA
        return self._numeric_to_dna(encoded_blocks)
    
    def _reed_solomon_decode(self, encoded_sequence: str) -> Tuple[str, List[ErrorPattern]]:
        """Apply Reed-Solomon decoding"""
        numeric_data = self._dna_to_numeric(encoded_sequence)
        detected_errors = []
        decoded_blocks = []
        
        # Process in blocks of n symbols
        for i in range(0, len(numeric_data), self.rs_n):
            if i + self.rs_n <= len(numeric_data):
                block = numeric_data[i:i + self.rs_n]
                
                # Detect and correct errors
                corrected_block, block_errors = self._correct_rs_block(block, i)
                detected_errors.extend(block_errors)
                
                # Extract data portion (remove parity)
                data_portion = corrected_block[:self.rs_k]
                decoded_blocks.extend(data_portion)
        
        # Convert back to DNA and remove padding
        decoded_sequence = self._numeric_to_dna(decoded_blocks)
        return self._remove_padding(decoded_sequence), detected_errors
    
    def _add_biological_redundancy(self, sequence: str, redundancy_level: int) -> str:
        """Add biological redundancy patterns"""
        if redundancy_level == 1:
            return sequence
        
        redundant_sequence = []
        
        for i, nucleotide in enumerate(sequence):
            redundant_sequence.append(nucleotide)
            
            # Add redundancy based on level
            if redundancy_level >= 2:
                # Add complement
                complement = self._get_complement(nucleotide)
                redundant_sequence.append(complement)
            
            if redundancy_level >= 3:
                # Add repetition
                redundant_sequence.append(nucleotide)
            
            if redundancy_level >= 4:
                # Add error detection pattern
                pattern = self._generate_error_detection_pattern(nucleotide, i)
                redundant_sequence.extend(pattern)
            
            if redundancy_level >= 5:
                # Add maximum redundancy
                redundant_sequence.extend([nucleotide, complement, nucleotide])
        
        return ''.join(redundant_sequence)
    
    def _process_biological_redundancy(self, sequence: str) -> Tuple[str, List[ErrorPattern]]:
        """Process and verify biological redundancy"""
        detected_errors = []
        corrected_sequence = []
        
        # This is a simplified implementation
        # In practice, would need to know the original redundancy level
        
        i = 0
        while i < len(sequence):
            if i + 2 < len(sequence):
                # Check triplet pattern (assuming level 3 redundancy)
                nucleotide = sequence[i]
                complement = sequence[i + 1]
                repeat = sequence[i + 2]
                
                expected_complement = self._get_complement(nucleotide)
                
                # Majority voting
                candidates = [nucleotide, self._get_complement(complement), repeat]
                most_common = max(set(candidates), key=candidates.count)
                
                if most_common != nucleotide:
                    error = ErrorPattern(
                        error_type=ErrorType.SUBSTITUTION,
                        position=len(corrected_sequence),
                        original=nucleotide,
                        corrected=most_common,
                        confidence=0.8
                    )
                    detected_errors.append(error)
                
                corrected_sequence.append(most_common)
                i += 3
            else:
                corrected_sequence.append(sequence[i])
                i += 1
        
        return ''.join(corrected_sequence), detected_errors
    
    def _add_checksum_sequences(self, sequence: str) -> str:
        """Add DNA checksum sequences"""
        # Calculate multiple checksums for robustness
        
        # Simple XOR checksum
        xor_checksum = 0
        for nucleotide in sequence:
            xor_checksum ^= self._nucleotide_to_int(nucleotide)
        
        xor_checksum_dna = self._int_to_nucleotide(xor_checksum % 4)
        
        # CRC-like checksum for DNA
        crc_checksum = self._calculate_dna_crc(sequence)
        crc_checksum_dna = self._int_to_nucleotide(crc_checksum % 4)
        
        # Length checksum
        length_checksum = len(sequence) % 4
        length_checksum_dna = self._int_to_nucleotide(length_checksum)
        
        # Combine checksums
        checksum_sequence = xor_checksum_dna + crc_checksum_dna + length_checksum_dna
        
        return sequence + "AAAA" + checksum_sequence + "GGGG"  # Markers around checksum
    
    def _verify_and_remove_checksums(self, sequence: str) -> Tuple[str, List[ErrorPattern]]:
        """Verify and remove checksum sequences"""
        detected_errors = []
        
        # Find checksum markers
        start_marker = "AAAA"
        end_marker = "GGGG"
        
        start_idx = sequence.rfind(start_marker)
        end_idx = sequence.rfind(end_marker)
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            # Checksum corrupted
            error = ErrorPattern(
                error_type=ErrorType.DELETION,
                position=-1,
                original="checksum",
                corrected="missing",
                confidence=1.0
            )
            detected_errors.append(error)
            return sequence, detected_errors
        
        # Extract data and checksum
        data_sequence = sequence[:start_idx]
        checksum_sequence = sequence[start_idx + len(start_marker):end_idx]
        
        if len(checksum_sequence) != 3:
            error = ErrorPattern(
                error_type=ErrorType.DELETION,
                position=-1,
                original="checksum",
                corrected="corrupted",
                confidence=0.9
            )
            detected_errors.append(error)
            return data_sequence, detected_errors
        
        # Verify checksums
        expected_xor = self._nucleotide_to_int(checksum_sequence[0])
        expected_crc = self._nucleotide_to_int(checksum_sequence[1])
        expected_length = self._nucleotide_to_int(checksum_sequence[2])
        
        # Calculate actual checksums
        actual_xor = 0
        for nucleotide in data_sequence:
            actual_xor ^= self._nucleotide_to_int(nucleotide)
        actual_xor %= 4
        
        actual_crc = self._calculate_dna_crc(data_sequence) % 4
        actual_length = len(data_sequence) % 4
        
        # Check for errors
        if actual_xor != expected_xor:
            error = ErrorPattern(
                error_type=ErrorType.SUBSTITUTION,
                position=-1,
                original="data",
                corrected="xor_error",
                confidence=0.7
            )
            detected_errors.append(error)
        
        if actual_crc != expected_crc:
            error = ErrorPattern(
                error_type=ErrorType.SUBSTITUTION,
                position=-1,
                original="data",
                corrected="crc_error",
                confidence=0.8
            )
            detected_errors.append(error)
        
        if actual_length != expected_length:
            error = ErrorPattern(
                error_type=ErrorType.INSERTION,
                position=-1,
                original="data",
                corrected="length_error",
                confidence=0.9
            )
            detected_errors.append(error)
        
        return data_sequence, detected_errors
    
    def _add_sync_markers(self, sequence: str) -> str:
        """Add synchronization markers"""
        # Add start and end markers
        start_marker = "AUAUAUAU"  # Distinctive pattern
        end_marker = "CGGCCGGC"    # Another distinctive pattern
        
        # Add periodic sync markers throughout sequence
        marked_sequence = [start_marker]
        
        for i, nucleotide in enumerate(sequence):
            marked_sequence.append(nucleotide)
            
            # Add sync marker every 64 nucleotides
            if (i + 1) % 64 == 0:
                marked_sequence.append("AUCG")  # Short sync marker
        
        marked_sequence.append(end_marker)
        
        return ''.join(marked_sequence)
    
    def _remove_sync_markers(self, sequence: str) -> str:
        """Remove synchronization markers"""
        # Remove start and end markers
        start_marker = "AUAUAUAU"
        end_marker = "CGGCCGGC"
        
        # Find and remove markers
        if sequence.startswith(start_marker):
            sequence = sequence[len(start_marker):]
        
        if sequence.endswith(end_marker):
            sequence = sequence[:-len(end_marker)]
        
        # Remove periodic sync markers
        sync_marker = "AUCG"
        cleaned_sequence = sequence.replace(sync_marker, "")
        
        return cleaned_sequence
    
    def _apply_biological_corrections(self, sequence: str) -> Tuple[str, List[ErrorPattern]]:
        """Apply biological-specific error corrections"""
        detected_errors = []
        corrected_sequence = list(sequence)
        
        # Fix homopolymer runs (common DNA synthesis errors)
        for i in range(len(corrected_sequence) - 4):
            if len(set(corrected_sequence[i:i+5])) == 1:  # 5+ identical nucleotides
                # Break up homopolymer
                nucleotide = corrected_sequence[i]
                complement = self._get_complement(nucleotide)
                corrected_sequence[i+2] = complement
                
                error = ErrorPattern(
                    error_type=ErrorType.SUBSTITUTION,
                    position=i+2,
                    original=nucleotide,
                    corrected=complement,
                    confidence=0.6
                )
                detected_errors.append(error)
        
        # Fix obvious invalid patterns
        for i in range(len(corrected_sequence)):
            if corrected_sequence[i] not in self.nucleotides:
                corrected_sequence[i] = 'A'  # Default replacement
                
                error = ErrorPattern(
                    error_type=ErrorType.SUBSTITUTION,
                    position=i,
                    original="invalid",
                    corrected='A',
                    confidence=1.0
                )
                detected_errors.append(error)
        
        # Apply context-specific corrections using correction matrices
        for i in range(1, len(corrected_sequence) - 1):
            context = corrected_sequence[i-1:i+2]
            correction = self._get_context_correction(context, i)
            
            if correction and correction != corrected_sequence[i]:
                error = ErrorPattern(
                    error_type=ErrorType.SUBSTITUTION,
                    position=i,
                    original=corrected_sequence[i],
                    corrected=correction,
                    confidence=0.5
                )
                detected_errors.append(error)
                corrected_sequence[i] = correction
        
        return ''.join(corrected_sequence), detected_errors
    
    def _build_correction_matrices(self) -> Dict[str, Dict[str, str]]:
        """Build correction matrices for common error patterns"""
        matrices = {}
        
        # Context-based corrections
        # These would be learned from biological data
        matrices['context'] = {
            'AAA': 'A',  # Likely correct in middle of AAA
            'GGG': 'G',  # Likely correct in middle of GGG
            'AUA': 'U',  # Likely correct
            'CUC': 'U',  # Likely correct
        }
        
        # Secondary structure corrections
        matrices['hairpin'] = {
            'GCGC': 'GCAC',  # Common hairpin correction
            'CGCG': 'CGAG',  # Another common pattern
        }
        
        return matrices
    
    def _get_context_correction(self, context: List[str], position: int) -> Optional[str]:
        """Get context-based correction suggestion"""
        context_str = ''.join(context)
        
        if context_str in self.correction_matrices['context']:
            return self.correction_matrices['context'][context_str]
        
        # Check for hairpin patterns
        for pattern, correction in self.correction_matrices['hairpin'].items():
            if pattern in context_str:
                return correction[len(pattern)//2]  # Middle nucleotide
        
        return None
    
    # Helper methods
    def _get_complement(self, nucleotide: str) -> str:
        """Get complement of nucleotide"""
        complements = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
        return complements.get(nucleotide, 'A')
    
    def _nucleotide_to_int(self, nucleotide: str) -> int:
        """Convert nucleotide to integer"""
        mapping = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
        return mapping.get(nucleotide, 0)
    
    def _int_to_nucleotide(self, value: int) -> str:
        """Convert integer to nucleotide"""
        mapping = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}
        return mapping.get(value % 4, 'A')
    
    def _dna_to_numeric(self, dna_sequence: str) -> List[int]:
        """Convert DNA sequence to numeric representation"""
        return [self._nucleotide_to_int(nuc) for nuc in dna_sequence]
    
    def _numeric_to_dna(self, numeric_data: List[int]) -> str:
        """Convert numeric data to DNA sequence"""
        return ''.join(self._int_to_nucleotide(val) for val in numeric_data)
    
    def _calculate_rs_parity(self, data_block: List[int]) -> List[int]:
        """Calculate Reed-Solomon parity symbols (simplified)"""
        parity_count = self.rs_n - self.rs_k
        parity_symbols = []
        
        for i in range(parity_count):
            parity = 0
            for j, data_symbol in enumerate(data_block):
                parity ^= data_symbol * ((j + i) % 4)  # Simplified calculation
            parity_symbols.append(parity % 4)
        
        return parity_symbols
    
    def _correct_rs_block(self, block: List[int], block_position: int) -> Tuple[List[int], List[ErrorPattern]]:
        """Correct Reed-Solomon block (simplified)"""
        detected_errors = []
        
        # Extract data and parity
        data_portion = block[:self.rs_k]
        parity_portion = block[self.rs_k:]
        
        # Recalculate parity
        expected_parity = self._calculate_rs_parity(data_portion)
        
        # Compare with received parity
        error_positions = []
        for i, (expected, received) in enumerate(zip(expected_parity, parity_portion)):
            if expected != received:
                error_positions.append(i + self.rs_k)
        
        # Simple error correction (more sophisticated RS would be implemented here)
        corrected_block = block.copy()
        
        for pos in error_positions[:self.rs_t]:  # Correct up to t errors
            # Simple bit flip correction
            corrected_block[pos] = expected_parity[pos - self.rs_k]
            
            error = ErrorPattern(
                error_type=ErrorType.SUBSTITUTION,
                position=block_position + pos,
                original=str(block[pos]),
                corrected=str(corrected_block[pos]),
                confidence=0.9
            )
            detected_errors.append(error)
        
        return corrected_block, detected_errors
    
    def _calculate_dna_crc(self, sequence: str) -> int:
        """Calculate CRC-like checksum for DNA"""
        crc = 0
        for i, nucleotide in enumerate(sequence):
            value = self._nucleotide_to_int(nucleotide)
            crc = ((crc << 2) ^ value) & 0xFF  # 8-bit CRC
        return crc
    
    def _generate_error_detection_pattern(self, nucleotide: str, position: int) -> List[str]:
        """Generate error detection pattern for nucleotide"""
        # Simple pattern based on position and nucleotide
        base_value = self._nucleotide_to_int(nucleotide)
        pattern_value = (base_value + position) % 4
        return [self._int_to_nucleotide(pattern_value)]
    
    def _remove_padding(self, sequence: str) -> str:
        """Remove padding from decoded sequence"""
        # Remove trailing null characters (represented as 'A')
        return sequence.rstrip('A')
    
    def get_error_correction_statistics(self) -> Dict[str, Any]:
        """Get error correction system statistics"""
        return {
            'reed_solomon_params': {
                'n': self.rs_n,
                'k': self.rs_k,
                't': self.rs_t,
                'efficiency': self.rs_k / self.rs_n
            },
            'biological_error_rates': {etype.value: rate for etype, rate in self.biological_error_rates.items()},
            'context_error_rates': self.context_errors,
            'correction_matrices_count': sum(len(matrix) for matrix in self.correction_matrices.values())
        }