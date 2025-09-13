#!/usr/bin/env python3
"""
Bioartlan Programming Language - Refactored Main Module
Integrated high-performance bioartlan programming language with modular architecture
"""

import os
import sys
from typing import Optional, Dict, Any, List
import json
import time

from .core.encoding import DNAEncoder
from .vm.instruction_set import DNAInstructionSet
from .vm.virtual_machine import DNAVirtualMachine, VMState
from .compiler.dna_compiler import DNACompiler
from .utils.file_manager import DNAFileManager

class BioartlanLanguage:
    """
    Main Bioartlan Programming Language interface
    Provides high-level API for bioartlan programming operations
    """
    
    def __init__(self, memory_size: int = 256, register_count: int = 4):
        """Initialize Bioartlan Language system"""
        # Core components
        self.encoder = DNAEncoder()
        self.instruction_set = DNAInstructionSet()
        self.vm = DNAVirtualMachine(memory_size, register_count)
        self.compiler = DNACompiler(self.encoder, self.instruction_set)
        self.file_manager = DNAFileManager()
        
        # System configuration
        self.config = {
            'memory_size': memory_size,
            'register_count': register_count,
            'debug_mode': False,
            'optimization_level': 1,
            'output_format': 'text'
        }
        
        # Performance metrics
        self.metrics = {
            'programs_compiled': 0,
            'programs_executed': 0,
            'total_execution_time': 0.0,
            'total_instructions_executed': 0
        }
    
    def compile_dna_program(self, dna_source: str, optimize: bool = True) -> bytes:
        """
        Compile DNA source code to optimized bytecode
        
        Args:
            dna_source: DNA source code string
            optimize: Enable compiler optimizations
            
        Returns:
            Compiled bytecode
        """
        try:
            bytecode = self.compiler.compile(dna_source, optimize)
            self.metrics['programs_compiled'] += 1
            return bytecode
        except Exception as e:
            raise RuntimeError(f"Compilation failed: {e}")
    
    def execute_dna_program(self, dna_source: str, max_cycles: int = 1000000) -> Dict[str, Any]:
        """
        Compile and execute DNA program
        
        Args:
            dna_source: DNA source code
            max_cycles: Maximum execution cycles
            
        Returns:
            Execution results and statistics
        """
        try:
            # Compile program
            bytecode = self.compile_dna_program(dna_source)
            
            # Load and execute
            self.vm.load_program(bytecode)
            stats = self.vm.execute(max_cycles)
            
            # Update metrics
            self.metrics['programs_executed'] += 1
            self.metrics['total_execution_time'] += stats.execution_time
            self.metrics['total_instructions_executed'] += stats.instructions_executed
            
            return {
                'success': True,
                'vm_state': self.vm.get_state_info(),
                'execution_stats': {
                    'instructions_executed': stats.instructions_executed,
                    'cycles_elapsed': stats.cycles_elapsed,
                    'execution_time': stats.execution_time,
                    'memory_accesses': stats.memory_accesses,
                    'io_operations': stats.io_operations
                },
                'final_state': self.vm.state.value
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'vm_state': self.vm.get_state_info() if hasattr(self, 'vm') else None
            }
    
    def encode_data(self, data: bytes) -> str:
        """Encode binary data to DNA sequence"""
        return self.encoder.encode_bytes(data)
    
    def decode_data(self, dna_sequence: str) -> bytes:
        """Decode DNA sequence to binary data"""
        return self.encoder.decode_dna(dna_sequence)
    
    def encode_string(self, text: str) -> str:
        """Encode text string to DNA sequence"""
        return self.encoder.encode_string(text)
    
    def decode_string(self, dna_sequence: str) -> str:
        """Decode DNA sequence to text string"""
        return self.encoder.decode_to_string(dna_sequence)
    
    def save_dna_program(self, filename: str, dna_source: str, metadata: Optional[Dict] = None):
        """Save DNA program to file with metadata"""
        self.file_manager.save_dna_program(filename, dna_source, metadata)
    
    def load_dna_program(self, filename: str) -> Dict[str, Any]:
        """Load DNA program from file"""
        return self.file_manager.load_dna_program(filename)
    
    def save_compiled_program(self, filename: str, bytecode: bytes, metadata: Optional[Dict] = None):
        """Save compiled bytecode to file"""
        self.file_manager.save_compiled_program(filename, bytecode, metadata)
    
    def load_compiled_program(self, filename: str) -> bytes:
        """Load compiled bytecode from file"""
        return self.file_manager.load_compiled_program(filename)
    
    def disassemble_program(self, bytecode: bytes) -> List[str]:
        """Disassemble bytecode to human-readable assembly"""
        self.vm.load_program(bytecode)
        return self.vm.disassemble_program()
    
    def validate_dna_sequence(self, dna_sequence: str) -> Dict[str, Any]:
        """Validate DNA sequence format and content"""
        is_valid, message = self.encoder.validate_dna_sequence(dna_sequence)
        
        result = {
            'valid': is_valid,
            'message': message,
            'length': len(dna_sequence.replace(' ', '')),
            'byte_count': len(dna_sequence.replace(' ', '')) // 4
        }
        
        if is_valid:
            try:
                # Additional validation as instruction sequence
                is_program_valid, errors = self.instruction_set.validate_program(dna_sequence)
                result['valid_as_program'] = is_program_valid
                result['program_errors'] = errors
            except:
                result['valid_as_program'] = False
                result['program_errors'] = ['Failed to validate as program']
        
        return result
    
    def get_instruction_info(self, identifier: Any) -> Optional[Dict]:
        """Get detailed information about an instruction"""
        return self.instruction_set.get_instruction_info(identifier)
    
    def list_all_instructions(self) -> List[Dict]:
        """Get list of all available instructions"""
        return self.instruction_set.list_all_instructions()
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'language_info': {
                'version': '2.0.0-refactored',
                'encoding': '2-bit DNA (A=00, U=01, C=10, G=11)',
                'architecture': f'{self.config["memory_size"]} bytes memory, {self.config["register_count"]} registers'
            },
            'instruction_set': self.instruction_set.get_instruction_statistics(),
            'performance_metrics': self.metrics.copy(),
            'vm_configuration': self.config.copy(),
            'encoding_stats': {
                'nucleotides_per_byte': 4,
                'theoretical_max_efficiency': '100%',
                'reversibility': 'Perfect (lossless)'
            }
        }
    
    def benchmark_performance(self, test_data_size: int = 1000) -> Dict[str, Any]:
        """Run performance benchmarks"""
        import random
        
        # Generate test data
        test_data = bytes([random.randint(0, 255) for _ in range(test_data_size)])
        
        # Benchmark encoding
        start_time = time.time()
        dna_sequence = self.encoder.encode_bytes(test_data)
        encoding_time = time.time() - start_time
        
        # Benchmark decoding
        start_time = time.time()
        decoded_data = self.encoder.decode_dna(dna_sequence)
        decoding_time = time.time() - start_time
        
        # Verify accuracy
        accuracy = (test_data == decoded_data)
        
        return {
            'test_size_bytes': test_data_size,
            'dna_sequence_length': len(dna_sequence),
            'encoding_time_ms': encoding_time * 1000,
            'decoding_time_ms': decoding_time * 1000,
            'encoding_speed_mbps': (test_data_size / encoding_time) / (1024 * 1024),
            'decoding_speed_mbps': (test_data_size / decoding_time) / (1024 * 1024),
            'accuracy': accuracy,
            'efficiency_ratio': len(dna_sequence) / test_data_size
        }
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.config['debug_mode'] = enabled
        self.vm.set_debug_mode(enabled)
    
    def add_breakpoint(self, address: int):
        """Add breakpoint for debugging"""
        self.vm.add_breakpoint(address)
    
    def remove_breakpoint(self, address: int):
        """Remove breakpoint"""
        self.vm.remove_breakpoint(address)
    
    def step_execution(self):
        """Execute single instruction (debug mode)"""
        if self.vm.state == VMState.PAUSED:
            # Implementation for single-step execution
            pass
    
    def get_memory_dump(self, start: int = 0, length: int = 16) -> List[int]:
        """Get memory dump for debugging"""
        end = min(start + length, len(self.vm.memory))
        return list(self.vm.memory[start:end])
    
    def reset_system(self):
        """Reset entire system to initial state"""
        self.vm.reset()
        self.metrics = {
            'programs_compiled': 0,
            'programs_executed': 0,
            'total_execution_time': 0.0,
            'total_instructions_executed': 0
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current system configuration"""
        return {
            'config': self.config.copy(),
            'metrics': self.metrics.copy(),
            'vm_state': self.vm.get_state_info(),
            'timestamp': time.time()
        }
    
    def import_configuration(self, config_data: Dict[str, Any]):
        """Import system configuration"""
        if 'config' in config_data:
            self.config.update(config_data['config'])
            
        # Apply configuration to VM
        if 'debug_mode' in self.config:
            self.vm.set_debug_mode(self.config['debug_mode'])
    
    def create_sample_programs(self) -> Dict[str, str]:
        """Create sample DNA programs for demonstration"""
        return {
            'hello_world': 'AAAU UACA AAUG AAGA',  # Load 72('H'), Print, Halt
            'fibonacci': '''
                AAAU AAAA    # Load 0
                AAAC AAAA    # Store at address 0
                AAAU AAAU    # Load 1  
                AAAC AAAU    # Store at address 1
                AAAU AAAC    # Load 2 (counter)
                AAAC AAAC    # Store at address 2
            ''',
            'counter': '''
                AAAU AAAA    # Load 0
                AUCA         # Increment
                AAUG         # Print
                AAAU AAAU    # Load 1
                AAAG AAAA    # Add 0 (counter check)
                AACU AAAC    # Jump to increment
                AAGA         # Halt
            ''',
            'data_processing': '''
                AAAU ACAA    # Load data
                ACAU         # NOT operation
                AAAG AUAU    # Add mask
                AAUG         # Print result
                AAGA         # Halt
            '''
        }


# Convenience functions for direct use
def create_bioartlan_language(memory_size: int = 256, registers: int = 4) -> BioartlanLanguage:
    """Create a new Bioartlan Language instance"""
    return BioartlanLanguage(memory_size, registers)

def quick_execute(dna_program: str) -> Dict[str, Any]:
    """Quick execution of DNA program"""
    dna_lang = BioartlanLanguage()
    return dna_lang.execute_dna_program(dna_program)

def quick_encode(data: str) -> str:
    """Quick encoding of text to DNA"""
    dna_lang = BioartlanLanguage()
    return dna_lang.encode_string(data)

def quick_decode(dna_sequence: str) -> str:
    """Quick decoding of DNA to text"""
    dna_lang = BioartlanLanguage()
    return dna_lang.decode_string(dna_sequence)