#!/usr/bin/env python3
"""
Bioartlan Programming Language - Refactored Version 2.0
High-performance modular bioartlan programming language implementation
"""

__version__ = "2.0.0-refactored"
__author__ = "Bioartlan Programming Language Project"
__license__ = "MIT"

# Import main components for easy access
from .bioartlan_language import BioartlanLanguage, create_bioartlan_system
from .core.encoding import DNAEncoder, encode_bytes, decode_dna, encode_string, decode_to_string
from .vm.virtual_machine import DNAVirtualMachine, create_vm, VMState
from .vm.instruction_set import DNAInstructionSet, get_instruction, is_valid_instruction
from .compiler.dna_compiler import DNACompiler
from .utils.file_manager import DNAFileManager

# New enhanced modules
from .biological.synthesis_systems import DNASynthesisManager
from .biological.storage_systems import BiologicalStorageManager  
from .biological.genetic_tools import GeneticEngineeringInterface
from .biological.error_correction import BiologicalErrorCorrection
from .parallel.dna_threading import DNAThreadManager
from .parallel.parallel_executor import ParallelDNAExecutor, ExecutionStrategy
from .parallel.distributed_computing import DistributedDNAComputer, NodeType

# Version information
VERSION_INFO = {
    'version': __version__,
    'architecture': 'Modular refactored implementation',
    'encoding': '2-bit DNA (A=00, U=01, C=10, G=11)',
    'features': [
        'High-performance encoding with lookup tables',
        'Extended instruction set with 52+ instructions',
        'Advanced virtual machine with biological simulation',
        'Biological DNA synthesis and storage integration',
        'CRISPR and genetic engineering tools interface',
        'Multi-threading and parallel execution support',
        'Distributed computing capabilities', 
        'Advanced error correction for biological environments',
        'Real-time biological constraint simulation'
    ],
    'improvements': [
        'Biological synthesis systems integration',
        'Real DNA storage and retrieval mechanisms', 
        'Error correction coding for biological environments',
        'Multi-threading support for parallel DNA execution',
        'Interface with genetic engineering tools',
        'Extended instruction set for complex operations',
        'Advanced biological constraint simulation',
        'Distributed computing framework'
    ]
}

def get_version_info():
    """Get detailed version information"""
    return VERSION_INFO.copy()

def create_default_system():
    """Create a default bioartlan language system with standard configuration"""
    return create_bioartlan_system(memory_size=256, enable_all_features=True)

# Package-level convenience functions
def encode_text_to_dna(text: str) -> str:
    """Quick function to encode text to DNA"""
    encoder = DNAEncoder()
    return encoder.encode_string(text)

def decode_dna_to_text(dna_sequence: str) -> str:
    """Quick function to decode DNA to text"""
    encoder = DNAEncoder()
    return encoder.decode_to_string(dna_sequence)

def run_dna_program(dna_code: str) -> dict:
    """Quick function to run DNA program"""
    system = create_default_system()
    try:
        bytecode = system.compile_dna_program(dna_code)
        result = system.execute_program(bytecode)
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Export all public components
__all__ = [
    # Main classes
    'BioartlanLanguage',
    'DNAEncoder', 
    'DNAVirtualMachine',
    'DNAInstructionSet',
    'DNACompiler',
    'DNAFileManager',
    
    # Enums and types
    'VMState',
    
    # Factory functions
    'create_bioartlan_system',
    'create_vm',
    'create_default_system',
    
    # Convenience functions
    'encode_bytes',
    'decode_dna',
    'encode_string',
    'decode_to_string',
    'encode_text_to_dna',
    'decode_dna_to_text',
    'run_dna_program',
    'get_instruction',
    'is_valid_instruction',
    
    # Enhanced biological components
    'DNASynthesisManager',
    'BiologicalStorageManager',
    'GeneticEngineeringInterface', 
    'BiologicalErrorCorrection',
    
    # Parallel processing components
    'DNAThreadManager',
    'ParallelDNAExecutor',
    'ExecutionStrategy',
    'DistributedDNAComputer',
    'NodeType',
    
    # Utility functions
    'get_version_info',
    
    # Version info
    '__version__',
    'VERSION_INFO'
] 