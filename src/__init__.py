#!/usr/bin/env python3
"""
Bioartlan Programming Language - Refactored Version 2.0
High-performance modular bioartlan programming language implementation
"""

__version__ = "2.0.0-refactored"
__author__ = "Bioartlan Programming Language Project"
__license__ = "MIT"

# Import main components for easy access
from .bioartlan_language import BioartlanLanguage, create_bioartlan_language, quick_execute, quick_encode, quick_decode
from .core.encoding import DNAEncoder, encode_bytes, decode_dna, encode_string, decode_to_string
from .vm.virtual_machine import DNAVirtualMachine, create_vm, VMState
from .vm.instruction_set import DNAInstructionSet, get_instruction, is_valid_instruction
from .compiler.dna_compiler import DNACompiler
from .utils.file_manager import DNAFileManager

# Version information
VERSION_INFO = {
    'version': __version__,
    'architecture': 'Modular refactored implementation',
    'encoding': '2-bit DNA (A=00, U=01, C=10, G=11)',
    'features': [
        'High-performance encoding with lookup tables',
        'Extended instruction set with 18+ instructions',
        'Advanced virtual machine with debugging support',
        'Optimizing compiler with multiple passes',
        'File management with metadata and versioning',
        'Comprehensive error handling and validation'
    ],
    'improvements': [
        'Modular architecture for better maintainability',
        'Performance optimizations with lookup tables',
        'Extended instruction set for more functionality',
        'Debugging support with breakpoints and tracing',
        'File management with automatic backups',
        'Comprehensive testing and validation'
    ]
}

def get_version_info():
    """Get detailed version information"""
    return VERSION_INFO.copy()

def create_default_system():
    """Create a default bioartlan language system with standard configuration"""
    return BioartlanLanguage(memory_size=256, register_count=4)

# Package-level convenience functions
def encode_text_to_dna(text: str) -> str:
    """Quick function to encode text to DNA"""
    return quick_encode(text)

def decode_dna_to_text(dna_sequence: str) -> str:
    """Quick function to decode DNA to text"""
    return quick_decode(dna_sequence)

def run_dna_program(dna_code: str) -> dict:
    """Quick function to run DNA program"""
    return quick_execute(dna_code)

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
    'create_bioartlan_language',
    'create_vm',
    'create_default_system',
    
    # Convenience functions
    'quick_execute',
    'quick_encode', 
    'quick_decode',
    'encode_bytes',
    'decode_dna',
    'encode_string',
    'decode_to_string',
    'encode_text_to_dna',
    'decode_dna_to_text',
    'run_dna_program',
    'get_instruction',
    'is_valid_instruction',
    
    # Utility functions
    'get_version_info',
    
    # Version info
    '__version__',
    'VERSION_INFO'
] 