#!/usr/bin/env python3
"""
DNA Virtual Machine
High-performance virtual machine with improved architecture and debugging
"""

from typing import List, Dict, Optional, Callable, Any
import time
from enum import Enum
from dataclasses import dataclass

from ..core.encoding import DNAEncoder
from .instruction_set import DNAInstructionSet, InstructionType

class VMState(Enum):
    """Virtual machine execution states"""
    STOPPED = "stopped"
    RUNNING = "running" 
    PAUSED = "paused"
    HALTED = "halted"
    ERROR = "error"

@dataclass
class ExecutionStats:
    """Execution statistics tracking"""
    instructions_executed: int = 0
    cycles_elapsed: int = 0
    execution_time: float = 0.0
    memory_accesses: int = 0
    io_operations: int = 0

class DNAVirtualMachine:
    """
    High-performance DNA Virtual Machine with advanced features
    """
    
    def __init__(self, memory_size: int = 256, register_count: int = 4):
        """Initialize virtual machine with configurable architecture"""
        
        # Core components
        self.encoder = DNAEncoder()
        self.instruction_set = DNAInstructionSet()
        
        # Architecture configuration
        self.memory_size = memory_size
        self.register_count = register_count
        
        # VM State
        self.state = VMState.STOPPED
        self.reset()
        
        # Performance tracking
        self.stats = ExecutionStats()
        
        # Debugging support
        self.debug_mode = False
        self.breakpoints = set()
        self.instruction_trace = []
        
        # I/O handlers
        self.input_handler = input
        self.output_handler = print
        
        # Instruction execution mapping
        self._setup_instruction_handlers()
    
    def reset(self):
        """Reset virtual machine to initial state"""
        # Memory initialization
        self.memory = bytearray(self.memory_size)
        
        # Register initialization (A, B, C, D by default)
        self.registers = [0] * self.register_count
        
        # Control registers
        self.pc = 0  # Program counter
        self.sp = self.memory_size - 1  # Stack pointer
        self.flags = {
            'zero': False,
            'carry': False,
            'overflow': False,
            'negative': False
        }
        
        # Execution state
        self.state = VMState.STOPPED
        self.program = bytearray()
        self.program_size = 0
        
        # Reset statistics
        self.stats = ExecutionStats()
        self.instruction_trace.clear()
    
    def _setup_instruction_handlers(self):
        """Setup instruction execution handlers"""
        self.instruction_handlers = {
            0x00: self._execute_nop,
            0x01: self._execute_load,
            0x02: self._execute_store,
            0x03: self._execute_add,
            0x04: self._execute_sub,
            0x05: self._execute_mul,
            0x06: self._execute_div,
            0x07: self._execute_print,
            0x08: self._execute_input,
            0x09: self._execute_jmp,
            0x0A: self._execute_jeq,
            0x0B: self._execute_jne,
            0x0C: self._execute_halt,
            0x0D: self._execute_loadr,
            0x0E: self._execute_storer,
            0x0F: self._execute_mod,
            0x10: self._execute_inc,
            0x11: self._execute_dec,
            0x12: self._execute_and,
            0x13: self._execute_or,
            0x14: self._execute_xor,
            0x15: self._execute_not,
            0x16: self._execute_printc,
            0x17: self._execute_prints,
        }
    
    def load_program(self, program_data: bytes):
        """Load program into memory"""
        if len(program_data) > self.memory_size:
            raise ValueError(f"Program too large: {len(program_data)} bytes > {self.memory_size}")
        
        self.program = bytearray(program_data)
        self.program_size = len(program_data)
        
        # Load program into memory starting at address 0
        self.memory[:len(program_data)] = program_data
        
        self.pc = 0
        self.state = VMState.STOPPED
    
    def load_dna_program(self, dna_program: str):
        """Load DNA program by compiling to bytecode"""
        # Validate program
        is_valid, errors = self.instruction_set.validate_program(dna_program)
        if not is_valid:
            raise ValueError(f"Invalid DNA program: {errors}")
        
        # Compile DNA to bytecode
        bytecode = self._compile_dna_program(dna_program)
        self.load_program(bytecode)
    
    def _compile_dna_program(self, dna_program: str) -> bytes:
        """Compile DNA program to bytecode"""
        sequences = dna_program.replace(' ', '').replace('\n', '')
        bytecode = []
        
        for i in range(0, len(sequences), 4):
            dna_seq = sequences[i:i+4]
            opcode = self.instruction_set.dna_to_opcode(dna_seq)
            if opcode is not None:
                bytecode.append(opcode)
            else:
                # Try to interpret as data
                try:
                    byte_val = self.encoder.dna_to_byte(dna_seq)
                    bytecode.append(byte_val)
                except:
                    raise ValueError(f"Invalid DNA sequence: {dna_seq}")
        
        return bytes(bytecode)
    
    def execute(self, max_cycles: int = 1000000) -> ExecutionStats:
        """Execute loaded program with cycle limit"""
        if not self.program:
            raise RuntimeError("No program loaded")
        
        self.state = VMState.RUNNING
        start_time = time.time()
        cycles = 0
        
        try:
            while (self.state == VMState.RUNNING and 
                   cycles < max_cycles and 
                   self.pc < len(self.program)):
                
                # Check breakpoints in debug mode
                if self.debug_mode and self.pc in self.breakpoints:
                    self.state = VMState.PAUSED
                    break
                
                # Fetch instruction
                if self.pc >= len(self.program):
                    break
                
                opcode = self.program[self.pc]
                instruction = self.instruction_set.get_instruction_by_opcode(opcode)
                
                if instruction:
                    # Execute instruction
                    self._execute_instruction(instruction)
                    cycles += instruction.cycles
                    self.stats.instructions_executed += 1
                    self.stats.cycles_elapsed += instruction.cycles
                    
                    # Debug trace
                    if self.debug_mode:
                        self.instruction_trace.append({
                            'pc': self.pc,
                            'opcode': opcode,
                            'instruction': instruction.name,
                            'registers': self.registers.copy(),
                            'flags': self.flags.copy()
                        })
                else:
                    # Invalid instruction
                    self.state = VMState.ERROR
                    raise RuntimeError(f"Invalid instruction at PC {self.pc}: 0x{opcode:02X}")
        
        except Exception as e:
            self.state = VMState.ERROR
            raise RuntimeError(f"VM execution error at PC {self.pc}: {e}")
        
        finally:
            execution_time = time.time() - start_time
            self.stats.execution_time = execution_time
            
            if self.state == VMState.RUNNING:
                self.state = VMState.STOPPED
        
        return self.stats
    
    def _execute_instruction(self, instruction):
        """Execute single instruction"""
        handler = self.instruction_handlers.get(instruction.opcode)
        if handler:
            handler(instruction)
        else:
            raise RuntimeError(f"No handler for instruction: {instruction.name}")
    
    def _update_flags(self, result: int):
        """Update processor flags based on result"""
        self.flags['zero'] = (result == 0)
        self.flags['negative'] = (result < 0)
        self.flags['carry'] = (result > 255 or result < 0)
        self.flags['overflow'] = (result > 127 or result < -128)
    
    def _get_operand(self) -> int:
        """Get next operand from program"""
        self.pc += 1
        if self.pc < len(self.program):
            return self.program[self.pc]
        return 0
    
    # Instruction implementations
    def _execute_nop(self, instruction):
        """NOP - No operation"""
        self.pc += 1
    
    def _execute_load(self, instruction):
        """LOAD - Load immediate value to register A"""
        value = self._get_operand()
        self.registers[0] = value
        self._update_flags(value)
        self.pc += 1
    
    def _execute_store(self, instruction):
        """STORE - Store register A to memory address"""
        address = self._get_operand()
        if address < len(self.memory):
            self.memory[address] = self.registers[0] & 0xFF
            self.stats.memory_accesses += 1
        self.pc += 1
    
    def _execute_add(self, instruction):
        """ADD - Add immediate value to register A"""
        value = self._get_operand()
        result = self.registers[0] + value
        self.registers[0] = result & 0xFF
        self._update_flags(result)
        self.pc += 1
    
    def _execute_sub(self, instruction):
        """SUB - Subtract immediate value from register A"""
        value = self._get_operand()
        result = self.registers[0] - value
        self.registers[0] = result & 0xFF
        self._update_flags(result)
        self.pc += 1
    
    def _execute_mul(self, instruction):
        """MUL - Multiply register A by immediate value"""
        value = self._get_operand()
        result = self.registers[0] * value
        self.registers[0] = result & 0xFF
        self._update_flags(result)
        self.pc += 1
    
    def _execute_div(self, instruction):
        """DIV - Divide register A by immediate value"""
        value = self._get_operand()
        if value == 0:
            raise RuntimeError("Division by zero")
        result = self.registers[0] // value
        self.registers[0] = result & 0xFF
        self._update_flags(result)
        self.pc += 1
    
    def _execute_print(self, instruction):
        """PRINT - Print register A value"""
        self.output_handler(f"Output: {self.registers[0]}")
        self.stats.io_operations += 1
        self.pc += 1
    
    def _execute_input(self, instruction):
        """INPUT - Read input to register A"""
        try:
            value = int(self.input_handler("Input: "))
            self.registers[0] = value & 0xFF
            self.stats.io_operations += 1
        except (ValueError, EOFError):
            self.registers[0] = 0
        self.pc += 1
    
    def _execute_jmp(self, instruction):
        """JMP - Unconditional jump"""
        address = self._get_operand()
        self.pc = address
    
    def _execute_jeq(self, instruction):
        """JEQ - Jump if equal (zero flag set)"""
        address = self._get_operand()
        if self.flags['zero']:
            self.pc = address
        else:
            self.pc += 1
    
    def _execute_jne(self, instruction):
        """JNE - Jump if not equal (zero flag clear)"""
        address = self._get_operand()
        if not self.flags['zero']:
            self.pc = address
        else:
            self.pc += 1
    
    def _execute_halt(self, instruction):
        """HALT - Stop execution"""
        self.state = VMState.HALTED
        self.pc += 1
    
    def _execute_loadr(self, instruction):
        """LOADR - Load from register to register A"""
        reg_num = self._get_operand()
        if reg_num < len(self.registers):
            self.registers[0] = self.registers[reg_num]
        self.pc += 1
    
    def _execute_storer(self, instruction):
        """STORER - Store register A to another register"""
        reg_num = self._get_operand()
        if reg_num < len(self.registers):
            self.registers[reg_num] = self.registers[0]
        self.pc += 1
    
    def _execute_mod(self, instruction):
        """MOD - Modulo operation"""
        value = self._get_operand()
        if value == 0:
            raise RuntimeError("Modulo by zero")
        result = self.registers[0] % value
        self.registers[0] = result & 0xFF
        self._update_flags(result)
        self.pc += 1
    
    def _execute_inc(self, instruction):
        """INC - Increment register A"""
        result = self.registers[0] + 1
        self.registers[0] = result & 0xFF
        self._update_flags(result)
        self.pc += 1
    
    def _execute_dec(self, instruction):
        """DEC - Decrement register A"""
        result = self.registers[0] - 1
        self.registers[0] = result & 0xFF
        self._update_flags(result)
        self.pc += 1
    
    def _execute_and(self, instruction):
        """AND - Bitwise AND"""
        value = self._get_operand()
        result = self.registers[0] & value
        self.registers[0] = result
        self._update_flags(result)
        self.pc += 1
    
    def _execute_or(self, instruction):
        """OR - Bitwise OR"""
        value = self._get_operand()
        result = self.registers[0] | value
        self.registers[0] = result
        self._update_flags(result)
        self.pc += 1
    
    def _execute_xor(self, instruction):
        """XOR - Bitwise XOR"""
        value = self._get_operand()
        result = self.registers[0] ^ value
        self.registers[0] = result
        self._update_flags(result)
        self.pc += 1
    
    def _execute_not(self, instruction):
        """NOT - Bitwise NOT"""
        result = (~self.registers[0]) & 0xFF
        self.registers[0] = result
        self._update_flags(result)
        self.pc += 1
    
    def _execute_printc(self, instruction):
        """PRINTC - Print register A as character"""
        char = chr(self.registers[0] & 0xFF)
        self.output_handler(char, end='')
        self.stats.io_operations += 1
        self.pc += 1
    
    def _execute_prints(self, instruction):
        """PRINTS - Print string from memory"""
        address = self._get_operand()
        string_chars = []
        while address < len(self.memory) and self.memory[address] != 0:
            string_chars.append(chr(self.memory[address]))
            address += 1
        self.output_handler(''.join(string_chars))
        self.stats.io_operations += 1
        self.pc += 1
    
    # Debug and introspection methods
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
        if not enabled:
            self.instruction_trace.clear()
    
    def add_breakpoint(self, address: int):
        """Add breakpoint at address"""
        self.breakpoints.add(address)
    
    def remove_breakpoint(self, address: int):
        """Remove breakpoint at address"""
        self.breakpoints.discard(address)
    
    def get_state_info(self) -> dict:
        """Get comprehensive VM state information"""
        return {
            'state': self.state.value,
            'pc': self.pc,
            'sp': self.sp,
            'registers': self.registers.copy(),
            'flags': self.flags.copy(),
            'memory_size': self.memory_size,
            'program_size': self.program_size,
            'stats': {
                'instructions_executed': self.stats.instructions_executed,
                'cycles_elapsed': self.stats.cycles_elapsed,
                'execution_time': self.stats.execution_time,
                'memory_accesses': self.stats.memory_accesses,
                'io_operations': self.stats.io_operations
            }
        }
    
    def disassemble_program(self) -> List[str]:
        """Disassemble loaded program"""
        if not self.program:
            return []
        
        disassembly = []
        offset = 0
        
        while offset < len(self.program):
            instruction_text, next_offset = self.instruction_set.disassemble_instruction(
                self.program, offset
            )
            
            if instruction_text:
                disassembly.append(f"{offset:04X}: {instruction_text}")
            
            offset = next_offset if next_offset > offset else offset + 1
        
        return disassembly


# Convenience function for quick VM creation
def create_vm(memory_size: int = 256, registers: int = 4) -> DNAVirtualMachine:
    """Create a new DNA Virtual Machine instance"""
    return DNAVirtualMachine(memory_size, registers) 