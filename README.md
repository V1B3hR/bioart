# Bioart Programming Language

A research-driven programming and storage system that uses biological DNA (2‑bit nucleotide encoding: A=00, U=01, C=10, G=11) as a reversible medium for representing, executing, and analyzing computational processes.

---

## 🧬 Overview

Bioart demonstrates how nucleic acid sequences can act as a **complete, lossless, and reversible digital/programmatic substrate**. It includes:

- A virtual machine with a 100+ instruction set
- Deterministic DNA↔binary transforms
- Biological error modeling + multi‑layer error correction
- Real-world synthesis workflow abstractions
- Advanced algorithms (graph, ML, signal, matrix, floating point)
- Monitoring, automation, and optimization components

---

## ✅ Key Features

- **Maximum Encoding Efficiency**: Optimal 2‑bit nucleotide packing (4 nt/byte theoretical)
- **Universal File Compatibility**: Any binary or text payload
- **Perfect Reversibility**: Round‑trip integrity for all 256 byte values
- **Virtual Machine**: 106+ biological-computing instructions
- **High Throughput**: Up to 78M bytes/second (reported)
- **Error Correction Stack**: Hamming + redundancy + modeling + (optional) quantum layer
- **Synthesis Integration**: Multi-platform abstraction and cost heuristics
- **Intelligent Optimization**: ML-based sequence scoring & enhancement
- **Workflow & Monitoring**: Orchestrated pipelines + real-time synthesis dashboards

---

## 🗂 Repository Structure (High-Level)

```
bioart/
├── src/
│   └── bioart.py                 # Core interpreter / VM
├── examples/
│   ├── dna_demo.py               # Interactive demonstration
│   └── program.dna               # Example compiled DNA program
├── tests/
│   ├── advanced_tests.py
│   └── stress_tests.py
├── docs/
│   ├── readme.txt                # Extended technical documentation
│   ├── comprehensive_test_summary.txt
│   ├── AI_POC_VALIDATION_GUIDE.md
│   ├── automated_testing_guide.md
│   ├── ROADMAP.md
│   ├── ARCHITECTURE.md
│   ├── INSTRUCTION_SET.md
│   └── FORMAT_SPEC.md
├── run_tests.py
├── run_full_simulation.py
├── Makefile
└── README.md (this file)
```

---

## 🔧 Requirements

| Category | Requirement | Notes |
|----------|-------------|-------|
| Core Runtime | Python 3.8+ | Original text stated 3.6+; 3.8+ recommended for modern typing & performance |
| External Dependencies (Core) | None (as described) | Core VM & encoding appear standard-library only |
| Optional: ML Optimization | `numpy`, `scikit-learn` (potential) | Required if ML sequence optimization module uses vectorization/classifiers |
| Optional: Quantum Features | `qiskit` or similar (potential) | For quantum ECC demonstrations |
| Optional: Workflow / API | `requests`, `pydantic`, `rich` (potential) | For integrations, formatting & validation |
| Dev / Testing | `pytest`, `coverage` (suggested) | Not explicitly listed; recommended for CI |
| Packaging (future) | `build`, `setuptools` / `hatchling` | If distribution is desired |

If `requirements.txt` is introduced, consider splitting into:
- `requirements.txt` (strict minimal)
- `requirements-optional.txt`
- `requirements-dev.txt`

---

## 🚀 Installation

```bash
git clone <repository-url>
cd bioart
# (Optional) Create a virtual environment
python -m venv .venv && source .venv/bin/activate
# (Optional) Install dev/optional dependencies once enumerated
# pip install -r requirements.txt
```

---

## ⚡ Quick Start

Run the demonstration:

```bash
python examples/dna_demo.py
```

Run the interpreter:

```bash
python src/bioart.py
```

Full system simulation:

```bash
python run_full_simulation.py
# or
make all
```

---

## 🧪 Testing & Validation

Comprehensive test coverage (reported):
- Total Test Categories: 24
- Success Rate: 100%
- Data Processed: >500 KB end‑to‑end
- Byte Coverage: All 256 values
- Accuracy: 100% round‑trip DNA↔binary

Execute tests:

```bash
python run_tests.py
# or
make test
```

Targeted suites:

```bash
# Advanced
python tests/advanced_tests.py
make advanced

# Stress
python tests/stress_tests.py
make stress

# Example program
make example
```

---

## 🧬 DNA Encoding

| Base | Bits |
|------|------|
| A | 00 |
| U | 01 |
| C | 10 |
| G | 11 |

Examples:

```
Text "Hi!" → DNA: UACAUCCUACAU → Text "Hi!"
Binary [72,101,108] → DNA: UACAUCUUUCGAUC → [72,101,108]
```

---

## 💻 Virtual Machine & Instruction Set

Partial excerpt (see `docs/INSTRUCTION_SET.md` for the full specification):

| DNA | Binary | Instruction | Description |
|-----|--------|-------------|-------------|
| AAAA | 00000000 | NOP | No operation |
| AAAU | 00000001 | LOAD | Load value |
| AAAC | 00000010 | STORE | Store value |
| AAAG | 00000011 | ADD | Add |
| AAUA | 00000100 | SUB | Subtract |
| AAUU | 00000101 | MUL | Multiply |
| AAUC | 00000110 | DIV | Divide |
| AAUG | 00000111 | PRINT | Print output |
| AAGA | 00001100 | HALT | Halt |

Example:

```dna
AAAU AACA AAAG AAAC AAUG AAGA
```
Loads 42, adds 8, prints 50, halts.

---

## 🧪 Example DNA Program

```
DNA Program: AAAU ACCC AAAG AACA AAUG AAGA
Instructions: Load 42, Add 8, Print 50, Halt
Expected Output: 50
```

Run:

```bash
make example
```

---

## 📊 Performance Metrics (Reported)

- Processing Throughput: Up to 78M bytes/sec
- Storage Efficiency: 4.0 nucleotides/byte (optimal)
- Accuracy: 100% (all scenarios tested)
- Universal File Compatibility: Yes

---

## 🔬 Technical Specifications

### Virtual Machine
- Memory: 256 bytes
- Registers: A, B, C, D
- Instruction Set: 106+ ops (arithmetic, ML, graph, signal, matrix, floating point, string)
- File Format: `.dna` binary containers
- Error Handling: Hamming + redundancy + contextual biological checks
- Synthesis Integration: Platform abstraction, constraints, quality control

### Compiler / Tooling
- Source: DNA sequences
- Target: Bytecode
- Reversible: Full decompilation supported
- Disassembler: Human-readable linear form

---

## 🧠 Advanced Capabilities

- Environmental error modeling (UV, oxidative, thermal)
- ML-based sequence optimization
- Quantum error correction (e.g., Steane code)
- Workflow automation abstractions
- Real-time synthesis job monitoring
- Cost and fidelity optimization heuristics
- Complex algorithmic execution within the VM

---

## 📚 Documentation Hub

Core:
- [Project Roadmap](docs/ROADMAP.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Instruction Set Specification](docs/INSTRUCTION_SET.md)
- [Format Specification](docs/FORMAT_SPEC.md)

Development:
- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md) (if present)

Technical:
- `docs/readme.txt`
- `docs/comprehensive_test_summary.txt`
- `docs/automated_testing_guide.md`
- `docs/AI_POC_VALIDATION_GUIDE.md`

---

## 🤝 Contributing

Valuable contribution areas:
- Additional instruction implementations
- Performance & memory optimization
- Extended error correction / biological modeling
- Formal spec hardening (validation suites)
- More granular benchmarking metrics
- Documentation & tutorial expansion
- Packaging & distribution workflow

Suggested developer workflow (after adding dependencies file and packaging):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

---

## 🧭 Next Steps (Suggested Enhancements)

| Area | Proposal | Rationale |
|------|----------|-----------|
| Dependency Formalization | Publish `requirements*.txt` or `pyproject.toml` | Reproducibility & onboarding |
| Continuous Integration | Add CI (format, lint, test, coverage) | Ensures long-term integrity |
| Benchmark Suite | Structured micro & macro benchmarks | Track performance regressions |
| VM Extensibility | Plugin architecture for new instruction families | Decouple core evolution |
| Formal Spec Tests | Generate spec-conformance fixtures | Guard against semantic drift |
| Documentation Site | Convert docs to a static site (MkDocs / Sphinx) | Improved discoverability |
| Packaging | PyPI pre-release (alpha) | Wider adoption & feedback loop |
| API Layer | REST / gRPC interface for remote compilation & execution | Service integration |
| Interactive Playground | Web-based DNA↔binary + VM stepper | Educational impact |
| Provenance / Reproducibility | Embed metadata (hashes, environment) in `.dna` headers | Scientific auditability |
| Security Review | Threat model & sandboxing assessment | Safe execution of untrusted DNA code |
| Advanced ECC Analytics | Visual diffing + probability heatmaps | Research-grade insight |
| Hardware / FPGA Stub | Outline mapping for hardware acceleration | Future performance path |

---

## 🎯 Status

**Production-Ready Research Prototype**  
All foundational functionality is implemented and validated (v1.0).

---

## 📄 License

Distributed under the **GNU GPLv3**. See `LICENSE` for details.  
Intended for educational and research exploration of DNA-based computation.

---

## ℹ Version

Version: 1.0  
Status: Fully Functional Proof of Concept  
Last Updated: 2024 (README revision pending future tagged release alignment)

---
