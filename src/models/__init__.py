"""
Model modules for BioArt learning process.

This package contains non-functional models for artistic generation:
- DNA sequence embeddings (k-mer based, no biological function)
- LoRA fine-tuning for diffusion models using microbe textures

All models are for computational/artistic purposes only.
"""

from .dna_embedding import DNAEmbedding, KmerTokenizer
from .diffusion_lora import MicrobeTextureLoRA

__all__ = [
    'DNAEmbedding',
    'KmerTokenizer', 
    'MicrobeTextureLoRA'
]