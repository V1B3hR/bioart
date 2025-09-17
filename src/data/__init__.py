"""
Data processing modules for BioArt learning process.

This package contains modules for:
- Downloading Kaggle datasets
- Preprocessing genomic and microbial data
- Extracting features for artistic generation

All processing focuses on computational/artistic applications only.
No biological function prediction or sequence design.
"""

from .preprocessing import SequenceProcessor, ImageProcessor, MetadataProcessor

__all__ = [
    'SequenceProcessor',
    'ImageProcessor', 
    'MetadataProcessor'
]