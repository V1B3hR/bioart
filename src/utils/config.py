"""
Configuration management utilities for BioArt learning process.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(config: Dict[str, Any], config_type: str = "general") -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        config_type: Type of configuration to validate
        
    Returns:
        True if valid, raises exception if invalid
    """
    if config_type == "data":
        required_sections = ['datasets', 'feature_extraction', 'processing']
    elif config_type == "model":
        required_sections = ['dna_embedding', 'diffusion_lora']
    elif config_type == "art":
        required_sections = ['palettes', 'composition', 'output']
    else:
        # General validation
        return True
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return True


def get_default_data_config() -> Dict[str, Any]:
    """Get default data configuration."""
    return {
        'datasets': {},
        'feature_extraction': {
            'dna_sequences': {'kmer_size': [3, 4, 5, 6]},
            'microbe_images': {'image_size': [224, 224]},
            'haplogroup_metadata': {'categorical_encoding': 'one_hot'}
        },
        'processing': {
            'chunk_size': 1000,
            'n_jobs': -1,
            'random_seed': 42
        }
    }