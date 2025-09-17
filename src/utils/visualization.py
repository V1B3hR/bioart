"""
Visualization utilities for BioArt learning process.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_gc_distribution(gc_values: List[float], title: str = "GC Content Distribution", save_path: Optional[str] = None):
    """
    Plot GC content distribution.
    
    Args:
        gc_values: List of GC content values
        title: Plot title
        save_path: Optional path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(gc_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(gc_values), color='red', linestyle='--', label=f'Mean: {np.mean(gc_values):.3f}')
    
    plt.xlabel('GC Content')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_kmer_spectrum(kmer_counts: Counter, k: int = 4, top_n: int = 20, save_path: Optional[str] = None):
    """
    Visualize k-mer frequency spectrum.
    
    Args:
        kmer_counts: Counter of k-mer frequencies
        k: K-mer size
        top_n: Number of top k-mers to show
        save_path: Optional path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return
    
    # Get top k-mers
    top_kmers = kmer_counts.most_common(top_n)
    
    if not top_kmers:
        print("No k-mer data to visualize")
        return
    
    kmers, counts = zip(*top_kmers)
    
    plt.figure(figsize=(12, 8))
    
    bars = plt.bar(range(len(kmers)), counts, color='lightcoral', alpha=0.7)
    plt.xlabel(f'{k}-mer')
    plt.ylabel('Frequency')
    plt.title(f'Top {len(kmers)} {k}-mer Frequencies')
    
    # Rotate x-axis labels
    plt.xticks(range(len(kmers)), kmers, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_palette_visualization(palette: List[tuple], title: str = "Color Palette", save_path: Optional[str] = None):
    """
    Visualize a color palette.
    
    Args:
        palette: List of RGB tuples
        title: Plot title
        save_path: Optional path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return
    
    # Convert RGB tuples to matplotlib format
    colors = [(r/255, g/255, b/255) for r, g, b in palette]
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    
    # Draw color swatches
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
        # Add RGB values as text
        ax.text(i + 0.5, 0.5, f'RGB\n{palette[i]}', ha='center', va='center', 
                fontsize=8, color='white' if sum(color) < 1.5 else 'black')
    
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()