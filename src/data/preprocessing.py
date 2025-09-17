"""
Data preprocessing for BioArt learning process.

Processes genomic sequences, microbe images, and metadata for artistic generation.
All processing is for computational/creative purposes only - no biological function.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import itertools

try:
    from Bio import SeqIO
    from Bio.SeqUtils import gc_fraction
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Some sequence analysis features disabled.")

try:
    import cv2
    from PIL import Image
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV/PIL not available. Image processing features disabled.")


class SequenceProcessor:
    """Processes DNA sequences for artistic feature extraction."""
    
    def __init__(self, kmer_sizes: List[int] = [3, 4, 5, 6]):
        """
        Initialize sequence processor.
        
        Args:
            kmer_sizes: List of k-mer sizes to analyze
        """
        self.kmer_sizes = kmer_sizes
        self.nucleotides = ['A', 'T', 'G', 'C']
        
    def extract_dna_features(self, sequences: Union[str, List[str], Path]) -> Dict:
        """
        Extract artistic features from DNA sequences.
        
        Args:
            sequences: Single sequence, list of sequences, or path to FASTA file
            
        Returns:
            Dictionary of extracted features for artistic generation
        """
        if isinstance(sequences, (str, Path)):
            sequences = self._load_sequences(sequences)
        elif isinstance(sequences, str) and not Path(sequences).exists():
            # Single sequence string
            sequences = [sequences]
        
        features = {
            'gc_content_distribution': [],
            'sequence_lengths': [],
            'kmer_spectra': {k: Counter() for k in self.kmer_sizes},
            'nucleotide_composition': Counter(),
            'complexity_scores': [],
            'metadata': {
                'total_sequences': len(sequences),
                'processing_method': 'artistic_only'
            }
        }
        
        for seq in sequences:
            seq = seq.upper().replace('U', 'T')  # Normalize to DNA
            
            # Basic composition features
            if BIOPYTHON_AVAILABLE:
                gc_content = gc_fraction(seq)
            else:
                gc_content = self._calculate_gc_content(seq)
            
            features['gc_content_distribution'].append(gc_content)
            features['sequence_lengths'].append(len(seq))
            
            # Nucleotide composition
            seq_composition = Counter(seq)
            features['nucleotide_composition'].update(seq_composition)
            
            # K-mer analysis
            for k in self.kmer_sizes:
                kmers = self._extract_kmers(seq, k)
                features['kmer_spectra'][k].update(kmers)
            
            # Complexity score (Shannon entropy)
            complexity = self._calculate_complexity(seq)
            features['complexity_scores'].append(complexity)
        
        # Convert to arrays for easier processing
        features['gc_content_distribution'] = np.array(features['gc_content_distribution'])
        features['sequence_lengths'] = np.array(features['sequence_lengths'])
        features['complexity_scores'] = np.array(features['complexity_scores'])
        
        return features
    
    def _load_sequences(self, file_path: Union[str, Path]) -> List[str]:
        """Load sequences from FASTA file."""
        file_path = Path(file_path)
        sequences = []
        
        if BIOPYTHON_AVAILABLE and file_path.suffix.lower() in ['.fasta', '.fa', '.fas']:
            for record in SeqIO.parse(file_path, 'fasta'):
                sequences.append(str(record.seq))
        else:
            # Simple text file reading
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('>'):
                    # Simple FASTA parsing
                    sequences = [line for line in content.split('\n') 
                               if not line.startswith('>') and line.strip()]
                else:
                    sequences = [content]
        
        return sequences
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content manually."""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0
    
    def _extract_kmers(self, sequence: str, k: int) -> List[str]:
        """Extract k-mers from sequence."""
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity using Shannon entropy."""
        if len(sequence) == 0:
            return 0.0
        
        # Count nucleotides
        counts = Counter(sequence)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in counts.values():
            freq = count / len(sequence)
            if freq > 0:
                entropy -= freq * np.log2(freq)
        
        return entropy
    
    def create_kmer_vocabulary(self, k: int) -> List[str]:
        """Create vocabulary of all possible k-mers."""
        return [''.join(kmer) for kmer in itertools.product(self.nucleotides, repeat=k)]


class ImageProcessor:
    """Processes microbe images for texture and visual feature extraction."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target size for image resizing
        """
        self.target_size = target_size
        
    def extract_image_features(self, image_path: Union[str, Path]) -> Dict:
        """
        Extract visual features from microbe images.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary of visual features for artistic generation
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV and PIL required for image processing")
        
        image_path = Path(image_path)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.target_size)
        
        features = {
            'color_histogram': self._extract_color_histogram(image_resized),
            'texture_features': self._extract_texture_features(image_resized),
            'shape_features': self._extract_shape_features(image_resized),
            'dominant_colors': self._extract_dominant_colors(image_resized),
            'image_stats': {
                'mean_brightness': np.mean(image_resized),
                'contrast': np.std(image_resized),
                'size': self.target_size
            }
        }
        
        return features
    
    def _extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> Dict:
        """Extract color histograms in different color spaces."""
        # RGB histogram
        rgb_hist = {
            'r': cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten(),
            'g': cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten(),
            'b': cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
        }
        
        # HSV histogram
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_hist = {
            'h': cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten(),
            's': cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten(),
            'v': cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
        }
        
        return {'rgb': rgb_hist, 'hsv': hsv_hist}
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """Extract basic texture features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern approximation
        texture_energy = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'texture_energy': texture_energy,
            'edge_density': edge_density,
            'gradient_magnitude': np.mean(np.gradient(gray.astype(float)))
        }
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict:
        """Extract basic shape features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            return {
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'num_contours': len(contours)
            }
        else:
            return {
                'area': 0,
                'perimeter': 0,
                'circularity': 0,
                'num_contours': 0
            }
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means clustering."""
        # Reshape image to pixel array
        pixels = image.reshape(-1, 3)
        
        # Simple clustering approximation (without sklearn)
        # Just return most frequent colors
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by frequency and return top k
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = unique_colors[sorted_indices[:k]]
        
        return [tuple(color) for color in dominant_colors]


class MetadataProcessor:
    """Processes haplogroup and other metadata for compositional rules."""
    
    def __init__(self):
        """Initialize metadata processor."""
        pass
    
    def process_haplogroup_data(self, data_path: Union[str, Path]) -> Dict:
        """
        Process Y-DNA haplogroup data for artistic composition rules.
        
        Args:
            data_path: Path to haplogroup CSV file
            
        Returns:
            Processed metadata for artistic generation
        """
        data_path = Path(data_path)
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Extract features for artistic mapping
        features = {
            'haplogroup_distribution': df.groupby('haplogroup').size().to_dict(),
            'ethnic_group_mapping': df.groupby('ethnic_group')['haplogroup'].apply(list).to_dict(),
            'diversity_metrics': self._calculate_diversity_metrics(df),
            'categorical_features': self._extract_categorical_features(df)
        }
        
        return features
    
    def _calculate_diversity_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate diversity metrics for artistic variation."""
        # Shannon diversity of haplogroups
        haplogroup_counts = df['haplogroup'].value_counts()
        total = len(df)
        
        shannon_diversity = -sum((count/total) * np.log2(count/total) 
                               for count in haplogroup_counts)
        
        return {
            'shannon_diversity': shannon_diversity,
            'num_unique_haplogroups': len(haplogroup_counts),
            'num_ethnic_groups': df['ethnic_group'].nunique()
        }
    
    def _extract_categorical_features(self, df: pd.DataFrame) -> Dict:
        """Extract categorical features for one-hot encoding."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        features = {}
        for col in categorical_cols:
            features[col] = {
                'unique_values': df[col].unique().tolist(),
                'value_counts': df[col].value_counts().to_dict()
            }
        
        return features