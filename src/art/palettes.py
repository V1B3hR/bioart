"""
Color palette generation from DNA features for BioArt.

Maps biological sequence features to artistic color palettes.
All mappings are for creative/educational purposes only.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import colorsys

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Palette visualization disabled.")


class GCContentPalette:
    """Generate color palettes based on GC content."""
    
    def __init__(self):
        """Initialize GC content palette generator."""
        # Define color mappings for different GC content ranges
        self.gc_color_map = {
            'at_rich': {'hue': 240, 'saturation': 0.7, 'value': 0.9},     # Blue for AT-rich
            'balanced': {'hue': 120, 'saturation': 0.6, 'value': 0.8},    # Green for balanced
            'gc_rich': {'hue': 0, 'saturation': 0.8, 'value': 0.9}        # Red for GC-rich
        }
        
        self.gc_thresholds = {
            'low': 0.4,
            'high': 0.6
        }
    
    def generate_palette(self, gc_content: float, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Generate color palette based on GC content.
        
        Args:
            gc_content: GC content (0.0 to 1.0)
            num_colors: Number of colors in palette
            
        Returns:
            List of RGB tuples
        """
        # Determine base color based on GC content
        if gc_content < self.gc_thresholds['low']:
            base_hsv = self.gc_color_map['at_rich']
        elif gc_content > self.gc_thresholds['high']:
            base_hsv = self.gc_color_map['gc_rich']
        else:
            base_hsv = self.gc_color_map['balanced']
        
        # Generate palette variations
        palette = []
        base_hue = base_hsv['hue']
        base_sat = base_hsv['saturation']
        base_val = base_hsv['value']
        
        for i in range(num_colors):
            # Vary hue slightly
            hue_offset = (i - num_colors // 2) * 15  # ±15 degrees per color
            hue = (base_hue + hue_offset) % 360
            
            # Vary saturation and value slightly
            sat_variation = 0.1 * (i - num_colors // 2) / num_colors
            val_variation = 0.1 * (i - num_colors // 2) / num_colors
            
            saturation = max(0.2, min(1.0, base_sat + sat_variation))
            value = max(0.4, min(1.0, base_val + val_variation))
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            rgb_int = tuple(int(c * 255) for c in rgb)
            palette.append(rgb_int)
        
        return palette
    
    def interpolate_palette(self, gc_content1: float, gc_content2: float, steps: int = 10) -> List[List[Tuple[int, int, int]]]:
        """
        Create interpolated palettes between two GC contents.
        
        Args:
            gc_content1: Starting GC content
            gc_content2: Ending GC content
            steps: Number of interpolation steps
            
        Returns:
            List of palettes interpolating between the two GC contents
        """
        palettes = []
        
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            interpolated_gc = gc_content1 + t * (gc_content2 - gc_content1)
            palette = self.generate_palette(interpolated_gc)
            palettes.append(palette)
        
        return palettes


class KmerDiversityPalette:
    """Generate palettes based on k-mer diversity."""
    
    def __init__(self, k: int = 4):
        """
        Initialize k-mer diversity palette generator.
        
        Args:
            k: K-mer size
        """
        self.k = k
        self.max_possible_kmers = 4 ** k
    
    def calculate_diversity(self, kmer_counts: Counter) -> float:
        """Calculate k-mer diversity using Shannon entropy."""
        total_kmers = sum(kmer_counts.values())
        if total_kmers == 0:
            return 0.0
        
        entropy = 0.0
        for count in kmer_counts.values():
            if count > 0:
                freq = count / total_kmers
                entropy -= freq * np.log2(freq)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(min(self.max_possible_kmers, total_kmers))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def generate_palette(self, kmer_counts: Counter, base_hue: int = 180) -> List[Tuple[int, int, int]]:
        """
        Generate palette based on k-mer diversity.
        
        Args:
            kmer_counts: Counter of k-mer frequencies
            base_hue: Base hue for the palette
            
        Returns:
            List of RGB tuples
        """
        diversity = self.calculate_diversity(kmer_counts)
        
        # Map diversity to saturation (high diversity = high saturation)
        saturation = 0.3 + 0.7 * diversity
        
        # Generate palette with varying values
        palette = []
        num_colors = 7
        
        for i in range(num_colors):
            # Vary value based on position
            value = 0.4 + 0.6 * (i / (num_colors - 1))
            
            # Slight hue variation
            hue_offset = (i - num_colors // 2) * 10
            hue = (base_hue + hue_offset) % 360
            
            rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            rgb_int = tuple(int(c * 255) for c in rgb)
            palette.append(rgb_int)
        
        return palette


class HaplogroupPalette:
    """Generate palettes based on haplogroup metadata."""
    
    def __init__(self):
        """Initialize haplogroup palette generator."""
        # Define color associations for major haplogroups
        self.haplogroup_colors = {
            'R1a': {'hue': 240, 'base_name': 'blue'},      # Blue family
            'R1b': {'hue': 0, 'base_name': 'red'},         # Red family
            'I1': {'hue': 60, 'base_name': 'yellow'},      # Yellow family
            'I2': {'hue': 30, 'base_name': 'orange'},      # Orange family
            'J1': {'hue': 300, 'base_name': 'magenta'},    # Magenta family
            'J2': {'hue': 270, 'base_name': 'purple'},     # Purple family
            'E1b1b': {'hue': 120, 'base_name': 'green'},   # Green family
            'G2a': {'hue': 180, 'base_name': 'cyan'},      # Cyan family
            'T1a': {'hue': 200, 'base_name': 'light_blue'}, # Light blue family
            'default': {'hue': 160, 'base_name': 'teal'}   # Default for unmapped
        }
    
    def generate_palette(self, haplogroup: str, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Generate palette for specific haplogroup.
        
        Args:
            haplogroup: Haplogroup identifier
            num_colors: Number of colors to generate
            
        Returns:
            List of RGB tuples
        """
        # Get base color for haplogroup
        color_info = self.haplogroup_colors.get(haplogroup, self.haplogroup_colors['default'])
        base_hue = color_info['hue']
        
        palette = []
        
        for i in range(num_colors):
            # Create variations around base hue
            hue_variation = (i - num_colors // 2) * 20  # ±20 degrees variation
            hue = (base_hue + hue_variation) % 360
            
            # Vary saturation and value for richness
            saturation = 0.5 + 0.3 * np.sin(i * np.pi / num_colors)
            value = 0.6 + 0.4 * np.cos(i * np.pi / num_colors)
            
            # Ensure minimum saturation and value
            saturation = max(0.3, min(1.0, saturation))
            value = max(0.4, min(1.0, value))
            
            rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            rgb_int = tuple(int(c * 255) for c in rgb)
            palette.append(rgb_int)
        
        return palette
    
    def create_population_palette(self, haplogroup_distribution: Dict[str, int]) -> List[Tuple[int, int, int]]:
        """
        Create palette representing population haplogroup distribution.
        
        Args:
            haplogroup_distribution: Dictionary of haplogroup counts
            
        Returns:
            Weighted palette based on distribution
        """
        total_count = sum(haplogroup_distribution.values())
        palette = []
        
        for haplogroup, count in haplogroup_distribution.items():
            weight = count / total_count
            num_colors_for_group = max(1, int(weight * 10))  # Up to 10 colors total
            
            group_palette = self.generate_palette(haplogroup, num_colors_for_group)
            palette.extend(group_palette)
        
        # Limit total palette size
        if len(palette) > 12:
            # Sample evenly from the palette
            indices = np.linspace(0, len(palette) - 1, 12, dtype=int)
            palette = [palette[i] for i in indices]
        
        return palette


class DNAPaletteGenerator:
    """Main palette generator combining all DNA-based features."""
    
    def __init__(self):
        """Initialize DNA palette generator."""
        self.gc_palette = GCContentPalette()
        self.kmer_palette = KmerDiversityPalette()
        self.haplogroup_palette = HaplogroupPalette()
    
    def generate_comprehensive_palette(
        self,
        gc_content: float,
        kmer_counts: Counter,
        haplogroup: Optional[str] = None,
        num_colors: int = 8
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Generate comprehensive palette from all DNA features.
        
        Args:
            gc_content: GC content (0.0 to 1.0)
            kmer_counts: K-mer frequency counts
            haplogroup: Haplogroup identifier (optional)
            num_colors: Number of colors per palette type
            
        Returns:
            Dictionary of different palette types
        """
        palettes = {}
        
        # GC content palette
        palettes['gc_content'] = self.gc_palette.generate_palette(gc_content, num_colors)
        
        # K-mer diversity palette
        palettes['kmer_diversity'] = self.kmer_palette.generate_palette(kmer_counts)
        
        # Haplogroup palette (if provided)
        if haplogroup:
            palettes['haplogroup'] = self.haplogroup_palette.generate_palette(haplogroup, num_colors)
        
        # Combined palette (blend of all features)
        palettes['combined'] = self._create_combined_palette(palettes, num_colors)
        
        return palettes
    
    def _create_combined_palette(self, palettes: Dict[str, List[Tuple[int, int, int]]], num_colors: int) -> List[Tuple[int, int, int]]:
        """Create combined palette from multiple palette types."""
        all_colors = []
        
        # Collect all colors
        for palette_type, colors in palettes.items():
            if palette_type != 'combined':  # Avoid recursion
                all_colors.extend(colors)
        
        if not all_colors:
            # Fallback to default palette
            return [(128, 128, 128)] * num_colors
        
        # Sample evenly from all colors
        if len(all_colors) <= num_colors:
            combined = all_colors[:]
            # Fill remaining slots with variations
            while len(combined) < num_colors:
                base_color = all_colors[len(combined) % len(all_colors)]
                # Create slight variation
                varied_color = self._vary_color(base_color, 0.1)
                combined.append(varied_color)
        else:
            # Sample evenly
            indices = np.linspace(0, len(all_colors) - 1, num_colors, dtype=int)
            combined = [all_colors[i] for i in indices]
        
        return combined
    
    def _vary_color(self, rgb_color: Tuple[int, int, int], variation: float = 0.1) -> Tuple[int, int, int]:
        """Create a slight variation of a color."""
        r, g, b = rgb_color
        
        # Convert to HSV for easier manipulation
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        
        # Add small random variations
        h = (h + np.random.uniform(-variation, variation)) % 1.0
        s = max(0, min(1, s + np.random.uniform(-variation/2, variation/2)))
        v = max(0, min(1, v + np.random.uniform(-variation/2, variation/2)))
        
        # Convert back to RGB
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return tuple(int(c * 255) for c in rgb)
    
    def visualize_palette(self, palette: List[Tuple[int, int, int]], title: str = "DNA Palette") -> None:
        """
        Visualize a color palette.
        
        Args:
            palette: List of RGB tuples
            title: Title for the visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for palette visualization")
            return
        
        # Convert RGB tuples to matplotlib format
        colors = [(r/255, g/255, b/255) for r, g, b in palette]
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
        
        # Draw color swatches
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
            
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_palette(self, palette: List[Tuple[int, int, int]], filename: str) -> None:
        """
        Save palette to file.
        
        Args:
            palette: List of RGB tuples
            filename: Output filename
        """
        # Save as simple JSON format
        import json
        
        palette_data = {
            'colors': palette,
            'format': 'RGB',
            'generator': 'DNAPaletteGenerator'
        }
        
        with open(filename, 'w') as f:
            json.dump(palette_data, f, indent=2)
        
        print(f"Palette saved to {filename}")


def demo_palette_generation():
    """Demonstrate palette generation from DNA features."""
    print("🎨 DNA Palette Generation Demo")
    print("=" * 40)
    
    # Initialize generator
    generator = DNAPaletteGenerator()
    
    # Demo data
    gc_contents = [0.3, 0.5, 0.7]  # AT-rich, balanced, GC-rich
    kmer_counts = Counter({
        'ATCG': 10, 'GCTA': 8, 'TTAA': 15, 'CCGG': 5,
        'ATGG': 12, 'CGAT': 7, 'TGCA': 9, 'ACGT': 11
    })
    haplogroups = ['R1a', 'R1b', 'I1']
    
    # Generate palettes for different scenarios
    for i, gc in enumerate(gc_contents):
        print(f"\nScenario {i+1}: GC content = {gc:.1%}")
        
        haplogroup = haplogroups[i] if i < len(haplogroups) else None
        
        palettes = generator.generate_comprehensive_palette(
            gc_content=gc,
            kmer_counts=kmer_counts,
            haplogroup=haplogroup
        )
        
        for palette_type, colors in palettes.items():
            print(f"  {palette_type}: {len(colors)} colors")
            # Show first few colors
            for j, color in enumerate(colors[:3]):
                print(f"    Color {j+1}: RGB{color}")
            if len(colors) > 3:
                print(f"    ... and {len(colors) - 3} more")
    
    print("\n✅ Palette generation demo completed!")


if __name__ == "__main__":
    demo_palette_generation()