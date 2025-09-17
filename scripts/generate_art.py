#!/usr/bin/env python3
"""
BioArt Generation Script

Generates bioart from processed biological datasets using the BioArt pipeline.
This is the main entry point for creating computational bioart.

Usage:
    python scripts/generate_art.py [--config CONFIG_FILE] [--output OUTPUT_DIR]
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import SequenceProcessor, ImageProcessor, MetadataProcessor
from art.composition import BioArtComposer
from art.palettes import DNAPaletteGenerator
from art.prompts import BiologicalPromptGenerator


class BioArtGenerator:
    """Main class for generating bioart from biological data."""
    
    def __init__(self, config_path: str = "configs/art_config.yaml"):
        """
        Initialize bioart generator.
        
        Args:
            config_path: Path to art configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.composer = BioArtComposer()
        self.palette_generator = DNAPaletteGenerator()
        self.prompt_generator = BiologicalPromptGenerator()
        
        # Initialize processors
        self.sequence_processor = SequenceProcessor()
        self.image_processor = ImageProcessor()
        self.metadata_processor = MetadataProcessor()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load art generation configuration."""
        if not self.config_path.exists():
            print(f"⚠️  Config file not found: {self.config_path}")
            print("Using default configuration")
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file not found."""
        return {
            'palettes': {
                'gc_content_mapping': {
                    'low_gc': [0.2, 0.4],
                    'medium_gc': [0.4, 0.6],
                    'high_gc': [0.6, 0.8]
                }
            },
            'textures': {
                'microbe_based': {
                    'style_strength': 0.7
                }
            },
            'composition': {
                'layout': {
                    'grid_based': True
                }
            },
            'output': {
                'formats': ['PNG'],
                'resolutions': {
                    'medium': [1024, 1024]
                }
            },
            'generation': {
                'batch_size': 4,
                'variations_per_input': 3
            }
        }
    
    def load_biological_data(self, data_dir: Path) -> Dict[str, Any]:
        """
        Load and process biological data from directory.
        
        Args:
            data_dir: Directory containing processed biological data
            
        Returns:
            Dictionary of processed biological features
        """
        data_dir = Path(data_dir)
        biological_data = {}
        
        print(f"📂 Loading biological data from {data_dir}")
        
        # Load DNA sequence features
        dna_features_file = data_dir / "dna_features.json"
        if dna_features_file.exists():
            with open(dna_features_file, 'r') as f:
                dna_data = json.load(f)
                
            # Convert back to Counter for k-mer data
            if 'kmer_spectra' in dna_data:
                for k, counts in dna_data['kmer_spectra'].items():
                    dna_data['kmer_spectra'][k] = Counter(counts)
            
            biological_data['dna_features'] = dna_data
            print("   ✅ DNA sequence features loaded")
        
        # Load microbe image features
        microbe_features_file = data_dir / "microbe_features.json"
        if microbe_features_file.exists():
            with open(microbe_features_file, 'r') as f:
                biological_data['microbe_features'] = json.load(f)
            print("   ✅ Microbe image features loaded")
        
        # Load haplogroup metadata
        haplogroup_file = data_dir / "haplogroup_data.json"
        if haplogroup_file.exists():
            with open(haplogroup_file, 'r') as f:
                biological_data['haplogroup_data'] = json.load(f)
            print("   ✅ Haplogroup metadata loaded")
        
        if not biological_data:
            print("   ⚠️  No processed data files found")
            print("   Creating demo data for testing...")
            biological_data = self._create_demo_data()
        
        return biological_data
    
    def _create_demo_data(self) -> Dict[str, Any]:
        """Create demo biological data for testing."""
        return {
            'dna_features': {
                'gc_content_distribution': [0.45, 0.52, 0.38, 0.61],
                'sequence_lengths': [1000, 1200, 800, 1500],
                'kmer_spectra': {
                    '4': Counter({
                        'ATCG': 15, 'GCTA': 12, 'TTAA': 20, 'CCGG': 8,
                        'ATGG': 10, 'CGAT': 14, 'TGCA': 9, 'ACGT': 11
                    })
                },
                'complexity_scores': [2.1, 1.8, 2.3, 2.0],
                'metadata': {
                    'total_sequences': 4,
                    'processing_method': 'demo_data'
                }
            },
            'haplogroup_data': {
                'haplogroup_distribution': {
                    'R1a': 40, 'R1b': 35, 'I1': 15, 'J2': 10
                },
                'diversity_metrics': {
                    'shannon_diversity': 1.8,
                    'num_unique_haplogroups': 4
                }
            },
            'microbe_features': {
                'sample_images': 3,
                'dominant_colors_avg': [(120, 80, 200), (200, 150, 100), (80, 180, 120)],
                'texture_energy_avg': 45.2,
                'circularity_avg': 0.7
            }
        }
    
    def generate_single_piece(
        self, 
        biological_data: Dict[str, Any], 
        output_dir: Path,
        piece_name: str = "bioart_piece"
    ) -> Dict[str, Any]:
        """
        Generate a single bioart piece.
        
        Args:
            biological_data: Processed biological data
            output_dir: Output directory
            piece_name: Name for the generated piece
            
        Returns:
            Generation results
        """
        print(f"🎨 Generating bioart piece: {piece_name}")
        
        # Create composition plan
        artistic_params = {
            'style': 'scientific',
            'palette_size': self.config['generation'].get('palette_size', 8),
            'resolution': self.config['output']['resolutions']['medium'],
            'quality': 'medium'
        }
        
        composition_plan = self.composer.create_composition_plan(
            biological_data, artistic_params
        )
        
        # Generate art specification
        art_spec = self.composer.generate_art_specification(composition_plan)
        
        # Save composition plan
        plan_file = output_dir / f"{piece_name}_plan.json"
        self.composer.save_composition_plan(composition_plan, plan_file)
        
        # Save art specification
        spec_file = output_dir / f"{piece_name}_spec.json"
        with open(spec_file, 'w') as f:
            # Convert Counter objects for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, Counter):
                    return dict(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json_spec = convert_for_json(art_spec)
            json.dump(json_spec, f, indent=2)
        
        # Simulate generation (in full implementation, this would create actual images)
        results = self.composer.simulate_generation(art_spec)
        
        # Save results
        results_file = output_dir / f"{piece_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   📋 Composition plan: {plan_file}")
        print(f"   🎯 Art specification: {spec_file}")
        print(f"   📊 Results: {results_file}")
        print(f"   🖼️  Simulated {len(results['generated_pieces'])} art pieces")
        
        return results
    
    def generate_series(
        self,
        data_sources: List[Dict[str, Any]],
        output_dir: Path,
        series_name: str = "bioart_series",
        theme: str = "genomic_diversity"
    ) -> Dict[str, Any]:
        """
        Generate a series of related bioart pieces.
        
        Args:
            data_sources: List of biological data dictionaries
            output_dir: Output directory
            series_name: Name for the series
            theme: Series theme
            
        Returns:
            Series generation results
        """
        print(f"🧬 Generating bioart series: {series_name}")
        print(f"   Theme: {theme}")
        print(f"   Pieces: {len(data_sources)}")
        
        # Create series specification
        series_spec = self.composer.create_bioart_series(
            data_sources, theme, artistic_style='scientific'
        )
        
        # Save series specification
        series_file = output_dir / f"{series_name}_series.json"
        with open(series_file, 'w') as f:
            # Convert for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, Counter):
                    return dict(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json_series = convert_for_json(series_spec)
            json.dump(json_series, f, indent=2)
        
        # Generate individual pieces
        series_results = {
            'series_name': series_name,
            'theme': theme,
            'pieces': [],
            'series_file': str(series_file)
        }
        
        for i, piece_spec in enumerate(series_spec['pieces']):
            piece_name = f"{series_name}_piece_{i+1:03d}"
            
            # Simulate piece generation
            piece_results = self.composer.simulate_generation(
                piece_spec['art_specification']
            )
            
            piece_info = {
                'name': piece_name,
                'index': i,
                'results': piece_results
            }
            
            series_results['pieces'].append(piece_info)
        
        print(f"   📋 Series specification: {series_file}")
        print(f"   🖼️  Generated {len(series_results['pieces'])} pieces")
        
        return series_results
    
    def demonstrate_pipeline(self, output_dir: Path) -> Dict[str, Any]:
        """
        Demonstrate the complete bioart generation pipeline.
        
        Args:
            output_dir: Output directory for demonstration
            
        Returns:
            Demonstration results
        """
        print("🚀 BioArt Generation Pipeline Demonstration")
        print("=" * 50)
        
        # Create demo biological data
        demo_data = self._create_demo_data()
        
        # Generate single piece
        print("\n1️⃣  Generating single bioart piece...")
        single_results = self.generate_single_piece(
            demo_data, output_dir, "demo_single"
        )
        
        # Generate series (using variations of demo data)
        print("\n2️⃣  Generating bioart series...")
        
        # Create variations of demo data
        series_data = []
        for i in range(3):
            # Vary GC content and other parameters
            varied_data = demo_data.copy()
            varied_data['dna_features'] = demo_data['dna_features'].copy()
            
            # Modify GC content distribution
            base_gc = 0.5
            variation = 0.15 * (i - 1)  # -0.15, 0, +0.15
            varied_gc = [gc + variation for gc in demo_data['dna_features']['gc_content_distribution']]
            varied_gc = [max(0.1, min(0.9, gc)) for gc in varied_gc]  # Clamp to valid range
            
            varied_data['dna_features']['gc_content_distribution'] = varied_gc
            series_data.append(varied_data)
        
        series_results = self.generate_series(
            series_data, output_dir, "demo_series", "gc_content_variation"
        )
        
        # Generate comprehensive demo
        print("\n3️⃣  Generating comprehensive palette and prompt demo...")
        
        # Extract features for demonstration
        dna_features = demo_data['dna_features']
        gc_content_mean = sum(dna_features['gc_content_distribution']) / len(dna_features['gc_content_distribution'])
        kmer_counts = dna_features['kmer_spectra']['4']
        
        # Generate palettes
        palettes = self.palette_generator.generate_comprehensive_palette(
            gc_content=gc_content_mean,
            kmer_counts=kmer_counts,
            haplogroup='R1a',
            num_colors=8
        )
        
        # Generate prompts
        prompts = []
        for style in ['scientific', 'artistic', 'organic']:
            prompt = self.prompt_generator.generate_comprehensive_prompt(
                gc_content=gc_content_mean,
                kmer_counts=kmer_counts,
                haplogroup='R1a',
                style=style
            )
            prompts.append(prompt)
        
        # Save demonstration outputs
        demo_outputs = {
            'single_piece': single_results,
            'series': series_results,
            'palette_demo': {
                'gc_content_used': gc_content_mean,
                'palette_types': list(palettes.keys()),
                'palette_sizes': {k: len(v) for k, v in palettes.items()}
            },
            'prompt_demo': {
                'styles_generated': ['scientific', 'artistic', 'organic'],
                'example_prompts': prompts
            }
        }
        
        demo_file = output_dir / "pipeline_demonstration.json"
        with open(demo_file, 'w') as f:
            json.dump(demo_outputs, f, indent=2, default=str)
        
        print(f"\n📋 Complete demonstration saved to: {demo_file}")
        print("\n✅ Pipeline demonstration completed!")
        
        return demo_outputs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate bioart from biological datasets"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/art_config.yaml",
        help="Path to art configuration file"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data/processed",
        help="Directory containing processed biological data"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        help="Output directory for generated art"
    )
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run demonstration mode with sample data"
    )
    parser.add_argument(
        "--series",
        action="store_true",
        help="Generate art series instead of single piece"
    )
    parser.add_argument(
        "--name", "-n",
        default="bioart_generation",
        help="Name for generated art piece or series"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize generator
        generator = BioArtGenerator(args.config)
        
        if args.demo:
            # Run demonstration
            results = generator.demonstrate_pipeline(output_dir)
        else:
            # Load biological data
            data_dir = Path(args.data_dir)
            biological_data = generator.load_biological_data(data_dir)
            
            if args.series:
                # Generate series (using single dataset as base)
                series_data = [biological_data]  # In practice, load multiple datasets
                results = generator.generate_series(
                    series_data, output_dir, args.name
                )
            else:
                # Generate single piece
                results = generator.generate_single_piece(
                    biological_data, output_dir, args.name
                )
        
        print(f"\n🎉 Generation completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())