"""
BioArt composition and generation orchestration.

Combines DNA features, microbe textures, and metadata for multi-modal art creation.
All generation is for creative/educational purposes only.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from collections import Counter
import json

from .palettes import DNAPaletteGenerator
from .prompts import BiologicalPromptGenerator


class BioArtComposer:
    """Orchestrates multi-modal bioart composition."""
    
    def __init__(self):
        """Initialize bioart composer."""
        self.palette_generator = DNAPaletteGenerator()
        self.prompt_generator = BiologicalPromptGenerator()
        
        # Composition rules
        self.composition_rules = {
            'grid_layouts': [(2, 2), (3, 2), (2, 3), (3, 3)],
            'symmetry_types': ['radial', 'bilateral', 'none', 'rotational'],
            'blend_modes': ['overlay', 'multiply', 'screen', 'soft_light']
        }
        
    def create_composition_plan(
        self,
        biological_data: Dict[str, Any],
        artistic_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a plan for bioart composition based on biological data.
        
        Args:
            biological_data: Dictionary containing biological features
            artistic_parameters: Optional artistic style parameters
            
        Returns:
            Composition plan dictionary
        """
        if artistic_parameters is None:
            artistic_parameters = {}
        
        plan = {
            'metadata': {
                'data_sources': list(biological_data.keys()),
                'artistic_intent': 'computational_bioart',
                'generation_timestamp': None,  # Would be set during generation
                'ethics_note': 'Educational/artistic use only - no biological function'
            },
            'palette_plan': {},
            'prompt_plan': {},
            'layout_plan': {},
            'generation_parameters': {}
        }
        
        # Generate palette plan
        if 'dna_features' in biological_data:
            dna_data = biological_data['dna_features']
            gc_content = dna_data.get('gc_content_mean', 0.5)
            kmer_counts = dna_data.get('kmer_counts', Counter())
            
            plan['palette_plan'] = {
                'primary_palette': 'gc_content_based',
                'gc_content': gc_content,
                'kmer_diversity': len(kmer_counts),
                'color_scheme': self._determine_color_scheme(gc_content),
                'palette_size': artistic_parameters.get('palette_size', 8)
            }
        
        # Generate prompt plan
        prompt_data = {}
        if 'dna_features' in biological_data:
            prompt_data['dna'] = True
            prompt_data['gc_content'] = biological_data['dna_features'].get('gc_content_mean')
            prompt_data['kmer_counts'] = biological_data['dna_features'].get('kmer_counts')
        
        if 'haplogroup_data' in biological_data:
            prompt_data['haplogroup'] = biological_data['haplogroup_data'].get('primary_haplogroup')
        
        if 'microbe_features' in biological_data:
            prompt_data['microbe_features'] = biological_data['microbe_features']
        
        plan['prompt_plan'] = {
            'base_style': artistic_parameters.get('style', 'scientific'),
            'prompt_data': prompt_data,
            'num_variations': artistic_parameters.get('prompt_variations', 3)
        }
        
        # Generate layout plan
        plan['layout_plan'] = self._create_layout_plan(biological_data, artistic_parameters)
        
        # Generation parameters
        plan['generation_parameters'] = {
            'output_resolution': artistic_parameters.get('resolution', [1024, 1024]),
            'batch_size': artistic_parameters.get('batch_size', 4),
            'quality_level': artistic_parameters.get('quality', 'medium'),
            'include_metadata': True
        }
        
        return plan
    
    def _determine_color_scheme(self, gc_content: float) -> str:
        """Determine color scheme based on GC content."""
        if gc_content < 0.4:
            return 'warm_blue'  # AT-rich -> blue tones
        elif gc_content > 0.6:
            return 'warm_red'   # GC-rich -> red tones
        else:
            return 'balanced_green'  # Balanced -> green tones
    
    def _create_layout_plan(
        self, 
        biological_data: Dict[str, Any], 
        artistic_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create layout plan based on biological data."""
        layout_plan = {
            'composition_type': 'grid',
            'grid_size': (2, 2),
            'symmetry': 'none',
            'blend_mode': 'overlay'
        }
        
        # Determine grid size based on data complexity
        data_complexity = len(biological_data)
        if data_complexity >= 4:
            layout_plan['grid_size'] = (3, 3)
        elif data_complexity >= 3:
            layout_plan['grid_size'] = (3, 2)
        else:
            layout_plan['grid_size'] = (2, 2)
        
        # Determine symmetry from haplogroup data
        if 'haplogroup_data' in biological_data:
            haplogroup = biological_data['haplogroup_data'].get('primary_haplogroup', 'default')
            symmetry_map = {
                'R1a': 'radial',
                'R1b': 'bilateral',
                'I1': 'rotational',
                'I2': 'bilateral',
                'J1': 'radial',
                'J2': 'bilateral'
            }
            layout_plan['symmetry'] = symmetry_map.get(haplogroup, 'none')
        
        # Override with artistic parameters
        if 'layout_style' in artistic_parameters:
            layout_plan.update(artistic_parameters['layout_style'])
        
        return layout_plan
    
    def generate_art_specification(self, composition_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed art specification from composition plan.
        
        Args:
            composition_plan: Plan created by create_composition_plan
            
        Returns:
            Detailed specification for art generation
        """
        spec = {
            'palettes': {},
            'prompts': [],
            'layout': composition_plan['layout_plan'],
            'metadata': composition_plan['metadata'],
            'generation_ready': True
        }
        
        # Generate palettes
        palette_plan = composition_plan['palette_plan']
        if 'gc_content' in palette_plan:
            gc_content = palette_plan['gc_content']
            kmer_counts = composition_plan['prompt_plan']['prompt_data'].get('kmer_counts', Counter())
            haplogroup = composition_plan['prompt_plan']['prompt_data'].get('haplogroup')
            
            palettes = self.palette_generator.generate_comprehensive_palette(
                gc_content=gc_content,
                kmer_counts=kmer_counts,
                haplogroup=haplogroup,
                num_colors=palette_plan['palette_size']
            )
            spec['palettes'] = palettes
        
        # Generate prompts
        prompt_plan = composition_plan['prompt_plan']
        prompt_data = prompt_plan['prompt_data']
        
        base_prompt = self.prompt_generator.generate_comprehensive_prompt(
            gc_content=prompt_data.get('gc_content'),
            kmer_counts=prompt_data.get('kmer_counts'),
            haplogroup=prompt_data.get('haplogroup'),
            microbe_features=prompt_data.get('microbe_features'),
            style=prompt_plan['base_style']
        )
        
        # Create prompt variations
        prompt_variations = self.prompt_generator.create_prompt_variations(
            base_prompt, 
            prompt_plan['num_variations']
        )
        spec['prompts'] = prompt_variations
        
        return spec
    
    def simulate_generation(self, art_specification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate art generation process (preview mode).
        
        Args:
            art_specification: Specification from generate_art_specification
            
        Returns:
            Simulation results
        """
        print("🎨 Starting bioart generation simulation...")
        
        # Simulate palette application
        palettes = art_specification['palettes']
        print(f"   Applying {len(palettes)} palette types")
        
        # Simulate prompt processing
        prompts = art_specification['prompts']
        print(f"   Processing {len(prompts)} prompt variations")
        
        # Simulate layout composition
        layout = art_specification['layout']
        grid_size = layout['grid_size']
        print(f"   Composing {grid_size[0]}x{grid_size[1]} grid layout")
        
        # Simulate generation results
        results = {
            'status': 'simulation_complete',
            'generated_pieces': [],
            'processing_summary': {
                'palettes_applied': len(palettes),
                'prompts_processed': len(prompts),
                'layout_type': layout['composition_type'],
                'grid_dimensions': grid_size,
                'symmetry_applied': layout['symmetry']
            },
            'metadata': art_specification['metadata']
        }
        
        # Simulate individual art pieces
        for i, prompt in enumerate(prompts):
            piece = {
                'id': f'bioart_piece_{i+1:03d}',
                'prompt_used': prompt,
                'palette_type': 'combined',  # Would use actual palette
                'estimated_generation_time': '2-5 minutes',
                'output_format': 'PNG',
                'resolution': '1024x1024',
                'status': 'simulated'
            }
            results['generated_pieces'].append(piece)
        
        print("✅ Generation simulation completed")
        return results
    
    def create_bioart_series(
        self,
        biological_datasets: List[Dict[str, Any]],
        series_theme: str = "genomic_diversity",
        artistic_style: str = "scientific"
    ) -> Dict[str, Any]:
        """
        Create a series of related bioart pieces.
        
        Args:
            biological_datasets: List of biological data dictionaries
            series_theme: Theme for the series
            artistic_style: Overall artistic style
            
        Returns:
            Series specification and generation plan
        """
        print(f"🧬 Creating bioart series: '{series_theme}'")
        print(f"   Style: {artistic_style}")
        print(f"   Datasets: {len(biological_datasets)}")
        
        series_spec = {
            'theme': series_theme,
            'style': artistic_style,
            'pieces': [],
            'cohesion_elements': {
                'shared_palette_base': True,
                'consistent_style': True,
                'progressive_variation': True
            },
            'metadata': {
                'series_created': True,
                'num_pieces': len(biological_datasets),
                'educational_purpose': True
            }
        }
        
        # Process each dataset
        for i, dataset in enumerate(biological_datasets):
            print(f"   Processing dataset {i+1}/{len(biological_datasets)}")
            
            # Create composition plan with series consistency
            artistic_params = {
                'style': artistic_style,
                'series_index': i,
                'series_total': len(biological_datasets),
                'maintain_cohesion': True
            }
            
            composition_plan = self.create_composition_plan(dataset, artistic_params)
            art_spec = self.generate_art_specification(composition_plan)
            
            piece_info = {
                'index': i,
                'composition_plan': composition_plan,
                'art_specification': art_spec,
                'series_position': f"{i+1}/{len(biological_datasets)}"
            }
            
            series_spec['pieces'].append(piece_info)
        
        print("✅ Series specification completed")
        return series_spec
    
    def save_composition_plan(self, plan: Dict[str, Any], output_path: Path) -> None:
        """Save composition plan to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any Counter objects to regular dicts for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_plan = convert_for_json(plan)
        
        with open(output_path, 'w') as f:
            json.dump(json_plan, f, indent=2)
        
        print(f"💾 Composition plan saved to {output_path}")
    
    def load_composition_plan(self, input_path: Path) -> Dict[str, Any]:
        """Load composition plan from file."""
        with open(input_path, 'r') as f:
            plan = json.load(f)
        
        # Convert dict back to Counter where appropriate
        def restore_counters(obj):
            if isinstance(obj, dict):
                # Check if this looks like a counter (all values are integers)
                if all(isinstance(v, int) for v in obj.values()) and 'kmer' in str(obj):
                    return Counter(obj)
                else:
                    return {k: restore_counters(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_counters(item) for item in obj]
            else:
                return obj
        
        return restore_counters(plan)


def demo_bioart_composition():
    """Demonstrate bioart composition pipeline."""
    print("🎨 BioArt Composition Demo")
    print("=" * 30)
    
    composer = BioArtComposer()
    
    # Demo biological data
    demo_data = {
        'dna_features': {
            'gc_content_mean': 0.45,
            'sequence_lengths': [1000, 1200, 800],
            'kmer_counts': Counter({
                'ATCG': 15, 'GCTA': 12, 'TTAA': 20,
                'CCGG': 8, 'ATGG': 10, 'CGAT': 14
            }),
            'complexity_scores': [2.1, 1.8, 2.3]
        },
        'haplogroup_data': {
            'primary_haplogroup': 'R1a',
            'distribution': {'R1a': 60, 'R1b': 30, 'I1': 10}
        },
        'microbe_features': {
            'type': 'bacteria',
            'dominant_colors': [(120, 80, 200), (200, 150, 100), (80, 180, 120)],
            'texture_energy': 45.2,
            'circularity': 0.7
        }
    }
    
    artistic_params = {
        'style': 'scientific',
        'palette_size': 6,
        'resolution': [1024, 1024],
        'quality': 'high'
    }
    
    # Create composition plan
    print("\n📋 Creating composition plan...")
    composition_plan = composer.create_composition_plan(demo_data, artistic_params)
    
    print("   Data sources:", composition_plan['metadata']['data_sources'])
    print("   Color scheme:", composition_plan['palette_plan']['color_scheme'])
    print("   Layout:", composition_plan['layout_plan']['composition_type'])
    
    # Generate art specification
    print("\n🎯 Generating art specification...")
    art_spec = composer.generate_art_specification(composition_plan)
    
    print("   Palette types:", len(art_spec['palettes']))
    print("   Prompt variations:", len(art_spec['prompts']))
    print("   Grid layout:", art_spec['layout']['grid_size'])
    
    # Simulate generation
    print("\n🖼️  Simulating art generation...")
    results = composer.simulate_generation(art_spec)
    
    print("   Generated pieces:", len(results['generated_pieces']))
    print("   Palettes applied:", results['processing_summary']['palettes_applied'])
    
    # Show some example prompts
    print("\n📝 Example prompts:")
    for i, prompt in enumerate(art_spec['prompts'][:2]):
        print(f"   {i+1}: {prompt}")
    
    print("\n✅ Composition demo completed!")
    return composition_plan, art_spec, results


if __name__ == "__main__":
    demo_bioart_composition()