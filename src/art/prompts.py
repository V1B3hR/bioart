"""
Biological prompt generation for BioArt diffusion models.

Creates text prompts from biological data features for generative art.
All prompts are for creative/educational purposes only.
"""

import random
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter
import numpy as np


class BiologicalPromptGenerator:
    """Generate text prompts from biological data for art generation."""
    
    def __init__(self):
        """Initialize biological prompt generator."""
        self.base_templates = {
            'scientific': [
                "microscopic {organism} {feature}, scientific illustration style",
                "cellular structure of {organism}, {feature} patterns, detailed",
                "biological texture showing {feature}, {organism} morphology",
                "{organism} specimen with {feature}, laboratory photography style"
            ],
            'artistic': [
                "bioart interpretation of {organism}, {feature} aesthetic",
                "abstract {feature} patterns inspired by {organism}",
                "generative art based on {organism} {feature}",
                "artistic visualization of {feature} in {organism}"
            ],
            'organic': [
                "organic {feature} forms, {organism} inspired patterns",
                "natural {organism} textures, {feature} characteristics",
                "biological {feature} geometry, {organism} structure",
                "living {organism} patterns, {feature} composition"
            ]
        }
        
        self.style_modifiers = {
            'traditional': ["watercolor", "oil painting", "pencil drawing", "botanical illustration"],
            'digital': ["digital art", "3D render", "procedural", "algorithmic"],
            'photographic': ["macro photography", "microscopy", "high resolution", "detailed"],
            'abstract': ["abstract", "geometric", "minimalist", "conceptual"]
        }
        
        self.quality_enhancers = [
            "high quality", "detailed", "sharp focus", "professional",
            "8k resolution", "ultra detailed", "masterpiece", "award winning"
        ]
        
        # DNA feature descriptors
        self.gc_content_descriptors = {
            'low': ['AT-rich regions', 'adenine-thymine dominant', 'low complexity'],
            'medium': ['balanced composition', 'moderate complexity', 'mixed patterns'],
            'high': ['GC-rich regions', 'guanine-cytosine dominant', 'high complexity']
        }
        
        # Microbe type descriptors
        self.microbe_descriptors = {
            'bacteria': ['bacterial', 'prokaryotic', 'cellular', 'colony'],
            'fungi': ['fungal', 'mycelial', 'spore', 'hyphal'],
            'protist': ['protist', 'single-celled', 'motile', 'ciliated'],
            'virus': ['viral', 'crystalline', 'geometric', 'capsid']
        }
        
        # Haplogroup cultural associations (for artistic variation only)
        self.haplogroup_aesthetics = {
            'R1a': ['northern', 'crystalline', 'geometric', 'cold tones'],
            'R1b': ['western', 'flowing', 'organic', 'warm tones'],
            'I1': ['nordic', 'angular', 'structured', 'blue tones'],
            'I2': ['alpine', 'layered', 'textured', 'earth tones'],
            'J1': ['desert', 'sand-like', 'weathered', 'golden tones'],
            'J2': ['mediterranean', 'flowing', 'water-like', 'azure tones'],
            'E1b1b': ['sun-baked', 'radial', 'bright', 'amber tones'],
            'G2a': ['mountain', 'stratified', 'mineral', 'grey tones']
        }
    
    def generate_from_gc_content(
        self,
        gc_content: float,
        style: str = 'scientific',
        organism: str = 'microorganism'
    ) -> str:
        """
        Generate prompt based on GC content.
        
        Args:
            gc_content: GC content (0.0 to 1.0)
            style: Prompt style ('scientific', 'artistic', 'organic')
            organism: Organism type
            
        Returns:
            Generated text prompt
        """
        # Classify GC content
        if gc_content < 0.4:
            gc_category = 'low'
        elif gc_content > 0.6:
            gc_category = 'high'
        else:
            gc_category = 'medium'
        
        # Select descriptors
        feature_descriptors = self.gc_content_descriptors[gc_category]
        feature = random.choice(feature_descriptors)
        
        # Select template
        templates = self.base_templates.get(style, self.base_templates['scientific'])
        template = random.choice(templates)
        
        # Fill template
        prompt = template.format(organism=organism, feature=feature)
        
        # Add style modifier
        if random.random() < 0.7:  # 70% chance to add style modifier
            style_categories = list(self.style_modifiers.keys())
            style_category = random.choice(style_categories)
            modifier = random.choice(self.style_modifiers[style_category])
            prompt += f", {modifier}"
        
        # Add quality enhancer
        if random.random() < 0.5:  # 50% chance to add quality enhancer
            enhancer = random.choice(self.quality_enhancers)
            prompt += f", {enhancer}"
        
        return prompt
    
    def generate_from_kmer_diversity(
        self,
        kmer_counts: Counter,
        k: int = 4,
        organism: str = 'cellular structure'
    ) -> str:
        """
        Generate prompt based on k-mer diversity.
        
        Args:
            kmer_counts: K-mer frequency counts
            k: K-mer size
            organism: Organism type
            
        Returns:
            Generated text prompt
        """
        # Calculate diversity
        total_kmers = sum(kmer_counts.values())
        if total_kmers == 0:
            diversity = 0.0
        else:
            entropy = 0.0
            for count in kmer_counts.values():
                if count > 0:
                    freq = count / total_kmers
                    entropy -= freq * np.log2(freq)
            
            max_entropy = np.log2(min(4**k, total_kmers))
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Map diversity to descriptors
        if diversity < 0.3:
            complexity_desc = ['simple patterns', 'repetitive motifs', 'uniform texture']
        elif diversity > 0.7:
            complexity_desc = ['complex patterns', 'diverse motifs', 'intricate texture']
        else:
            complexity_desc = ['moderate patterns', 'varied motifs', 'balanced texture']
        
        feature = random.choice(complexity_desc)
        
        # Generate prompt
        templates = [
            f"biological {organism} showing {feature}",
            f"{organism} with {feature}, microscopic detail",
            f"organic {feature} in {organism} structure",
            f"{organism} displaying {feature} characteristics"
        ]
        
        base_prompt = random.choice(templates)
        
        # Add artistic elements based on diversity
        if diversity > 0.6:
            artistic_elements = ['fractal-like', 'highly detailed', 'complex geometry']
        elif diversity < 0.4:
            artistic_elements = ['minimalist', 'simple forms', 'clean lines']
        else:
            artistic_elements = ['balanced composition', 'moderate complexity', 'harmonious']
        
        element = random.choice(artistic_elements)
        final_prompt = f"{base_prompt}, {element}"
        
        # Add random style
        if random.random() < 0.6:
            style_desc = random.choice(['bioart style', 'scientific illustration', 'abstract art'])
            final_prompt += f", {style_desc}"
        
        return final_prompt
    
    def generate_from_haplogroup(
        self,
        haplogroup: str,
        base_organism: str = 'ancestral patterns'
    ) -> str:
        """
        Generate prompt incorporating haplogroup aesthetic associations.
        
        Args:
            haplogroup: Haplogroup identifier
            base_organism: Base organism description
            
        Returns:
            Generated text prompt
        """
        # Get aesthetic descriptors for haplogroup
        aesthetics = self.haplogroup_aesthetics.get(haplogroup, ['neutral', 'balanced', 'natural', 'earth tones'])
        
        # Select aesthetic elements
        selected_aesthetics = random.sample(aesthetics, min(2, len(aesthetics)))
        
        # Generate base prompt
        templates = [
            f"bioart inspired by {base_organism}, {', '.join(selected_aesthetics)}",
            f"generative patterns reflecting {base_organism}, {selected_aesthetics[0]} aesthetic",
            f"artistic interpretation of {base_organism} with {', '.join(selected_aesthetics)}",
            f"{base_organism} visualization, {selected_aesthetics[0]} style"
        ]
        
        prompt = random.choice(templates)
        
        # Add compositional elements based on haplogroup
        compositional_elements = {
            'R1a': ['geometric arrangement', 'crystalline structure', 'symmetric composition'],
            'R1b': ['organic flow', 'curved lines', 'natural arrangement'],
            'I1': ['angular patterns', 'sharp contrasts', 'structured layout'],
            'I2': ['layered composition', 'depth variation', 'textural contrast'],
            'J1': ['radial patterns', 'central focus', 'outward flow'],
            'J2': ['fluid dynamics', 'wave-like patterns', 'smooth transitions'],
            'E1b1b': ['solar motifs', 'bright highlights', 'energetic composition'],
            'G2a': ['mineral textures', 'stratified layers', 'geological forms']
        }
        
        if haplogroup in compositional_elements:
            element = random.choice(compositional_elements[haplogroup])
            prompt += f", {element}"
        
        return prompt
    
    def generate_from_microbe_image_features(
        self,
        dominant_colors: List[Tuple[int, int, int]],
        texture_energy: float,
        circularity: float,
        microbe_type: str = 'microorganism'
    ) -> str:
        """
        Generate prompt from microbe image analysis features.
        
        Args:
            dominant_colors: List of dominant RGB colors
            texture_energy: Texture energy measure
            circularity: Shape circularity measure
            microbe_type: Type of microorganism
            
        Returns:
            Generated text prompt
        """
        # Analyze colors
        color_descriptors = self._analyze_color_palette(dominant_colors)
        
        # Analyze texture
        if texture_energy > 100:
            texture_desc = ['highly textured', 'rough surface', 'complex texture']
        elif texture_energy < 20:
            texture_desc = ['smooth surface', 'minimal texture', 'clean appearance']
        else:
            texture_desc = ['moderate texture', 'balanced surface', 'subtle patterns']
        
        # Analyze shape
        if circularity > 0.8:
            shape_desc = ['circular forms', 'round morphology', 'spherical structure']
        elif circularity < 0.3:
            shape_desc = ['irregular shapes', 'complex morphology', 'asymmetric forms']
        else:
            shape_desc = ['oval forms', 'moderate asymmetry', 'balanced morphology']
        
        # Construct prompt
        color_desc = random.choice(color_descriptors)
        texture = random.choice(texture_desc)
        shape = random.choice(shape_desc)
        
        prompt_templates = [
            f"{microbe_type} with {color_desc}, {texture}, {shape}",
            f"microscopic {microbe_type} showing {texture} and {shape}, {color_desc}",
            f"biological specimen: {microbe_type} featuring {color_desc} and {texture}",
            f"{microbe_type} morphology: {shape} with {texture}, {color_desc}"
        ]
        
        base_prompt = random.choice(prompt_templates)
        
        # Add artistic style
        styles = ['scientific photography', 'bioart aesthetic', 'microscopy style', 'cellular art']
        style = random.choice(styles)
        
        final_prompt = f"{base_prompt}, {style}"
        
        return final_prompt
    
    def _analyze_color_palette(self, rgb_colors: List[Tuple[int, int, int]]) -> List[str]:
        """Analyze color palette and return descriptive terms."""
        if not rgb_colors:
            return ['neutral tones']
        
        # Convert to HSV for analysis
        hsv_colors = []
        for r, g, b in rgb_colors:
            import colorsys
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hsv_colors.append((h*360, s, v))
        
        # Analyze hue distribution
        hues = [h for h, s, v in hsv_colors]
        avg_hue = np.mean(hues)
        
        # Analyze saturation and value
        saturations = [s for h, s, v in hsv_colors]
        values = [v for h, s, v in hsv_colors]
        avg_saturation = np.mean(saturations)
        avg_value = np.mean(values)
        
        descriptors = []
        
        # Hue-based descriptors
        if avg_hue < 30 or avg_hue > 330:
            descriptors.append('red tones')
        elif 30 <= avg_hue < 90:
            descriptors.append('yellow-orange tones')
        elif 90 <= avg_hue < 150:
            descriptors.append('green tones')
        elif 150 <= avg_hue < 210:
            descriptors.append('cyan-blue tones')
        elif 210 <= avg_hue < 270:
            descriptors.append('blue tones')
        else:
            descriptors.append('purple-magenta tones')
        
        # Saturation-based descriptors
        if avg_saturation > 0.7:
            descriptors.append('vibrant colors')
        elif avg_saturation < 0.3:
            descriptors.append('muted colors')
        else:
            descriptors.append('moderate saturation')
        
        # Value-based descriptors
        if avg_value > 0.7:
            descriptors.append('bright appearance')
        elif avg_value < 0.3:
            descriptors.append('dark appearance')
        else:
            descriptors.append('medium brightness')
        
        return descriptors
    
    def generate_comprehensive_prompt(
        self,
        gc_content: Optional[float] = None,
        kmer_counts: Optional[Counter] = None,
        haplogroup: Optional[str] = None,
        microbe_features: Optional[Dict] = None,
        style: str = 'scientific'
    ) -> str:
        """
        Generate comprehensive prompt from multiple biological features.
        
        Args:
            gc_content: GC content (0.0 to 1.0)
            kmer_counts: K-mer frequency counts
            haplogroup: Haplogroup identifier
            microbe_features: Dictionary of microbe image features
            style: Overall style preference
            
        Returns:
            Comprehensive generated prompt
        """
        prompt_parts = []
        
        # Base organism description
        if microbe_features and 'type' in microbe_features:
            organism = microbe_features['type']
        else:
            organism = 'biological specimen'
        
        # Add GC content information
        if gc_content is not None:
            gc_prompt = self.generate_from_gc_content(gc_content, style, organism)
            prompt_parts.append(gc_prompt.split(',')[0])  # Take main part
        
        # Add k-mer diversity information
        if kmer_counts:
            kmer_prompt = self.generate_from_kmer_diversity(kmer_counts, organism=organism)
            # Extract feature description
            kmer_feature = kmer_prompt.split(' showing ')[1].split(',')[0] if ' showing ' in kmer_prompt else 'complex patterns'
            prompt_parts.append(kmer_feature)
        
        # Add haplogroup aesthetic
        if haplogroup:
            aesthetics = self.haplogroup_aesthetics.get(haplogroup, ['natural'])
            aesthetic_desc = random.choice(aesthetics[:2])  # Take first 1-2 aesthetics
            prompt_parts.append(f"{aesthetic_desc} aesthetic")
        
        # Add microbe features
        if microbe_features:
            if 'dominant_colors' in microbe_features:
                color_desc = self._analyze_color_palette(microbe_features['dominant_colors'])
                prompt_parts.append(random.choice(color_desc))
        
        # Combine parts
        if prompt_parts:
            main_prompt = f"{organism} with " + ", ".join(prompt_parts[:3])  # Limit to 3 features
        else:
            main_prompt = f"bioart interpretation of {organism}"
        
        # Add style and quality
        style_options = ['scientific illustration', 'bioart style', 'microscopic detail', 'organic art']
        chosen_style = random.choice(style_options)
        
        final_prompt = f"{main_prompt}, {chosen_style}"
        
        # Optional quality enhancer
        if random.random() < 0.4:
            enhancer = random.choice(self.quality_enhancers)
            final_prompt += f", {enhancer}"
        
        return final_prompt
    
    def create_prompt_variations(self, base_prompt: str, num_variations: int = 3) -> List[str]:
        """
        Create variations of a base prompt.
        
        Args:
            base_prompt: Base prompt to vary
            num_variations: Number of variations to create
            
        Returns:
            List of prompt variations
        """
        variations = [base_prompt]
        
        for i in range(num_variations):
            # Add different style modifiers
            variation = base_prompt
            
            # Add random artistic modifier
            if random.random() < 0.6:
                modifiers = ['abstract', 'detailed', 'artistic', 'stylized', 'realistic', 'conceptual']
                modifier = random.choice(modifiers)
                variation += f", {modifier}"
            
            # Add random technique
            if random.random() < 0.5:
                techniques = ['digital art', 'watercolor', 'oil painting', 'photography', '3D render']
                technique = random.choice(techniques)
                variation += f", {technique}"
            
            # Add random quality term
            if random.random() < 0.3:
                quality = random.choice(self.quality_enhancers)
                variation += f", {quality}"
            
            variations.append(variation)
        
        return variations


def demo_prompt_generation():
    """Demonstrate biological prompt generation."""
    print("🧬 Biological Prompt Generation Demo")
    print("=" * 45)
    
    generator = BiologicalPromptGenerator()
    
    # Demo data
    demo_scenarios = [
        {
            'name': 'AT-rich sequence',
            'gc_content': 0.25,
            'kmer_counts': Counter({'AAAA': 20, 'TTTT': 18, 'ATAT': 15, 'TATA': 12}),
            'haplogroup': 'R1a',
            'microbe_type': 'bacteria'
        },
        {
            'name': 'GC-rich sequence',
            'gc_content': 0.75,
            'kmer_counts': Counter({'GCGC': 25, 'CCGG': 22, 'GGCC': 20, 'CGCG': 18}),
            'haplogroup': 'J2',
            'microbe_type': 'fungi'
        },
        {
            'name': 'Balanced sequence',
            'gc_content': 0.50,
            'kmer_counts': Counter({'ATCG': 15, 'CGAT': 14, 'TACG': 13, 'GCTA': 12}),
            'haplogroup': 'I1',
            'microbe_type': 'protist'
        }
    ]
    
    for scenario in demo_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   GC content: {scenario['gc_content']:.1%}")
        print(f"   Haplogroup: {scenario['haplogroup']}")
        print(f"   Microbe type: {scenario['microbe_type']}")
        
        # Generate different types of prompts
        prompt_types = [
            ('GC Content', lambda: generator.generate_from_gc_content(
                scenario['gc_content'], 'scientific', scenario['microbe_type']
            )),
            ('K-mer Diversity', lambda: generator.generate_from_kmer_diversity(
                scenario['kmer_counts'], organism=scenario['microbe_type']
            )),
            ('Haplogroup', lambda: generator.generate_from_haplogroup(
                scenario['haplogroup'], f"{scenario['microbe_type']} patterns"
            )),
            ('Comprehensive', lambda: generator.generate_comprehensive_prompt(
                gc_content=scenario['gc_content'],
                kmer_counts=scenario['kmer_counts'],
                haplogroup=scenario['haplogroup'],
                microbe_features={'type': scenario['microbe_type']}
            ))
        ]
        
        for prompt_type, generator_func in prompt_types:
            prompt = generator_func()
            print(f"   {prompt_type}: {prompt}")
    
    print("\n✅ Prompt generation demo completed!")


if __name__ == "__main__":
    demo_prompt_generation()