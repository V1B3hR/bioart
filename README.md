# BioArt Learning Process: From Genomic and Microbial Data to Generative Art

This project outlines a learning process and production pipeline for creating bioart informed by biological datasets. It uses:
- Human genome sequences (GRCh38)
- Human DNA samples
- Y-DNA haplogroups by ethnic group (tabular metadata)
- Microbe images (for texture and motif training)

Data sources (Kaggle):
- GRCh38 Human Genome DNA: https://www.kaggle.com/datasets/aliabedimadiseh/grch38-human-genome-dna
- Microbes Dataset: https://www.kaggle.com/datasets/sayansh001/microbes-dataset
- Y-DNA Haplogroups by Ethnic Group: https://www.kaggle.com/datasets/mariusc/y-dna-haplogroups-by-ethnic-group
- Human DNA Data: https://www.kaggle.com/datasets/neelvasani/humandnadata

Important notes:
- This project is strictly for computational, creative, and educational purposes. It does not involve any wet-lab protocols or sequence design/optimization for biological function.
- Respect all dataset licenses and Kaggle terms. Avoid re-identification or personal data misuse. Use aggregated, non-identifiable features.

---

## Learning Process Overview

The process is split into phases. Each phase has goals, activities, and outputs. You can iterate through phases as your aesthetic and technical direction evolves.


### Phase 0: Ethics, Intent, and Constraints
- Define your artistic intent: What relationships between life, code, and form do you want to explore?
- Establish guardrails:
  - Do not attempt functional predictions, sequence engineering, or biological optimization.
  - Aggregate features to avoid individual-level inference.
  - Use dataset licensing-compliant workflows (no redistribution of raw data if restricted).
- Output: A short artistic statement and an ethics note.

### Phase 1: Data Familiarization and EDA
- Explore datasets:
  - GRCh38/Human DNA: sequence lengths, GC-content distributions, k-mer spectra.
  - Y-DNA Haplogroups: categorical/ethnic metadata; build palettes or motif rules conditioned on metadata.
  - Microbes images: categories, textures, color spaces; identify motifs.
- Activities:
  - Compute descriptive statistics (k-mer histograms, GC-content).
  - Visualize microbe texture distributions (e.g., via CLIP or simple CNN embeddings).
- Output: EDA notebooks and an initial design brief linking data features to visual motifs.

### Phase 2: Conceptual Mappings (Data → Visual Language)
- Define mappings from biological features to art parameters:
  - DNA-derived palettes: map nucleotide/k-mer frequencies to color palettes or gradients.
  - Haplogroup metadata → compositional rules (e.g., layout, symmetry, rhythm).
  - Microbe textures → learned texture priors via generative models (diffusion or VQ-VAE).
- Activities:
  - Build feature extractors/encoders for DNA sequences (non-functional embeddings).
  - Create palette builders (e.g., GC-content → hue/temperature, k-mer diversity → saturation).
  - Train or fine-tune lightweight texture models using microbe images.
- Output: Feature extraction pipelines and mapping function definitions.

### Phase 3: Generative Model Integration
- Implement models for creative generation:
  - Non-functional DNA embeddings (e.g., k-mer tokenization → transformer features).
  - LoRA (Low-Rank Adaptation) fine-tuning for microbial texture generation via diffusion models.
  - Prompt engineering using DNA-derived descriptors.
- Activities:
  - Build a simple DNA encoder (k-mer based, not for biological function).
  - Set up a diffusion model fine-tuning pipeline (e.g., Stable Diffusion + LoRA).
  - Create text prompts/descriptors from sequence features.
- Output: Trained models and generation pipelines ready for creative iteration.

### Phase 4: Creative Iteration and Composition
- Use trained models and mappings for art creation:
  - Generate textures and color palettes from real biological data.
  - Create compositions combining multiple data sources (DNA + microbes + metadata).
  - Iterate on aesthetic parameters and refine mappings.
- Activities:
  - Batch generate art samples with different hyperparameters.
  - Evaluate aesthetic coherence and biological authenticity.
  - Document the creative process and insights.
- Output: A portfolio of bioart pieces and documentation of the creative process.

---

## Repository Structure

```
bioart/
├── data/
│   ├── raw/                 # Kaggle datasets (not committed to git)
│   ├── processed/           # Cleaned and preprocessed data
│   └── features/            # Extracted features for art generation
├── src/
│   ├── data/
│   │   ├── download.py      # Kaggle dataset downloading
│   │   └── preprocessing.py # Data cleaning and feature extraction
│   ├── models/
│   │   ├── dna_embedding.py # Non-functional DNA sequence embedding
│   │   └── diffusion_lora.py # LoRA fine-tuning for texture generation
│   ├── art/
│   │   ├── palettes.py      # Color palette generation from DNA features
│   │   ├── prompts.py       # Text prompt generation from biological data
│   │   └── composition.py   # Art composition and generation
│   └── utils/
│       ├── config.py        # Configuration management
│       └── visualization.py # EDA and result visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_mappings.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_art_generation.ipynb
├── scripts/
│   ├── setup_environment.py
│   ├── download_data.py
│   └── generate_art.py
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── art_config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Kaggle account and API key for dataset access
- GPU recommended for diffusion model training (optional for embeddings)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd bioart

# Install dependencies
pip install -r requirements.txt

# Set up Kaggle API (place kaggle.json in ~/.kaggle/)
# Download datasets
python scripts/download_data.py

# Set up environment
python scripts/setup_environment.py
```

---

## Quick Start

### Example Workflow
```bash
# 1. Download and explore data
python scripts/download_data.py
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Extract features and build mappings
python -c "
from src.data.preprocessing import SequenceProcessor
processor = SequenceProcessor()
features = processor.extract_dna_features('path/to/sequences.fasta')
print(f'Extracted features for {len(features)} sequences')
"

# 3. Generate DNA-based color palettes
python -c "
from src.art.palettes import DNAPaletteGenerator
generator = DNAPaletteGenerator()
palette = generator.gc_content_palette(gc_content=0.45)
print(f'Generated palette: {palette}')
"

# 4. Create art compositions
python scripts/generate_art.py --config configs/art_config.yaml
```

---

## Key Components

### Data Processing
- **`src/data/download.py`**: Automated Kaggle dataset downloading with proper attribution
- **`src/data/preprocessing.py`**: Feature extraction from genomic sequences and microbe images
- **Safe aggregation**: No individual-level data, only population-level statistics

### Models
- **`src/models/dna_embedding.py`**: Non-functional DNA sequence embeddings using k-mer tokenization
- **`src/models/diffusion_lora.py`**: Lightweight LoRA fine-tuning for microbial texture generation
- **No biological function**: Purely computational/artistic feature extraction

### Art Generation
- **`src/art/palettes.py`**: DNA sequence features → color palette mappings
- **`src/art/prompts.py`**: Biological descriptors → generative art prompts
- **`src/art/composition.py`**: Multi-modal art composition combining DNA + microbe + metadata features

---

## Ethical Guidelines

This project adheres to strict ethical guidelines for computational bioart:

1. **No Functional Biology**: No sequence design, optimization, or prediction of biological function
2. **Privacy Protection**: Use only aggregated, non-identifiable features from population data
3. **Dataset Compliance**: Respect all dataset licenses and Kaggle terms of service
4. **Attribution**: Proper citation of all data sources and methodologies
5. **Transparency**: Clear documentation of all processing steps and mappings

---

## Dataset Information

### GRCh38 Human Genome DNA
- **Source**: https://www.kaggle.com/datasets/aliabedimadiseh/grch38-human-genome-dna
- **Usage**: K-mer analysis, GC content distribution, sequence composition features
- **Features Extracted**: Non-functional statistical properties only

### Microbes Dataset
- **Source**: https://www.kaggle.com/datasets/sayansh001/microbes-dataset
- **Usage**: Texture and visual motif training for generative models
- **Features Extracted**: Visual textures, color distributions, morphological patterns

### Y-DNA Haplogroups by Ethnic Group
- **Source**: https://www.kaggle.com/datasets/mariusc/y-dna-haplogroups-by-ethnic-group
- **Usage**: Population-level metadata for compositional rules and artistic variation
- **Features Extracted**: Categorical distributions for palette and style variation

### Human DNA Data
- **Source**: https://www.kaggle.com/datasets/neelvasani/humandnadata
- **Usage**: Additional sequence diversity for k-mer spectrum analysis
- **Features Extracted**: Aggregate sequence statistics and compositional metrics

---

## Configuration

The project uses YAML configuration files in the `configs/` directory:

- **`data_config.yaml`**: Dataset paths, preprocessing parameters, feature extraction settings
- **`model_config.yaml`**: Embedding dimensions, training hyperparameters, model architectures
- **`art_config.yaml`**: Color mapping parameters, composition rules, output formats

---

## Contributing

Contributions are welcome! Please focus on:

- Creative mapping functions from biological data to artistic parameters
- New generative model integrations (keeping them lightweight and non-functional)
- Documentation improvements and ethical guideline refinements
- Additional dataset integrations (with proper licensing compliance)

---

## License

This project is for educational and artistic purposes. Please respect all dataset licenses and use responsibly. Do not use for any biological engineering, medical applications, or individual identification purposes.

---

## Acknowledgments

- Kaggle dataset contributors for providing open biological data
- The computational biology and digital art communities for inspiration and methodological guidance
- Open source machine learning and computer vision libraries that make this creative exploration possible

