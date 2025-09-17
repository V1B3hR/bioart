#!/usr/bin/env python3
"""
Data Download Script for BioArt Learning Process

Downloads Kaggle datasets for bioart generation.
Requires Kaggle API credentials (~/.kaggle/kaggle.json)

Usage:
    python scripts/download_data.py [--dataset DATASET_NAME] [--all]
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List

try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: kaggle package not found. Please install with: pip install kaggle")
    sys.exit(1)


class BioArtDataDownloader:
    """Downloads and organizes Kaggle datasets for bioart generation."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize downloader with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.api = KaggleApi()
        
    def _load_config(self) -> Dict:
        """Load data configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_kaggle_api(self) -> bool:
        """Set up Kaggle API authentication."""
        try:
            self.api.authenticate()
            print("✅ Kaggle API authentication successful")
            return True
        except Exception as e:
            print(f"❌ Kaggle API authentication failed: {e}")
            print("\nPlease ensure you have:")
            print("1. A Kaggle account")
            print("2. API credentials at ~/.kaggle/kaggle.json")
            print("3. Accepted the terms for each dataset on Kaggle website")
            return False
    
    def download_dataset(self, dataset_key: str) -> bool:
        """Download a specific dataset."""
        if dataset_key not in self.config['datasets']:
            print(f"❌ Unknown dataset: {dataset_key}")
            return False
            
        dataset_info = self.config['datasets'][dataset_key]
        dataset_name = dataset_info['name']
        local_path = Path(dataset_info['local_path'])
        
        print(f"\n📥 Downloading {dataset_info['description']}...")
        print(f"   Dataset: {dataset_name}")
        print(f"   URL: {dataset_info['url']}")
        
        try:
            # Create local directory
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            self.api.dataset_download_files(
                dataset_name, 
                path=str(local_path),
                unzip=True
            )
            
            print(f"✅ Downloaded to: {local_path}")
            
            # Verify download
            if self._verify_download(local_path):
                print("✅ Download verified")
                return True
            else:
                print("⚠️  Download verification failed")
                return False
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def _verify_download(self, path: Path) -> bool:
        """Verify that download was successful."""
        if not path.exists():
            return False
            
        # Check if directory has files
        files = list(path.glob("*"))
        return len(files) > 0
    
    def download_all(self) -> bool:
        """Download all configured datasets."""
        print("🚀 Starting download of all bioart datasets...")
        print("=" * 50)
        
        success_count = 0
        total_count = len(self.config['datasets'])
        
        for dataset_key in self.config['datasets']:
            if self.download_dataset(dataset_key):
                success_count += 1
        
        print("\n" + "=" * 50)
        print(f"🎯 Download Summary: {success_count}/{total_count} successful")
        
        if success_count == total_count:
            print("✅ All datasets downloaded successfully!")
            self._print_next_steps()
            return True
        else:
            print("⚠️  Some downloads failed. Check error messages above.")
            return False
    
    def _print_next_steps(self):
        """Print next steps after successful download."""
        print("\n🎨 Next Steps:")
        print("1. Explore the data:")
        print("   jupyter notebook notebooks/01_data_exploration.ipynb")
        print("2. Extract features:")
        print("   python -m src.data.preprocessing")
        print("3. Generate art:")
        print("   python scripts/generate_art.py")
        
    def list_datasets(self):
        """List all available datasets."""
        print("📋 Available Datasets:")
        print("=" * 50)
        
        for key, info in self.config['datasets'].items():
            status = "✅" if Path(info['local_path']).exists() else "❌"
            print(f"{status} {key}")
            print(f"   Description: {info['description']}")
            print(f"   URL: {info['url']}")
            print(f"   Local path: {info['local_path']}")
            print()
    
    def check_status(self):
        """Check download status of all datasets."""
        print("📊 Download Status:")
        print("=" * 50)
        
        downloaded = 0
        total = len(self.config['datasets'])
        
        for key, info in self.config['datasets'].items():
            path = Path(info['local_path'])
            if path.exists() and self._verify_download(path):
                status = "✅ Downloaded"
                downloaded += 1
            else:
                status = "❌ Not downloaded"
            
            print(f"{status} {key}")
        
        print(f"\nStatus: {downloaded}/{total} datasets downloaded")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets for BioArt learning process"
    )
    parser.add_argument(
        "--dataset", "-d",
        help="Download specific dataset (grch38_genome, microbes, y_dna_haplogroups, human_dna)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true", 
        help="List available datasets"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Check download status"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        downloader = BioArtDataDownloader(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    if args.list:
        downloader.list_datasets()
        return 0
    
    if args.status:
        downloader.check_status()
        return 0
    
    # Set up API authentication
    if not downloader.setup_kaggle_api():
        return 1
    
    success = True
    
    if args.all:
        success = downloader.download_all()
    elif args.dataset:
        success = downloader.download_dataset(args.dataset)
    else:
        # Interactive mode
        print("🎨 BioArt Data Downloader")
        print("Choose an option:")
        print("1. Download all datasets")
        print("2. Download specific dataset")
        print("3. Check status")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            success = downloader.download_all()
        elif choice == "2":
            downloader.list_datasets()
            dataset = input("Enter dataset key: ").strip()
            success = downloader.download_dataset(dataset)
        elif choice == "3":
            downloader.check_status()
        else:
            print("Invalid choice")
            return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())