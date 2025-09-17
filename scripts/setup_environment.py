#!/usr/bin/env python3
"""
Environment Setup Script for BioArt Learning Process

Sets up the development environment and verifies all dependencies.

Usage:
    python scripts/setup_environment.py [--check-only]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple


class BioArtEnvironmentSetup:
    """Sets up and verifies the BioArt learning environment."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.required_dirs = [
            "data/raw",
            "data/processed", 
            "data/features",
            "src/data",
            "src/models",
            "src/art",
            "src/utils",
            "notebooks",
            "scripts",
            "configs",
            "models/cache",
            "models/checkpoints",
            "outputs"
        ]
        
    def check_python_version(self) -> bool:
        """Check if Python version is sufficient."""
        version = sys.version_info
        min_version = (3, 8)
        
        print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version >= min_version:
            print("✅ Python version is sufficient")
            return True
        else:
            print(f"❌ Python {min_version[0]}.{min_version[1]}+ required")
            return False
    
    def create_directories(self) -> bool:
        """Create required directories."""
        print("📁 Creating directory structure...")
        
        success = True
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ {dir_path}")
            except Exception as e:
                print(f"❌ {dir_path}: {e}")
                success = False
        
        return success
    
    def check_dependencies(self) -> Tuple[List[str], List[str]]:
        """Check which dependencies are installed."""
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return [], []
        
        # Read requirements
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        # Parse package names
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before version specifiers)
                pkg_name = line.split('>=')[0].split('==')[0].split('<')[0].strip()
                packages.append(pkg_name)
        
        # Check each package
        installed = []
        missing = []
        
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
                installed.append(package)
            except ImportError:
                missing.append(package)
        
        return installed, missing
    
    def install_dependencies(self) -> bool:
        """Install missing dependencies."""
        print("📦 Installing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], cwd=self.project_root)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def verify_kaggle_setup(self) -> bool:
        """Verify Kaggle API setup."""
        print("🔑 Checking Kaggle API setup...")
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            print("⚠️  Kaggle API credentials not found")
            print("   Please download kaggle.json from https://www.kaggle.com/account")
            print(f"   and place it at: {kaggle_json}")
            return False
        
        # Check permissions
        try:
            import stat
            mode = kaggle_json.stat().st_mode
            if mode & stat.S_IROTH or mode & stat.S_IWOTH:
                print("⚠️  Kaggle credentials have incorrect permissions")
                print(f"   Run: chmod 600 {kaggle_json}")
                return False
        except Exception:
            pass  # Permission check might fail on some systems
        
        # Test API
        try:
            import kaggle
            api = kaggle.KaggleApi()
            api.authenticate()
            print("✅ Kaggle API setup successful")
            return True
        except Exception as e:
            print(f"❌ Kaggle API test failed: {e}")
            return False
    
    def create_gitkeeps(self) -> bool:
        """Create .gitkeep files for empty directories."""
        print("📝 Creating .gitkeep files...")
        
        gitkeep_dirs = [
            ("data/raw", "Raw Kaggle datasets (not committed to git)"),
            ("data/processed", "Cleaned and preprocessed data"),
            ("data/features", "Extracted features for art generation"),
            ("models/cache", "Cached model files"),
            ("models/checkpoints", "Model training checkpoints"),
            ("outputs", "Generated art outputs")
        ]
        
        success = True
        for dir_path, description in gitkeep_dirs:
            full_path = self.project_root / dir_path
            gitkeep_path = full_path / ".gitkeep"
            
            if not gitkeep_path.exists():
                try:
                    gitkeep_path.write_text(f"# {description}\n")
                    print(f"✅ {dir_path}/.gitkeep")
                except Exception as e:
                    print(f"❌ {dir_path}/.gitkeep: {e}")
                    success = False
        
        return success
    
    def run_setup(self, check_only: bool = False) -> bool:
        """Run complete environment setup."""
        print("🎨 BioArt Learning Process - Environment Setup")
        print("=" * 50)
        
        checks = []
        
        # Check Python version
        checks.append(self.check_python_version())
        
        if not check_only:
            # Create directories
            checks.append(self.create_directories())
            
            # Create .gitkeep files
            checks.append(self.create_gitkeeps())
        
        # Check dependencies
        print("\n📋 Checking dependencies...")
        installed, missing = self.check_dependencies()
        
        if installed:
            print(f"✅ Installed packages ({len(installed)}):")
            for pkg in installed[:5]:  # Show first 5
                print(f"   {pkg}")
            if len(installed) > 5:
                print(f"   ... and {len(installed) - 5} more")
        
        if missing:
            print(f"❌ Missing packages ({len(missing)}):")
            for pkg in missing:
                print(f"   {pkg}")
            
            if not check_only:
                print("\n📦 Installing missing dependencies...")
                checks.append(self.install_dependencies())
        else:
            checks.append(True)  # All dependencies satisfied
        
        # Check Kaggle setup
        print()
        checks.append(self.verify_kaggle_setup())
        
        # Summary
        print("\n" + "=" * 50)
        success_count = sum(checks)
        total_count = len(checks)
        
        print(f"🎯 Setup Summary: {success_count}/{total_count} checks passed")
        
        if success_count == total_count:
            print("✅ Environment setup successful!")
            self._print_next_steps()
            return True
        else:
            print("⚠️  Some setup steps failed. Please address the issues above.")
            return False
    
    def _print_next_steps(self):
        """Print next steps after successful setup."""
        print("\n🚀 Next Steps:")
        print("1. Download datasets:")
        print("   python scripts/download_data.py --all")
        print("2. Start exploring:")
        print("   jupyter notebook notebooks/01_data_exploration.ipynb")
        print("3. Generate your first bioart:")
        print("   python scripts/generate_art.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up BioArt learning environment"
    )
    parser.add_argument(
        "--check-only", "-c",
        action="store_true",
        help="Only check environment, don't make changes"
    )
    
    args = parser.parse_args()
    
    setup = BioArtEnvironmentSetup()
    success = setup.run_setup(check_only=args.check_only)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())