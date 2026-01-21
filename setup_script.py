"""
Setup script for CodeForge-Quantum
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from typing import Dict, List, Optional
import platform


class CodeForgeSetup:
    """Setup manager for CodeForge-Quantum"""
    
    def __init__(self, install_dir: str = "."):
        self.install_dir = Path(install_dir)
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.cuda_available = self._check_cuda()
        
        # Required Python version
        self.min_python = (3, 8)
        
        # Directory structure
        self.dirs = {
            'data': self.install_dir / 'data',
            'checkpoints': self.install_dir / 'checkpoints',
            'logs': self.install_dir / 'logs',
            'cache': self.install_dir / 'cache',
            'outputs': self.install_dir / 'outputs',
            'config': self.install_dir / 'config',
        }
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def check_requirements(self) -> bool:
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if self.python_version < self.min_python:
            print(f"❌ Python {self.min_python[0]}.{self.min_python[1]}+ required")
            print(f"   Current: {self.python_version.major}.{self.python_version.minor}")
            return False
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor}")
        
        # Check CUDA
        if self.cuda_available:
            print("✅ CUDA available")
        else:
            print("⚠️  CUDA not detected (CPU mode will be slower)")
        
        # Check disk space
        free_space = shutil.disk_usage(self.install_dir).free / (1024**3)  # GB
        required_space = 100  # GB
        
        if free_space < required_space:
            print(f"⚠️  Low disk space: {free_space:.1f}GB available")
            print(f"   Recommended: {required_space}GB+")
        else:
            print(f"✅ Disk space: {free_space:.1f}GB available")
        
        # Check RAM
        try:
            import psutil
            ram = psutil.virtual_memory().total / (1024**3)  # GB
            if ram < 32:
                print(f"⚠️  RAM: {ram:.1f}GB (32GB+ recommended)")
            else:
                print(f"✅ RAM: {ram:.1f}GB")
        except ImportError:
            print("ℹ️  Install psutil to check RAM")
        
        return True
    
    def create_directories(self):
        """Create required directory structure"""
        print("\n📁 Creating directory structure...")
        
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {name}/")
        
        # Create subdirectories
        subdirs = [
            self.dirs['data'] / 'train',
            self.dirs['data'] / 'val',
            self.dirs['data'] / 'test',
            self.dirs['data'] / 'code_database',
            self.dirs['data'] / 'sample_inputs',
            self.dirs['checkpoints'] / 'best',
            self.dirs['logs'] / 'tensorboard',
        ]
        
        for subdir in subdirs:
            subdir.mkdir(parents=True, exist_ok=True)
    
    def install_dependencies(self, dev: bool = False):
        """Install Python dependencies"""
        print("\n📦 Installing dependencies...")
        
        # Base requirements
        requirements_file = "requirements.txt"
        if not Path(requirements_file).exists():
            print(f"❌ {requirements_file} not found")
            return False
        
        # Install with pip
        cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
        
        if self.cuda_available:
            # Install CUDA-specific packages
            print("  Installing CUDA packages...")
            cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu118"])
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        
        # Development dependencies
        if dev:
            dev_requirements = "requirements-dev.txt"
            if Path(dev_requirements).exists():
                print("  Installing development dependencies...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", dev_requirements
                ], check=True)
        
        return True
    
    def download_models(self, model_size: str = "base"):
        """Download pre-trained models"""
        print("\n📥 Downloading models...")
        
        model_urls = {
            "base": "https://example.com/codeforge-quantum-base.tar.gz",
            "large": "https://example.com/codeforge-quantum-large.tar.gz",
            "full": "https://example.com/codeforge-quantum-full.tar.gz"
        }
        
        if model_size not in model_urls:
            print(f"❌ Unknown model size: {model_size}")
            return False
        
        # Note: These are placeholder URLs
        print(f"  ⚠️  Model download URLs are placeholders")
        print(f"  Please download models manually from the official source")
        
        # Create placeholder checkpoint
        checkpoint_path = self.dirs['checkpoints'] / 'best' / 'checkpoint.pt'
        if not checkpoint_path.exists():
            print("  Creating placeholder checkpoint...")
            
            import torch
            placeholder = {
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'global_step': 0,
                'current_epoch': 0,
                'best_metric': 0.0,
                'config': {
                    'model_name': 'Qwen/Qwen-72B',
                    'num_layers': 80,
                    'hidden_dim': 8192,
                    'vocab_size': 65536
                }
            }
            
            torch.save(placeholder, checkpoint_path)
            print(f"  ✓ Created placeholder at {checkpoint_path}")
        
        return True
    
    def setup_vector_database(self):
        """Initialize vector database"""
        print("\n🗄️ Setting up vector database...")
        
        db_path = self.dirs['data'] / 'code_database'
        
        # Create sample vector database
        try:
            from rag.vector_database import FAISSVectorDatabase, CodeSnippet
            import numpy as np
            
            db = FAISSVectorDatabase(dimension=768, index_type="HNSW")
            
            # Add sample snippets
            sample_snippets = [
                {
                    'code': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
                    'description': 'Recursive Fibonacci implementation',
                    'language': 'python',
                    'tags': ['recursion', 'fibonacci', 'dynamic-programming']
                },
                {
                    'code': 'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
                    'description': 'Binary search implementation',
                    'language': 'python',
                    'tags': ['search', 'binary-search', 'algorithm']
                }
            ]
            
            for snippet_data in sample_snippets:
                snippet = CodeSnippet(
                    code=snippet_data['code'],
                    description=snippet_data['description'],
                    language=snippet_data['language'],
                    tags=snippet_data['tags'],
                    complexity=1,
                    embedding=np.random.randn(768).astype('float32')
                )
                db.add_snippet(snippet)
            
            db.save(str(db_path / 'vector_db'))
            print(f"  ✓ Created vector database with {len(sample_snippets)} samples")
            
        except ImportError:
            print("  ⚠️  Could not import vector database modules")
            print("     Run setup after installation completes")
        
        return True
    
    def create_sample_data(self):
        """Create sample training data"""
        print("\n📝 Creating sample data...")
        
        sample_data = [
            {
                "id": "sample_001",
                "problem": "Write a function to calculate the factorial of a number",
                "solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "difficulty": "easy",
                "test_cases": [
                    {"input": "5", "output": "120"},
                    {"input": "0", "output": "1"}
                ]
            },
            {
                "id": "sample_002",
                "problem": "Implement a function to check if a string is a palindrome",
                "solution": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                "difficulty": "easy",
                "test_cases": [
                    {"input": "'racecar'", "output": "True"},
                    {"input": "'hello'", "output": "False"}
                ]
            }
        ]
        
        # Save sample data
        for split in ['train', 'val', 'test']:
            data_file = self.dirs['data'] / f'{split}.json'
            if not data_file.exists():
                with open(data_file, 'w') as f:
                    json.dump(sample_data, f, indent=2)
                print(f"  ✓ Created {split}.json")
        
        return True
    
    def create_config_files(self):
        """Create default configuration files"""
        print("\n⚙️  Creating configuration files...")
        
        # Training config
        training_config = {
            'model_name': 'Qwen/Qwen-72B',
            'batch_size': 8,
            'learning_rate': 2e-5,
            'num_epochs': 15,
            'device': 'cuda' if self.cuda_available else 'cpu'
        }
        
        config_file = self.dirs['config'] / 'training_config.yaml'
        if not config_file.exists():
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(training_config, f, default_flow_style=False)
            print(f"  ✓ Created training_config.yaml")
        
        return True
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        print("\n🧪 Running installation tests...")
        
        tests_passed = 0
        tests_failed = 0
        
        # Test 1: Import main modules
        try:
            from models.codeforge_quantum import CodeForgeQuantum
            print("  ✅ Core model imports")
            tests_passed += 1
        except ImportError as e:
            print(f"  ❌ Core model import failed: {e}")
            tests_failed += 1
        
        # Test 2: PyTorch availability
        try:
            import torch
            print(f"  ✅ PyTorch {torch.__version__}")
            if torch.cuda.is_available():
                print(f"     CUDA: {torch.cuda.get_device_name(0)}")
            tests_passed += 1
        except ImportError:
            print("  ❌ PyTorch not available")
            tests_failed += 1
        
        # Test 3: Transformers availability
        try:
            import transformers
            print(f"  ✅ Transformers {transformers.__version__}")
            tests_passed += 1
        except ImportError:
            print("  ❌ Transformers not available")
            tests_failed += 1
        
        print(f"\n📊 Tests: {tests_passed} passed, {tests_failed} failed")
        
        return tests_failed == 0
    
    def print_next_steps(self):
        """Print next steps for users"""
        print("\n" + "="*60)
        print("🎉 Setup Complete!")
        print("="*60)
        print("\n📚 Next Steps:")
        print("\n1. Quick test (inference):")
        print("   python scripts/inference.py --interactive")
        print("\n2. Train a model:")
        print("   python scripts/train.py --config config/training_config.yaml")
        print("\n3. Evaluate model:")
        print("   python scripts/evaluate.py --checkpoint checkpoints/best --test-data data/test.json")
        print("\n4. Read documentation:")
        print("   cat README.md")
        print("\n💡 Tips:")
        print("  • Use GPU for faster training/inference")
        print("  • Monitor training with TensorBoard: tensorboard --logdir logs/")
        print("  • Join our community for support and updates")
        print("\n" + "="*60)
    
    def full_setup(self, skip_models: bool = False, dev: bool = False):
        """Run complete setup process"""
        print("="*60)
        print("🚀 CodeForge-Quantum Setup")
        print("="*60)
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ Requirements check failed")
            return False
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.install_dependencies(dev=dev):
            print("\n❌ Dependency installation failed")
            return False
        
        # Download models (optional)
        if not skip_models:
            self.download_models()
        
        # Setup vector database
        self.setup_vector_database()
        
        # Create sample data
        self.create_sample_data()
        
        # Create config files
        self.create_config_files()
        
        # Run tests
        if not self.run_tests():
            print("\n⚠️  Some tests failed, but setup completed")
        
        # Print next steps
        self.print_next_steps()
        
        return True


def main():
    """Main setup entry point"""
    parser = argparse.ArgumentParser(
        description="Setup CodeForge-Quantum environment"
    )
    
    parser.add_argument(
        '--install-dir',
        type=str,
        default='.',
        help='Installation directory'
    )
    parser.add_argument(
        '--skip-models',
        action='store_true',
        help='Skip model download'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Install development dependencies'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean installation (remove existing files)'
    )
    
    args = parser.parse_args()
    
    # Clean installation if requested
    if args.clean:
        print("🧹 Cleaning existing installation...")
        dirs_to_clean = ['data', 'checkpoints', 'logs', 'cache', 'outputs']
        for dir_name in dirs_to_clean:
            dir_path = Path(args.install_dir) / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  ✓ Removed {dir_name}/")
    
    # Run setup
    setup = CodeForgeSetup(args.install_dir)
    success = setup.full_setup(
        skip_models=args.skip_models,
        dev=args.dev
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()