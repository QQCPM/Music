#!/usr/bin/env python3
"""
Setup script for MusicGen interpretability research project.
Checks system requirements and downloads necessary models.
"""

import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, but found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu():
    """Check for GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.2f} GB")

            if gpu_memory < 24:
                print(f"⚠️  Warning: MusicGen 3.3B requires ~24GB+ VRAM")
                print(f"   Consider using smaller models or Google Colab Pro")
            return True
        elif torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (MPS) detected")
            print("⚠️  MPS support may vary for large models")
            return True
        else:
            print("❌ No GPU detected")
            print("   MusicGen requires GPU for reasonable inference speed")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - will check GPU after installation")
        return True

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("❌ FFmpeg not found")
    print("   Install instructions:")
    if platform.system() == 'Darwin':  # macOS
        print("   brew install ffmpeg")
    elif platform.system() == 'Linux':
        print("   sudo apt-get install ffmpeg  # Ubuntu/Debian")
        print("   sudo yum install ffmpeg      # CentOS/RHEL")
    elif platform.system() == 'Windows':
        print("   Download from: https://ffmpeg.org/download.html")
    return False

def create_directory_structure():
    """Create project directory structure if it doesn't exist."""
    base_dir = Path(__file__).parent.parent

    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/models",
        "src/utils",
        "src/experiments",
        "results/visualizations",
        "results/models",
        "results/logs",
        "docs",
        "scripts",
    ]

    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)

    print("✅ Project directory structure created")
    return True

def main():
    """Main setup routine."""
    print("=" * 60)
    print("MusicGen Interpretability Research - Environment Setup")
    print("=" * 60)
    print()

    checks = {
        "Python version": check_python_version(),
        "FFmpeg": check_ffmpeg(),
        "Directory structure": create_directory_structure(),
    }

    print()
    print("=" * 60)
    print("Next steps:")
    print("=" * 60)
    print()

    if all(checks.values()):
        print("✅ All basic checks passed!")
        print()
        print("1. Install Python dependencies:")
        print("   pip install -r requirements.txt")
        print()
        print("2. Download MusicGen models:")
        print("   python scripts/download_models.py")
        print()
        print("3. Prepare datasets:")
        print("   python scripts/prepare_datasets.py")
        print()
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print()

    # GPU check (after PyTorch installation)
    print("After installing requirements, verify GPU with:")
    print("   python scripts/check_gpu.py")
    print()

if __name__ == "__main__":
    main()
