#!/usr/bin/env python3
"""
Download MusicGen models from Hugging Face.
Supports multiple model sizes with interactive selection.
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

# Available MusicGen models
MODELS = {
    "small": {
        "name": "facebook/musicgen-small",
        "params": "300M",
        "vram": "~8GB",
        "description": "Smallest model, fastest inference"
    },
    "medium": {
        "name": "facebook/musicgen-medium",
        "params": "1.5B",
        "vram": "~16GB",
        "description": "Good balance of quality and speed"
    },
    "large": {
        "name": "facebook/musicgen-large",
        "params": "3.3B",
        "vram": "~24GB",
        "description": "Highest quality, mono audio"
    },
    "stereo-small": {
        "name": "facebook/musicgen-stereo-small",
        "params": "300M",
        "vram": "~8GB",
        "description": "Stereo audio, smallest"
    },
    "stereo-medium": {
        "name": "facebook/musicgen-stereo-medium",
        "params": "1.5B",
        "vram": "~16GB",
        "description": "Stereo audio, medium size"
    },
    "stereo-large": {
        "name": "facebook/musicgen-stereo-large",
        "params": "3.3B",
        "vram": "~24GB",
        "description": "Stereo audio, largest and highest quality"
    },
    "melody": {
        "name": "facebook/musicgen-melody",
        "params": "1.5B",
        "vram": "~16GB",
        "description": "Melody-conditioned generation"
    },
}

def check_disk_space():
    """Check available disk space."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    print(f"📁 Available disk space: {free_gb} GB")
    if free_gb < 20:
        print("⚠️  Warning: Low disk space. Models can be 5-15GB each.")
    return free_gb

def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        return gpu_memory
    elif torch.backends.mps.is_available():
        print("🎮 GPU: Apple Silicon (MPS)")
        return None  # MPS doesn't expose memory info
    else:
        print("❌ No GPU detected")
        return 0

def print_model_info():
    """Print information about available models."""
    print("\n" + "=" * 70)
    print("Available MusicGen Models")
    print("=" * 70)
    for key, info in MODELS.items():
        print(f"\n{key}:")
        print(f"  Model: {info['name']}")
        print(f"  Parameters: {info['params']}")
        print(f"  VRAM needed: {info['vram']}")
        print(f"  Description: {info['description']}")
    print("=" * 70 + "\n")

def download_model(model_key, cache_dir=None):
    """Download a specific MusicGen model."""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_key]
    model_name = model_info['name']

    print(f"\n📥 Downloading {model_name}...")
    print(f"   Parameters: {model_info['params']}")
    print(f"   VRAM needed: {model_info['vram']}")
    print()

    try:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
        )

        print(f"✅ Successfully downloaded {model_name}")
        return True

    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def interactive_download():
    """Interactive model selection and download."""
    print_model_info()

    gpu_memory = check_gpu_memory()
    check_disk_space()

    print("\nRecommendations based on your GPU:")
    if gpu_memory is None:
        print("  - Any model should work on Apple Silicon (memory shared with system)")
    elif gpu_memory >= 24:
        print("  - You can run any model, including 'large' and 'stereo-large'")
    elif gpu_memory >= 16:
        print("  - Recommended: 'medium', 'stereo-medium', or 'melody'")
    elif gpu_memory >= 8:
        print("  - Recommended: 'small' or 'stereo-small'")
    else:
        print("  - Consider using Google Colab Pro or cloud GPU")

    print("\nWhich model would you like to download?")
    print("(You can download multiple models by running this script again)")
    print("Enter model name (e.g., 'large', 'stereo-large') or 'all' for all models:")
    print("(Press Ctrl+C to cancel)")

    try:
        choice = input("\nYour choice: ").strip().lower()

        if choice == 'all':
            print("\n⚠️  This will download ALL models (~50GB+ total). Are you sure? (yes/no)")
            confirm = input("Confirm: ").strip().lower()
            if confirm == 'yes':
                for model_key in MODELS.keys():
                    download_model(model_key)
            else:
                print("Cancelled.")
        elif choice in MODELS:
            download_model(choice)
        else:
            print(f"❌ Invalid choice: {choice}")
            print(f"Valid options: {', '.join(MODELS.keys())}, 'all'")

    except KeyboardInterrupt:
        print("\n\nCancelled by user.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download MusicGen models from Hugging Face"
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=list(MODELS.keys()) + ["all"],
        help="Model to download (or 'all' for all models)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Custom cache directory for models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list:
        print_model_info()
        return

    if args.model:
        if args.model == "all":
            print("Downloading all models...")
            for model_key in MODELS.keys():
                download_model(model_key, args.cache_dir)
        else:
            download_model(args.model, args.cache_dir)
    else:
        # Interactive mode
        interactive_download()

if __name__ == "__main__":
    main()
