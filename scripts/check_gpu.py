#!/usr/bin/env python3
"""
Check GPU availability and capabilities for MusicGen.
"""

import sys

def check_torch():
    """Check PyTorch installation and GPU support."""
    try:
        import torch
        print("✅ PyTorch installed")
        print(f"   Version: {torch.__version__}")
        return torch
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision torchaudio")
        return None

def check_cuda(torch):
    """Check CUDA availability."""
    if torch.cuda.is_available():
        print("\n✅ CUDA GPU detected")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_memory = props.total_memory / (1024**3)

            print(f"\n   GPU {i}: {props.name}")
            print(f"   - VRAM: {gpu_memory:.2f} GB")
            print(f"   - Compute capability: {props.major}.{props.minor}")

            # Recommendations
            print(f"\n   Recommendations for GPU {i}:")
            if gpu_memory >= 24:
                print(f"   ✅ Can run MusicGen Large (3.3B)")
                print(f"   ✅ Can run MusicGen Stereo Large (3.3B)")
                print(f"   ✅ Can run MusicGen Medium (1.5B)")
                print(f"   ✅ Can run MusicGen Small (300M)")
            elif gpu_memory >= 16:
                print(f"   ⚠️  MusicGen Large (3.3B) may be tight")
                print(f"   ✅ Can run MusicGen Medium (1.5B)")
                print(f"   ✅ Can run MusicGen Small (300M)")
            elif gpu_memory >= 8:
                print(f"   ❌ Cannot run MusicGen Large (3.3B)")
                print(f"   ⚠️  MusicGen Medium (1.5B) may be tight")
                print(f"   ✅ Can run MusicGen Small (300M)")
            else:
                print(f"   ❌ Insufficient VRAM for most models")
                print(f"   Consider using Google Colab Pro or cloud GPU")

        return True
    return False

def check_mps(torch):
    """Check Apple Silicon GPU (MPS) availability."""
    if torch.backends.mps.is_available():
        print("\n✅ Apple Silicon GPU (MPS) detected")
        print("   Note: MPS shares system RAM")

        # Try to get system memory info
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024**3)
            print(f"   System RAM: {total_memory:.1f} GB")

            if total_memory >= 32:
                print("\n   Recommendations:")
                print("   ✅ Should be able to run MusicGen Large (3.3B)")
                print("   ✅ Can run MusicGen Medium (1.5B)")
                print("   ✅ Can run MusicGen Small (300M)")
            elif total_memory >= 16:
                print("\n   Recommendations:")
                print("   ⚠️  MusicGen Large (3.3B) may be slow")
                print("   ✅ Can run MusicGen Medium (1.5B)")
                print("   ✅ Can run MusicGen Small (300M)")
            else:
                print("\n   Recommendations:")
                print("   ❌ May struggle with larger models")
                print("   ✅ Can run MusicGen Small (300M)")
        except ImportError:
            print("   Install psutil to see memory info: pip install psutil")

        return True
    return False

def check_cpu_fallback():
    """Warn about CPU-only execution."""
    print("\n⚠️  No GPU detected - CPU-only mode")
    print("   MusicGen will be VERY slow on CPU")
    print("   Recommendations:")
    print("   1. Use Google Colab (free GPU): https://colab.research.google.com/")
    print("   2. Use Paperspace Gradient (pay-per-use GPU)")
    print("   3. Consider cloud GPU services (AWS, GCP, Azure)")
    return False

def run_simple_test(torch):
    """Run a simple PyTorch GPU test."""
    print("\n" + "=" * 60)
    print("Running GPU Test")
    print("=" * 60)

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Creating tensor on {device}...")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print("✅ GPU computation successful!")

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Creating tensor on {device}...")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print("✅ MPS computation successful!")

        else:
            print("Running CPU test...")
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            z = torch.matmul(x, y)
            print("✅ CPU computation successful (but will be slow for models)")

        return True

    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Main GPU check routine."""
    print("=" * 60)
    print("GPU and PyTorch Environment Check")
    print("=" * 60)
    print()

    torch = check_torch()
    if not torch:
        sys.exit(1)

    has_gpu = check_cuda(torch)

    if not has_gpu:
        has_gpu = check_mps(torch)

    if not has_gpu:
        check_cpu_fallback()

    # Run simple test
    run_simple_test(torch)

    print("\n" + "=" * 60)
    print("Check complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
