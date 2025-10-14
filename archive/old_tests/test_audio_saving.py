#!/usr/bin/env python3
"""
Test audio saving with and without FFmpeg.
This verifies that the soundfile fallback works correctly.
"""

import sys
sys.path.append('.')

import torch
import shutil
from pathlib import Path
from src.utils.audio_utils import save_audio

print("=" * 70)
print("Testing Audio Saving Methods")
print("=" * 70)
print()

# Check FFmpeg status
ffmpeg_available = shutil.which('ffmpeg') is not None
print(f"FFmpeg status: {'✅ INSTALLED' if ffmpeg_available else '❌ NOT FOUND'}")
print()

# Create test audio (1 second of sine wave)
sample_rate = 32000
duration = 1.0
t = torch.linspace(0, duration, int(sample_rate * duration))
frequency = 440  # A4 note
test_audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)  # [1, samples]

print(f"Created test audio: {test_audio.shape}")
print(f"   Duration: {duration} seconds")
print(f"   Frequency: {frequency} Hz (A4 note)")
print()

# Ensure results directory exists
Path("results").mkdir(exist_ok=True)

# Test 1: Soundfile fallback (always works)
print("Test 1: Soundfile Fallback (no FFmpeg required)")
print("-" * 70)
try:
    save_audio(
        test_audio,
        "results/test_soundfile",
        sample_rate=sample_rate,
        strategy="loudness",
        use_ffmpeg=False
    )
    print("✅ Soundfile method works!")

    # Check if file exists
    if Path("results/test_soundfile.wav").exists():
        file_size = Path("results/test_soundfile.wav").stat().st_size
        print(f"   File created: results/test_soundfile.wav ({file_size} bytes)")
    else:
        print("❌ File was not created!")

except Exception as e:
    print(f"❌ Soundfile method failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: FFmpeg method (if available)
print("Test 2: FFmpeg Method (requires FFmpeg)")
print("-" * 70)
if ffmpeg_available:
    try:
        save_audio(
            test_audio,
            "results/test_ffmpeg",
            sample_rate=sample_rate,
            strategy="loudness",
            use_ffmpeg=True
        )
        print("✅ FFmpeg method works!")

        # Check if file exists
        if Path("results/test_ffmpeg.wav").exists():
            file_size = Path("results/test_ffmpeg.wav").stat().st_size
            print(f"   File created: results/test_ffmpeg.wav ({file_size} bytes)")
        else:
            print("❌ File was not created!")

    except Exception as e:
        print(f"❌ FFmpeg method failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  FFmpeg not installed - skipping this test")
    print("   Install FFmpeg for better audio quality:")
    print("   brew install ffmpeg")

print()

# Test 3: Automatic method selection
print("Test 3: Automatic Method Selection")
print("-" * 70)
try:
    save_audio(
        test_audio,
        "results/test_automatic",
        sample_rate=sample_rate,
        strategy="loudness",
        use_ffmpeg=True  # Will auto-fallback if FFmpeg not available
    )
    print("✅ Automatic selection works!")

    method_used = "FFmpeg" if ffmpeg_available else "Soundfile fallback"
    print(f"   Method used: {method_used}")

    if Path("results/test_automatic.wav").exists():
        file_size = Path("results/test_automatic.wav").stat().st_size
        print(f"   File created: results/test_automatic.wav ({file_size} bytes)")

except Exception as e:
    print(f"❌ Automatic method failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print()

# Count successful files
test_files = [
    "results/test_soundfile.wav",
    "results/test_automatic.wav"
]
if ffmpeg_available:
    test_files.append("results/test_ffmpeg.wav")

existing_files = [f for f in test_files if Path(f).exists()]
print(f"Files created: {len(existing_files)}/{len(test_files)}")

for f in existing_files:
    print(f"  ✅ {f}")

print()

if len(existing_files) == len(test_files):
    print("🎉 ALL TESTS PASSED!")
    print()
    print("Your audio saving is working correctly.")
    if not ffmpeg_available:
        print()
        print("Note: You're using soundfile fallback (FFmpeg not installed)")
        print("For better quality, install FFmpeg:")
        print("  brew install ffmpeg")
        print()
        print("See docs/FFMPEG_SETUP.md for detailed instructions")
else:
    print("⚠️  SOME TESTS FAILED")
    print()
    print("Check the errors above.")

print()
print("Next steps:")
print("  1. Listen to the generated test files to verify quality")
print("  2. If satisfied, continue with your research")
print("  3. If you want better quality, install FFmpeg (see docs/FFMPEG_SETUP.md)")
