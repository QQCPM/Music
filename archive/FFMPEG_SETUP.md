# FFmpeg Setup Guide

FFmpeg is an optional but recommended dependency for MusicGen audio processing. This guide explains why you need it, how to install it, and what to do if you can't install it.

---

## Do You Need FFmpeg?

### ✅ You DON'T need FFmpeg if:
- You're just doing activation extraction and analysis
- You're okay with basic WAV file output (using our soundfile fallback)
- You're testing code and don't need high-quality audio

### 🎵 You DO need FFmpeg if:
- You want the best audio quality
- You want to save in formats other than WAV (MP3, OGG, etc.)
- You want loudness normalization and compression
- You're doing serious music generation research

---

## Current Status

Run this command to check if FFmpeg is installed:

```bash
which ffmpeg
```

**If you see a path** (e.g., `/usr/local/bin/ffmpeg`):
✅ FFmpeg is installed!

**If you see** `ffmpeg not found`:
❌ FFmpeg is NOT installed (this is your current situation)

---

## Installation Instructions

### Option 1: Install Homebrew + FFmpeg (Recommended)

Homebrew is a package manager for macOS that makes installing tools like FFmpeg easy.

#### Step 1: Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

This will:
- Download and install Homebrew
- Take 5-10 minutes
- Ask for your password

#### Step 2: Install FFmpeg

```bash
brew install ffmpeg
```

This will:
- Download and install FFmpeg
- Take 2-5 minutes
- Install all dependencies

#### Step 3: Verify Installation

```bash
ffmpeg -version
```

You should see:
```
ffmpeg version 6.x.x ...
```

✅ Done! FFmpeg is now installed.

---

### Option 2: Download FFmpeg Binary Directly

If you don't want to install Homebrew:

#### Step 1: Download FFmpeg

1. Visit: https://evermeet.cx/ffmpeg/
2. Download the latest **ffmpeg** build (not ffprobe)
3. You'll get a file like `ffmpeg-6.x.x.7z`

#### Step 2: Extract and Install

```bash
# Create a directory for FFmpeg
mkdir -p ~/bin

# Extract (you may need to install 7-Zip first)
# Or just double-click the .7z file in Finder

# Move the ffmpeg binary
mv ~/Downloads/ffmpeg ~/bin/ffmpeg

# Make it executable
chmod +x ~/bin/ffmpeg

# Add to PATH (add this to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Step 3: Verify

```bash
ffmpeg -version
```

✅ Done!

---

### Option 3: Use the Soundfile Fallback (No FFmpeg)

**If you can't or don't want to install FFmpeg**, the project will automatically use a fallback method that saves basic WAV files.

Our code automatically detects if FFmpeg is available and uses the appropriate method.

**What you get:**
- ✅ Audio files work
- ✅ WAV format (widely compatible)
- ✅ Basic normalization
- ❌ No loudness compression
- ❌ No MP3/OGG export
- ❌ Slightly lower quality

**No changes needed** - the code handles this automatically!

---

## How Our Code Handles This

### Automatic Detection

```python
from src.utils.audio_utils import save_audio

# This automatically checks for FFmpeg
save_audio(wav, "output", sample_rate=32000)

# Or explicitly use soundfile fallback
save_audio(wav, "output", sample_rate=32000, use_ffmpeg=False)
```

### What Happens

1. **If FFmpeg is installed**: Uses `audiocraft.data.audio.audio_write` with full quality
2. **If FFmpeg is NOT installed**: Uses `soundfile` fallback with basic normalization

You'll see a warning if FFmpeg is not found:
```
⚠️  FFmpeg not found. Using soundfile fallback.
   Install FFmpeg for better quality: brew install ffmpeg
```

---

## Testing Your Setup

### Test 1: Check FFmpeg Status

```bash
cd "/Users/lending/Documents/AI PRJ/MusicGen"
python3 -c "import shutil; print('FFmpeg:', 'INSTALLED' if shutil.which('ffmpeg') else 'NOT FOUND')"
```

### Test 2: Test Audio Saving

```python
import torch
from src.utils.audio_utils import save_audio

# Create dummy audio
dummy_audio = torch.randn(1, 32000)  # 1 second of random audio

# Test with FFmpeg (if available)
save_audio(dummy_audio, "results/test_ffmpeg", use_ffmpeg=True)

# Test with soundfile fallback
save_audio(dummy_audio, "results/test_soundfile", use_ffmpeg=False)

print("✅ Both methods work!")
```

---

## Troubleshooting

### Issue: "brew: command not found"

You don't have Homebrew installed. Either:
- Install Homebrew (see Option 1 above)
- Or use Option 2 (direct download)
- Or use Option 3 (soundfile fallback)

### Issue: "Permission denied" when installing Homebrew

Run with sudo:
```bash
sudo /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Issue: FFmpeg installed but still not found

Check your PATH:
```bash
echo $PATH
```

Make sure `/usr/local/bin` or `~/bin` is in your PATH.

Add to your shell config (~/.zshrc or ~/.bash_profile):
```bash
export PATH="/usr/local/bin:$PATH"
```

Then reload:
```bash
source ~/.zshrc  # or source ~/.bash_profile
```

### Issue: "soundfile not found"

Install it:
```bash
pip install soundfile
```

(It should already be in requirements.txt, but just in case)

---

## Recommendations

### For Serious Research
**Install FFmpeg** - You want the best audio quality for your experiments.

```bash
brew install ffmpeg
```

**Time**: 10-15 minutes (including Homebrew installation)

### For Quick Testing
**Use soundfile fallback** - It works fine for activation extraction and testing.

**Time**: 0 minutes (already working)

### For Long-Term
Even if you start with soundfile fallback, **plan to install FFmpeg eventually**. The audio quality difference is noticeable for music generation research.

---

## Summary

| Method | Quality | Setup Time | Recommendation |
|--------|---------|-----------|----------------|
| FFmpeg via Homebrew | ⭐⭐⭐⭐⭐ | 10-15 min | Best for research |
| FFmpeg direct | ⭐⭐⭐⭐⭐ | 15-20 min | If no Homebrew |
| Soundfile fallback | ⭐⭐⭐ | 0 min | Works immediately |

---

## Current Project Status

✅ **Your code now works WITHOUT FFmpeg** (uses soundfile fallback)
⚠️  **But you should install FFmpeg for better results**

To install FFmpeg now:
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg

# Verify
ffmpeg -version
```

Then your code will automatically use the better method!

---

## Questions?

- **How do I know which method is being used?**
  - You'll see "Using soundfile fallback" if FFmpeg is not found

- **Will my existing code work?**
  - Yes! The fallback is automatic.

- **Should I change my code?**
  - No changes needed. Install FFmpeg when you can, and the code will automatically use it.

---

**Updated**: Audio saving now works with or without FFmpeg! 🎵
