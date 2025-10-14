# Quick Start Guide

Get up and running with MusicGen interpretability research in 30 minutes.

---

## Step 1: Check System Requirements

```bash
python3 scripts/setup_environment.py
```

This will check:
- ✅ Python 3.9+
- ✅ FFmpeg installation
- ✅ Directory structure

---

## Step 2: Install Dependencies

### Option A: Full Installation (Recommended)
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Option B: Minimal Installation (for testing)
```bash
pip install torch torchvision torchaudio
pip install audiocraft transformers accelerate
pip install librosa soundfile matplotlib jupyter
```

**Note**: For CUDA GPU support, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note**: For Apple Silicon Macs, standard PyTorch includes MPS support automatically.

---

## Step 3: Verify GPU

```bash
python3 scripts/check_gpu.py
```

This will show:
- GPU type and VRAM
- Recommended models for your hardware
- Quick PyTorch GPU test

**Expected Output**:
```
✅ CUDA GPU detected
   GPU 0: NVIDIA RTX 4090
   - VRAM: 24.00 GB

   Recommendations:
   ✅ Can run MusicGen Large (3.3B)
   ✅ Can run MusicGen Stereo Large (3.3B)
```

---

## Step 4: Download MusicGen Models

### Interactive Mode (Recommended for first time)
```bash
python3 scripts/download_models.py
```

Follow the prompts to select your model.

### Direct Download (if you know which model)

**For MusicGen Large (3.3B) - Your Choice!**
```bash
python3 scripts/download_models.py large
```

**For MusicGen Stereo Large (3.3B) - Even Better Quality**
```bash
python3 scripts/download_models.py stereo-large
```

**Other Options**:
```bash
# List all available models
python3 scripts/download_models.py --list

# Download all models (not recommended - very large!)
python3 scripts/download_models.py all
```

**Storage Requirements**:
- Small (300M): ~1.2 GB
- Medium (1.5B): ~6 GB
- **Large (3.3B): ~13 GB** ← Your choice
- **Stereo Large (3.3B): ~13 GB** ← Recommended

---

## Step 5: Test MusicGen

### Quick Test in Python

```python
from audiocraft.models import MusicGen
import torch

# Load the model (this will take a few minutes first time)
print("Loading MusicGen Large (3.3B)...")
model = MusicGen.get_pretrained('facebook/musicgen-large')

# Set generation parameters
model.set_generation_params(duration=8)  # 8 seconds

# Generate music!
print("Generating music...")
descriptions = [
    "happy upbeat electronic dance music",
    "sad melancholic piano ballad"
]

wav = model.generate(descriptions)

# Save outputs
from audiocraft.data.audio import audio_write
for idx, one_wav in enumerate(wav):
    # Save as WAV file
    audio_write(
        f'results/test_output_{idx}',
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness"
    )

print("✅ Music generated! Check results/test_output_*.wav")
```

**Save this as** `test_musicgen.py` and run:
```bash
python3 test_musicgen.py
```

---

## Step 6: Extract Your First Activations

```python
from audiocraft.models import MusicGen
import torch

model = MusicGen.get_pretrained('facebook/musicgen-large')

# Storage for activations
activations = {}

def get_activation(name):
    """Hook function to capture layer outputs"""
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Register hooks on specific layers
# MusicGen Large has 24 transformer layers
layers_to_probe = [0, 6, 12, 18, 24]

for layer_idx in layers_to_probe:
    layer = model.lm.layers[layer_idx]
    layer.register_forward_hook(get_activation(f'layer_{layer_idx}'))

# Generate with hooks active
print("Generating with activation capture...")
prompts = ["happy energetic music"]
model.set_generation_params(duration=8)
wav = model.generate(prompts)

# Inspect activations
print("\nCaptured Activations:")
for name, activation in activations.items():
    print(f"{name}: shape = {activation.shape}")
    # Expected shape: [batch_size, sequence_length, d_model]

# Save activations for later analysis
torch.save(activations, 'results/first_activations.pt')
print("\n✅ Activations saved to results/first_activations.pt")
```

**Save this as** `test_activations.py` and run:
```bash
python3 test_activations.py
```

---

## Step 7: Start Learning!

You're now set up! Here's what to do next:

### Week 1-2: Learn Mechanistic Interpretability
1. Read the [Phase 0 Roadmap](docs/phase0_roadmap.md)
2. Start with ARENA tutorials on superposition
3. Complete SAE exercises

### Week 3-4: Explore MusicGen
1. Generate 20-30 music samples
2. Extract activations from all layers
3. Visualize activation patterns

### Week 5-6: Prepare Datasets
```bash
# Download emotion-labeled music (coming soon)
python3 scripts/download_datasets.py
```

---

## Troubleshooting

### Issue: `ImportError: No module named 'audiocraft'`
**Solution**:
```bash
pip install audiocraft
# or
pip install git+https://github.com/facebookresearch/audiocraft.git
```

### Issue: `RuntimeError: CUDA out of memory`
**Solutions**:
1. Use smaller model: `musicgen-medium` or `musicgen-small`
2. Reduce generation duration: `model.set_generation_params(duration=5)`
3. Generate fewer samples at once
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Issue: FFmpeg not found
**Solutions**:
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

### Issue: Very slow on Apple Silicon
**Solutions**:
1. Ensure you're using MPS backend:
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```
2. Close other applications to free RAM
3. Consider using smaller model or cloud GPU

### Issue: `ModuleNotFoundError: No module named 'laion_clap'`
**Solution**:
```bash
# CLAP is optional for basic testing
pip install laion-clap
```

---

## GPU Memory Requirements

| Model | Parameters | VRAM Needed | Generation Time (8s) |
|-------|-----------|-------------|---------------------|
| Small | 300M | 8 GB | ~10s |
| Medium | 1.5B | 16 GB | ~20s |
| **Large** | **3.3B** | **24 GB** | **~30s** |
| Stereo Large | 3.3B | 24 GB | ~30s |

**Note**: Times are approximate for A100 GPU. Your hardware may vary.

---

## Cloud GPU Options

If you don't have a local GPU with 24GB+ VRAM:

### Google Colab Pro ($10/month)
- V100 (16GB) or A100 (40GB) GPUs
- Can run Large model on A100
- Good for experiments

### Paperspace Gradient (pay-per-use)
- $0.51/hr for A4000 (16GB)
- $2.30/hr for A100 (80GB)
- Good for longer experiments

### Lambda Labs (dedicated)
- $1.10/hr for A10 (24GB)
- Great for sustained research

---

## Next Steps

1. ✅ Everything installed? → Start [Phase 0 Learning Roadmap](docs/phase0_roadmap.md)
2. ✅ Generated first music? → Experiment with different prompts
3. ✅ Captured activations? → Visualize them in a Jupyter notebook
4. ✅ Questions? → Check the main [README.md](README.md)

---

**Remember**: Take your time in Phase 0. Deep understanding of the fundamentals will make your research much more productive!
