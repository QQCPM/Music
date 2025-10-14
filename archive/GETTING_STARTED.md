# Getting Started with Your Research

Congratulations on setting up your MusicGen interpretability research project! This guide will help you take the first steps.

---

## ✅ What's Been Set Up

Your project now has:

1. **Complete directory structure** for organizing code, data, and results
2. **Comprehensive documentation**:
   - [README.md](README.md) - Project overview
   - [QUICKSTART.md](QUICKSTART.md) - Installation guide
   - [docs/phase0_roadmap.md](docs/phase0_roadmap.md) - Detailed 8-week learning plan
3. **Installation scripts**:
   - `scripts/setup_environment.py` - Check system requirements
   - `scripts/check_gpu.py` - Verify GPU setup
   - `scripts/download_models.py` - Download MusicGen models
   - `scripts/prepare_datasets.py` - Prepare emotion-labeled datasets
4. **Core utilities**:
   - `src/utils/activation_utils.py` - Extract and analyze activations
   - `src/utils/audio_utils.py` - Audio processing and feature extraction
   - `src/utils/visualization_utils.py` - Plotting and visualization
5. **Dependencies**: `requirements.txt` with all necessary packages

---

## 📋 Your Next Steps

### Step 1: Verify Your Environment (5 minutes)

```bash
# Check system requirements
python3 scripts/setup_environment.py
```

This will verify:
- ✅ Python 3.9+
- ✅ FFmpeg (needed for audio processing)
- ✅ Directory structure

If FFmpeg is missing:
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/

---

### Step 2: Install Dependencies (10-20 minutes)

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all packages
pip install -r requirements.txt
```

**Note**: This will download ~3-5GB of packages. Be patient!

**For GPU users**: If you have an NVIDIA GPU, install PyTorch with CUDA:
```bash
# Check which CUDA version you have: nvidia-smi
# Then install appropriate PyTorch version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For Apple Silicon users**: Standard PyTorch includes MPS (Metal) support automatically.

---

### Step 3: Verify GPU Setup (2 minutes)

```bash
python3 scripts/check_gpu.py
```

This will show:
- Your GPU type and VRAM
- Which MusicGen models you can run
- A quick PyTorch test

**Expected output** (example):
```
✅ CUDA GPU detected
   GPU 0: NVIDIA RTX 4090
   - VRAM: 24.00 GB

   Recommendations:
   ✅ Can run MusicGen Large (3.3B) ← Your goal!
```

---

### Step 4: Download MusicGen Large (10-30 minutes)

You wanted the **3.3B model** - here's how to get it:

```bash
# Interactive mode (recommended)
python3 scripts/download_models.py

# Or directly specify the model
python3 scripts/download_models.py large
```

**Storage needed**: ~13 GB for MusicGen Large

**Alternative**: If you want stereo audio (higher quality):
```bash
python3 scripts/download_models.py stereo-large
```

---

### Step 5: Test Your Setup (10 minutes)

Create a file called `test_setup.py`:

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

print("🎵 Testing MusicGen Large (3.3B)...")
print()

# Load model
print("Loading model (this may take a few minutes)...")
model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(duration=8)  # 8 seconds
print("✅ Model loaded!")
print()

# Generate music
print("Generating music samples...")
descriptions = [
    "happy upbeat electronic dance music",
    "sad melancholic piano ballad"
]

wav = model.generate(descriptions)

# Save
print("Saving audio files...")
for idx, one_wav in enumerate(wav):
    audio_write(
        f'results/test_{idx}',
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness"
    )

print()
print("✅ Test complete!")
print("   Check results/test_0.wav and results/test_1.wav")
print()
print("🎉 Everything is working! You're ready to start research.")
```

Run it:
```bash
python3 test_setup.py
```

**Expected time**: 30-60 seconds per sample (depending on GPU)

Listen to the generated files - they should sound like real music!

---

### Step 6: Create Sample Dataset (2 minutes)

```bash
python3 scripts/prepare_datasets.py --create-sample
```

This creates a small test dataset in `data/processed/sample_emotion_dataset.csv` with prompts organized by emotion (happy, sad, calm, energetic).

Later, you'll download real emotion-labeled music datasets (PMEmo, DEAM).

---

### Step 7: Extract Your First Activations (5 minutes)

Create `test_activations.py`:

```python
from audiocraft.models import MusicGen
from src.utils.activation_utils import ActivationExtractor
import torch

print("🔬 Testing activation extraction...")
print()

# Load model
model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(duration=8)

# Create extractor for specific layers
# MusicGen Large has 24 transformer layers
# We'll sample layers: 0, 6, 12, 18, 24
extractor = ActivationExtractor(model, layers=[0, 6, 12, 18, 24])

# Generate with activation capture
print("Generating with hooks active...")
prompts = ["happy energetic music", "sad melancholic music"]
wav = extractor.generate(prompts)

# Get activations
activations = extractor.get_activations()

print()
print("✅ Captured activations:")
for name, act in activations.items():
    print(f"   {name}: {act.shape}")
    # Expected: [batch_size, sequence_length, d_model]
    # For MusicGen Large: d_model = 2048

# Save for later analysis
extractor.save_activations('results/first_activations.pt')

print()
print("🎉 Activation extraction working!")
print("   Saved to: results/first_activations.pt")
```

Run it:
```bash
python3 test_activations.py
```

This confirms you can extract internal representations from MusicGen!

---

## 🎓 What to Do Next

### Option A: Start Learning (Recommended)

If you're new to mechanistic interpretability, **start with the learning phase**:

1. **Read** [docs/phase0_roadmap.md](docs/phase0_roadmap.md)
2. **Week 1-2**: Study ARENA tutorials on superposition and SAEs
3. **Week 3-4**: Explore MusicGen architecture and generate samples
4. **Week 5-6**: Prepare real datasets (PMEmo, DEAM)
5. **Week 7-8**: Literature deep dive

**Why this matters**: Understanding the fundamentals will make your research much more productive.

### Option B: Experiment First (Also Valid!)

If you prefer hands-on learning:

1. **Generate lots of music**: Try different prompts, explore what MusicGen can do
2. **Extract activations**: Hook into different layers, visualize patterns
3. **Analyze audio**: Use `src/utils/audio_utils.py` to extract features
4. **Read papers as needed**: When you hit questions, dive into the literature

**Both approaches work** - choose what fits your learning style!

---

## 📚 Key Resources

### Documentation in This Project
- [README.md](README.md) - Full project overview
- [QUICKSTART.md](QUICKSTART.md) - Installation details
- [docs/phase0_roadmap.md](docs/phase0_roadmap.md) - 8-week learning plan

### Papers to Read
1. **MusicGen**: [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
2. **Sparse Autoencoders**: [Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600)
3. **Activation Steering**: [Steering Language Models](https://arxiv.org/abs/2308.10248)
4. **Music + SAEs**: [Discovering Interpretable Concepts in Music Models](https://arxiv.org/abs/2505.18186) (May 2025)
5. **Music + Steering**: [Fine-Grained control over Music Generation](https://arxiv.org/abs/2506.10225) (June 2025)

### Learning Resources
- **ARENA**: https://arena3-chapter1-transformer-interp.streamlit.app/
- **Neel Nanda's Blog**: https://www.neelnanda.io/mechanistic-interpretability
- **Anthropic's Interpretability Work**: https://transformer-circuits.pub/

### Communities
- **EleutherAI Discord**: https://discord.gg/eleutherai (interpretability channel)
- **ARENA Slack**: For course support
- **LessWrong**: https://www.lesswrong.com (mech interp posts)

---

## 🎯 Your Research Goal

**Core Question**: Do music generation models "understand" emotion, or do they just correlate patterns?

**How You'll Answer It**:

1. **Phase 1** (Months 3-4): Does MusicGen build geometric representations of emotion?
   - Extract activations, train SAEs, analyze clustering

2. **Phase 2** (Months 5-6): Does it use human-like causal pathways?
   - Causal probing: tempo → energy → emotion
   - Compare to neuroscience findings

3. **Phase 3** (Months 7-9): Can we control emotion through activation steering?
   - If steering produces coherent music → model "understands"
   - If not → just pattern matching

**Either result is publishable** - the question itself is novel!

---

## 🚀 First Research Task

Here's a concrete first experiment (after setup):

### Experiment: "Do MusicGen's activations differ for happy vs. sad music?"

1. **Generate 10 samples** (5 happy, 5 sad) with activation extraction
2. **Extract activations** from layers [0, 6, 12, 18, 24]
3. **Compute statistics**: mean, std, sparsity for each layer
4. **Visualize**: Plot activation statistics by layer and emotion
5. **Analyze**: Do happy and sad music have different activation patterns? Which layers differ most?

**Time**: 1-2 hours
**Deliverable**: Plots showing activation differences
**Learning**: Hands-on experience with the full pipeline

---

## 🔧 Troubleshooting

### "CUDA out of memory"
- Use smaller model: `musicgen-medium` (1.5B) or `musicgen-small` (300M)
- Reduce duration: `model.set_generation_params(duration=5)`
- Clear cache: `torch.cuda.empty_cache()`

### "FFmpeg not found"
- Install FFmpeg (see Step 1)

### "Model download is slow"
- This is normal - models are 5-15GB
- Use `ctrl+C` to cancel and resume later (downloads resume automatically)

### "ImportError: No module named 'audiocraft'"
```bash
pip install audiocraft
# or
pip install git+https://github.com/facebookresearch/audiocraft.git
```

---

## 💡 Tips for Success

1. **Take your time** - Understanding > speed
2. **Document everything** - Keep a research journal
3. **Visualize often** - Plot activations, features, comparisons
4. **Ask questions** - Join Discord communities
5. **Stay grounded** - Connect theory to concrete experiments
6. **Iterate** - Start simple, add complexity gradually

---

## 🎉 You're Ready!

Everything is set up for serious research. The infrastructure is in place - now it's time to explore!

**Immediate next action**:
```bash
# Test everything works
python3 test_setup.py
python3 test_activations.py

# Then choose your path:
# Path A: Read docs/phase0_roadmap.md and start learning
# Path B: Start experimenting and learn as you go
```

**Remember**: This is a 6-9 month journey. No rush. Deep understanding leads to better research.

---

Good luck with your research! 🎵🔬🎉
