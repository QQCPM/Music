# Project Navigation Guide

**Quick links**: [START_HERE](../START_HERE.md) | [README](../README.md) | [Plan](../PHASE0_COMPLETE_PLAN.md)

---

## 📁 File Structure (At a Glance)

```
MusicGen/
│
├─ START_HERE.md           ⭐ Read this first
├─ README.md               📖 Project overview
├─ PHASE0_COMPLETE_PLAN.md 📋 Week-by-week action plan
│
├─ ACTIVATION_EXTRACTION_FIX.md  🔧 Bug fix details
├─ DEBUG_SUMMARY.md              📝 Debug session log
├─ CODEBASE_CLEANUP_SUMMARY.md   🧹 Cleanup documentation
│
├─ src/utils/              💻 Core code
│  ├─ activation_utils.py  → Extract activations (FIXED)
│  ├─ audio_utils.py       → Audio processing
│  └─ visualization_utils.py → Plotting
│
├─ test_fixed_extractor.py     ✅ Main validation test
├─ debug_activation_extraction.py 🔍 Diagnostic tool
│
├─ notebooks/              📓 Interactive exploration
├─ docs/                   📚 Learning resources
├─ scripts/                ⚙️ Setup utilities
├─ results/                📊 Generated data
└─ archive/                📦 Old documentation
```

---

## 🎯 Quick Access by Goal

### I want to...

**Start working on this project**
→ [START_HERE.md](../START_HERE.md)

**Understand what this research is about**
→ [README.md](../README.md)

**Know what to do next**
→ [PHASE0_COMPLETE_PLAN.md](../PHASE0_COMPLETE_PLAN.md)

**Understand the activation bug that was fixed**
→ [ACTIVATION_EXTRACTION_FIX.md](../ACTIVATION_EXTRACTION_FIX.md)

**See how the bug was debugged**
→ [DEBUG_SUMMARY.md](../DEBUG_SUMMARY.md)

**Learn mechanistic interpretability**
→ [docs/phase0_roadmap.md](../docs/phase0_roadmap.md)

**Test if everything works**
→ Run `python3 test_fixed_extractor.py`

**Extract activations from MusicGen**
→ See [src/utils/activation_utils.py](../src/utils/activation_utils.py)

**Find old documentation**
→ Check `archive/` directory

---

## 📊 Documentation Hierarchy

```
Level 1: Entry Points
├─ START_HERE.md (quick orientation)
└─ README.md (full overview)

Level 2: Plans & Guides
├─ PHASE0_COMPLETE_PLAN.md (3-4 week plan)
└─ docs/phase0_roadmap.md (8-week learning)

Level 3: Technical Deep Dives
├─ ACTIVATION_EXTRACTION_FIX.md (bug analysis)
├─ DEBUG_SUMMARY.md (debugging process)
└─ CODEBASE_CLEANUP_SUMMARY.md (organization)
```

---

## 🔧 Active vs. Archived

### Active Documentation (Read These)

Located in root directory:
- START_HERE.md
- README.md
- PHASE0_COMPLETE_PLAN.md
- ACTIVATION_EXTRACTION_FIX.md
- DEBUG_SUMMARY.md
- CODEBASE_CLEANUP_SUMMARY.md

Located in docs/:
- phase0_roadmap.md

### Archived Documentation (Historical Reference)

Located in archive/:
- GETTING_STARTED.md
- SETUP_COMPLETE.md
- QUICKSTART.md
- FFMPEG_FIX_SUMMARY.md
- ARCHITECTURE_FIX.md
- FFMPEG_SETUP.md

---

## 🧪 Test Scripts

**Primary**: `test_fixed_extractor.py`
- Validates activation extraction
- Tests emotion differentiation
- **Run this first**

**Diagnostic**: `debug_activation_extraction.py`
- Deep debugging tool
- Traces generation process
- Use when troubleshooting

**Archived**: `archive/old_tests/`
- Old validation scripts
- Preserved for reference
- Not needed for current work

---

## 💡 Common Tasks

### Validate Setup
```bash
python3 test_fixed_extractor.py
```

### Extract Activations
```python
from audiocraft.models import MusicGen
from src.utils.activation_utils import ActivationExtractor

model = MusicGen.get_pretrained('facebook/musicgen-large')
extractor = ActivationExtractor(model, layers=[12, 24])
wav = extractor.generate(["happy music"])
acts = extractor.get_activations(concatenate=True)
```

### Compare Emotions
See [START_HERE.md](../START_HERE.md#quick-reference)

---

**Last updated**: Oct 7, 2024
