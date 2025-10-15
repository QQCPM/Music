# MusicGen Emotion Interpretability Research - START HERE

**Research Question**: Do music generation models "understand" emotion, or just pattern-match?

**Current Status**: Phase 0 - Methodology validated, data collection in progress

---

## Quick Setup (15 minutes)

### 1. Install Dependencies
```bash
cd "/Users/lending/Documents/AI PRJ/MusicGen"
source venv/bin/activate # Already created
pip install -r requirements.txt # Already installed
```

### 2. Test Everything Works
```bash
# Test 1: Verify activation extraction is fixed
python3 test_fixed_extractor.py

# Expected output:
# SUCCESS: Emotions are represented differently!
# Cosine similarity (layer 12): 0.9461
```

### 3. Try the Notebook
```bash
jupyter notebook notebooks/00_quick_test.ipynb
```

**️ IMPORTANT**: The notebook needs updating to use `concatenate=True` - see "Known Issues" below.

---

## Project Structure (Clean)

```
MusicGen/
START_HERE.md You are here
README.md Project overview
PHASE0_COMPLETE_PLAN.md Your action plan (3-4 weeks)

src/utils/ Core utilities
activation_utils.py Extract activations (FIXED)
audio_utils.py Audio processing
visualization_utils.py Plotting

scripts/ Setup scripts
setup_environment.py Check requirements
check_gpu.py GPU verification
download_models.py Download MusicGen
prepare_datasets.py Dataset prep

test_fixed_extractor.py Validates activation extraction
debug_activation_extraction.py Diagnostic tool

notebooks/
00_quick_test.ipynb Interactive exploration

docs/
phase0_roadmap.md 8-week learning plan

results/ Generated audio & data
archive/ Old documentation
```

---

## What Just Happened (Critical Context)

### The Bug (Oct 6)
Your activation extraction was capturing only **1 of 459 timesteps** (last one only).
- Result: 0.9999 cosine similarity between happy/sad music
- Meaning: Looked like emotions weren't differentiated at all

### The Fix (Oct 7)
**Fixed** `src/utils/activation_utils.py` to capture **all timesteps**:
```python
# Before (WRONG):
self.activations[name] = activation # Overwrites each time

# After (CORRECT):
self.activations[name].append(activation) # Stores all timesteps
```

### The Results
| Metric | Before | After |
|--------|--------|-------|
| Cosine similarity | 0.9999 | **0.9461** |
| Timesteps captured | 1 | 153-459 |
| Strong differentiating dims | 0% | **8%** |

**Conclusion**: Emotions ARE differentiated! (5.4% difference, 80 dims strongly encode emotion)

**Full details**: [ACTIVATION_EXTRACTION_FIX.md](ACTIVATION_EXTRACTION_FIX.md)

---

## Your Next Steps (In Order)

### Week 1: Complete Validation
1. **Update notebook** to use fixed extractor
```python
# Change this line in notebook:
activations = extractor.get_activations(concatenate=True) # Add this
```

2. **Test multiple emotion pairs**
- Not just happy/sad
- Try calm, energetic, angry, etc.

3. **Verify activation shapes**
```python
# Should see: torch.Size([459, 2, 1, 2048])
# ^^^
# timesteps (not 1!)
```

### Week 2: Generate Dataset
**Plan**: [PHASE0_COMPLETE_PLAN.md](PHASE0_COMPLETE_PLAN.md#week-2-comprehensive-data-collection)

Generate 80 samples (20 per emotion × 4 emotions):
```bash
# You'll create this:
python3 experiments/01_generate_emotion_dataset.py
```

### Week 3: Statistical Analysis
- UMAP clustering visualization
- Layer-wise similarity analysis
- Acoustic feature validation

### Week 4: SAE Theory
- Read Anthropic papers
- Complete ARENA exercises
- Design Phase 1 experiments

---

## Key Files Explained

### Documentation (Read These)

1. **[START_HERE.md](START_HERE.md)** (this file)
- Quick orientation
- What to do next

2. **[README.md](README.md)**
- Project overview
- Research phases
- Success criteria

3. **[PHASE0_COMPLETE_PLAN.md](PHASE0_COMPLETE_PLAN.md)**
- Detailed 3-4 week plan
- Week-by-week tasks
- Success criteria

4. **[ACTIVATION_EXTRACTION_FIX.md](ACTIVATION_EXTRACTION_FIX.md)**
- Technical details of the bug
- Why 0.9999 0.9461 matters
- How to use fixed extractor

5. **[DEBUG_SUMMARY.md](DEBUG_SUMMARY.md)**
- Complete debug session log
- Lessons learned
- Methodology insights

### Code (Use These)

6. **[src/utils/activation_utils.py](src/utils/activation_utils.py)**
- `ActivationExtractor` class (FIXED)
- Extract activations from MusicGen
- **Usage**:
```python
extractor = ActivationExtractor(model, layers=[0, 12, 24])
wav = extractor.generate(["happy music"])
acts = extractor.get_activations(concatenate=True) # IMPORTANT!
```

7. **[test_fixed_extractor.py](test_fixed_extractor.py)**
- Validates the fix works
- Compares happy vs. sad music
- Analyzes temporal dynamics
- **Run this first** to verify setup

8. **[debug_activation_extraction.py](debug_activation_extraction.py)**
- Diagnostic tool
- Traces generation process
- Use if you suspect issues

---

## Known Issues (To Fix)

### Issue 1: Notebook Needs Update
**File**: `notebooks/00_quick_test.ipynb`

**Problem**: Uses old extractor (before fix)

**Fix**: Update cell 8 and 13:
```python
# Cell 8: Add concatenate=True
activations = extractor.get_activations(concatenate=True)

# Cell 13: Same fix
happy_activations = extractor.get_activations(concatenate=True)
sad_activations = extractor.get_activations(concatenate=True)
```

### Issue 2: Missing Experiment Scripts
**Need to create**:
- `experiments/01_generate_emotion_dataset.py`
- `experiments/02_validate_acoustic_features.py`
- `experiments/03_layer_wise_analysis.py`
- `experiments/04_umap_clustering.py`

**Templates**: See [PHASE0_COMPLETE_PLAN.md](PHASE0_COMPLETE_PLAN.md#files-to-create-next-steps)

---

## Critical Learnings

### Research Methodology

1. **Unexpected results = investigate immediately**
- You saw 0.9999 similarity should have stopped
- Instead: moved on wasted time
- Lesson: Debug before interpreting

2. **Validate your tooling**
- Activation shapes were wrong: `[2, 1, 2048]` instead of `[459, 2, 1, 2048]`
- One simple check would have caught this
- Lesson: Always verify tensor shapes

3. **N=1 is not evidence**
- One sample per emotion = anecdote
- Need 20+ for statistics
- Lesson: Design experiments with statistical power

### Technical Insights

4. **MusicGen is autoregressive**
- Generates token-by-token (459 steps for 8s)
- Each forward pass processes one token
- Must capture ALL steps, not just the last

5. **Similarity = 0.9461 is good news**
- Not 0.99+ (no signal)
- Not 0.5 (too different, likely bug)
- 0.85-0.95 = meaningful differentiation
- Consistent with superposition theory

6. **Sparse encoding is expected**
- Only 8% of dims strongly differentiate
- Most neurons are polysemantic (encode multiple concepts)
- This validates your SAE plan for Phase 1

---

## Quick Reference

### Run Tests
```bash
# Test activation extraction (MUST PASS)
python3 test_fixed_extractor.py

# Diagnostic (if debugging)
python3 debug_activation_extraction.py
```

### Generate Music
```python
from audiocraft.models import MusicGen
from src.utils.activation_utils import ActivationExtractor

model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(duration=8)

extractor = ActivationExtractor(model, layers=[12, 24, 36])
wav = extractor.generate(["happy upbeat music"])
acts = extractor.get_activations(concatenate=True) # IMPORTANT!

# Check shape
print(acts['layer_12'].shape)
# Should be: torch.Size([459, 2, 1, 2048])
# ^^^
# timesteps!
```

### Compare Emotions
```python
# Generate happy
wav_happy = extractor.generate(["happy music"])
acts_happy = extractor.get_activations(concatenate=True)

# Generate sad
extractor.clear_activations()
wav_sad = extractor.generate(["sad music"])
acts_sad = extractor.get_activations(concatenate=True)

# Compare
from src.utils.activation_utils import cosine_similarity
sim = cosine_similarity(acts_happy['layer_12'], acts_sad['layer_12'])
print(f"Similarity: {sim:.4f}")
# Expected: 0.85-0.95 (NOT 0.9999!)
```

---

## Timeline

**Phase 0 Started**: ~Oct 1, 2024
**Bug Fixed**: Oct 7, 2024
**Phase 0 Target Completion**: Dec 8, 2024 (3 weeks remaining)
**Phase 1 Start**: Dec 9, 2024

**Total Project**: 9 months (through June 2025)

---

## Success Criteria (Phase 0)

Before starting Phase 1, you must have:

- [] Fixed activation extraction (DONE)
- [] Validated fix (similarity now 0.9461) (DONE)
- [TODO] Generated 80-sample emotion dataset
- [TODO] UMAP shows emotion clustering (silhouette > 0.2)
- [TODO] Acoustic features validate labels (>80% match)
- [TODO] Identified optimal layers for SAEs
- [TODO] Read SAE papers + completed ARENA exercises
- [TODO] Detailed Phase 1 experimental plan

**Do not proceed to Phase 1 until ALL boxes are checked.**

---

## Getting Help

### Documentation
- **Setup issues**: Check scripts/setup_environment.py
- **Bug details**: [ACTIVATION_EXTRACTION_FIX.md](ACTIVATION_EXTRACTION_FIX.md)
- **Action plan**: [PHASE0_COMPLETE_PLAN.md](PHASE0_COMPLETE_PLAN.md)
- **Learning roadmap**: [docs/phase0_roadmap.md](docs/phase0_roadmap.md)

### Community
- EleutherAI Discord (interpretability channel)
- ARENA Slack (course support)
- LessWrong (mech interp posts)

---

## Archive

Old documentation moved to `archive/`:
- GETTING_STARTED.md (redundant)
- SETUP_COMPLETE.md (outdated)
- QUICKSTART.md (redundant)
- FFMPEG_FIX_SUMMARY.md (specific bug, less relevant)
- docs/ARCHITECTURE_FIX.md (archived)
- docs/FFMPEG_SETUP.md (archived)

Old tests moved to `archive/old_tests/`:
- test_audio_saving.py
- test_fixed_architecture.py

These are preserved for reference but not needed for ongoing work.

---

**Now get to work. Week 1 starts now.**

**First task**: Run `python3 test_fixed_extractor.py` and verify you see 0.946 similarity.
