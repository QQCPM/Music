# Phase 1 Quick Start Guide

**Goal**: Train SAE to find emotion-encoding features in T5 text embeddings

**Time**: 3 weeks total, ~10 minutes per experiment

---

## Prerequisites (Complete)

- [x] Phase 0 validated: Emotions ARE encoded in T5 embeddings (96% accuracy)
- [x] T5 embedding dataset ready: 100 samples in `results/t5_embeddings/`
- [x] SAE implementation: `src/models/sparse_autoencoder.py`
- [x] Training pipeline: `experiments/train_sae_on_t5_embeddings.py`
- [x] Analysis tools: `experiments/analyze_sae_features.py`

---

## Week 1: Train Your First SAE (Start Here!)

### Step 1: Run Baseline Training (~10 minutes)

```bash
cd "/Users/lending/Documents/AI PRJ/MusicGen"
source venv/bin/activate
python3 experiments/train_sae_on_t5_embeddings.py
```

**What to expect**:
- Training progress bar with loss/sparsity metrics
- Model saved to `results/sae_training/sae_t5_exp8_l1e-03_[timestamp]/`
- Training curves plot saved

**Success indicators**:
- Reconstruction loss < 0.02
- L0 (active features) = 50-500
- Dead features < 100

---

### Step 2: Analyze Features (~2 minutes)

```bash
python3 experiments/analyze_sae_features.py
```

**What to expect**:
- Feature-emotion heatmap showing which features activate for which emotions
- List of emotion-selective features (selectivity > 2.0)
- Results saved to `results/sae_analysis/`

**Success indicators**:
- 50+ selective features found
- Features cluster by emotion in heatmap
- Top features have selectivity > 3.0

---

### Step 3: Hyperparameter Tuning

Edit `experiments/train_sae_on_t5_embeddings.py`:

```python
# Line 29: Try different L1 coefficients
CONFIG = {
'l1_coefficient': 3e-3, # Change this: 3e-4, 1e-3, 3e-3, 1e-2
# ... rest stays same
}
```

Run training 4 times with different values. Pick the one with:
- Best reconstruction (MSE < 0.01)
- Good sparsity (L0 = 50-200)
- Interpretable features (selectivity > 3.0)

---

## Week 1 Expected Results

### Quantitative
- 50-150 emotion-selective features
- Reconstruction MSE: 0.005-0.015
- L0: 100-300 active features per sample
- Selectivity (best features): 3.0-6.0x

### Qualitative
You should see features that activate strongly for specific emotions:

**Example**:
- Feature 42: Activates 80% for "happy" prompts, 10% for others **Selectivity: 5.3x**
- Feature 108: Activates 75% for "sad" prompts, 12% for others **Selectivity: 5.1x**

---

## Decision Point: End of Week 1

**GO to Week 2 if**:
- Found 50+ selective features
- Reconstruction loss < 0.02
- Features make semantic sense (activate for related prompts)

**ITERATE Week 1 if**:
- Too few selective features Increase L1 coefficient
- Poor reconstruction Decrease L1 coefficient or increase expansion factor
- Too many dead features Reinit more frequently or scale up dataset

---

## Week 2: Scale Up (Optional but Recommended)

Generate 400-500 diverse prompts to improve feature quality:

1. Create prompt generation script
2. Extract T5 embeddings for all prompts
3. Retrain SAE on larger dataset
4. Expect: More robust, interpretable features

*Detailed instructions in [PHASE1_ROADMAP.md](PHASE1_ROADMAP.md)*

---

## Week 3: Validate & Interpret

1. **Manual inspection**: Find what each top feature encodes
2. **Causal tests**: Clamp features and observe reconstruction changes
3. **Stability tests**: Train with different seeds, check if similar features emerge
4. **Final report**: Document findings and prepare for Phase 2

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'models'"

```bash
# Make sure you're in the project directory
cd "/Users/lending/Documents/AI PRJ/MusicGen"
source venv/bin/activate
```

---

### Training loss not decreasing

**Symptoms**: Loss stays constant or increases

**Fixes**:
1. Check learning rate (try 3e-4 instead of 1e-3)
2. Decrease L1 coefficient (try 3e-4 instead of 1e-3)
3. Check data loaded correctly (should see 70-80 train samples)

---

### All features are dead

**Symptoms**: `dead_features` count equals total features (6144)

**Fixes**:
1. Decrease L1 coefficient significantly (try 1e-4)
2. Check encoder initialization scale (should be 0.1)
3. Verify ReLU activation is working

---

### Features not selective

**Symptoms**: All selectivity scores < 2.0

**Fixes**:
1. Increase L1 coefficient to force more sparsity
2. Scale up to 500-sample dataset (more diverse data)
3. Increase expansion factor to 12x or 16x

---

## File Structure After Week 1

```
MusicGen/
PHASE1_ROADMAP.md Full roadmap
PHASE1_QUICKSTART.md This file

src/
models/
sparse_autoencoder.py SAE implementation 
utils/
dataset_utils.py Data loading 

experiments/
train_sae_on_t5_embeddings.py Training script 
analyze_sae_features.py Analysis script 
extract_t5_embeddings_at_scale.py Dataset creation 

results/
t5_embeddings/ Input data (100 samples) 
embeddings.npy
labels.npy
metadata.json

sae_training/ Training outputs 
sae_t5_exp8_l1e-03_[timestamp]/
best_model.pt
config.json
train_metrics.json
val_metrics.json
test_results.json
training_curves.png

sae_analysis/ Analysis outputs 
feature_emotion_heatmap.png
analysis_results.json
```

---

## Key Concepts

### Sparse Autoencoder (SAE)
- **Input**: 768-dim T5 embedding
- **Hidden**: 6144-dim (8x overcomplete)
- **Output**: 768-dim reconstruction
- **Goal**: Find sparse, interpretable features

### Sparsity (L0)
- Number of active (non-zero) features per sample
- Target: 50-200 (1-3% of 6144)
- Too high Not interpretable (polysemantic)
- Too low Poor reconstruction

### Selectivity
- How specific a feature is to one emotion
- Formula: `max(activation_rate) / mean(activation_rate)`
- Selectivity > 2.0 Feature is emotion-selective
- Selectivity > 4.0 Feature is highly selective

### Dead Features
- Features that never activate during training
- Caused by too much sparsity or poor initialization
- Solution: Reinitialize periodically during training

---

## Success Criteria for Phase 1

**Minimum (to proceed to Phase 2)**:
- 50+ emotion-selective features (selectivity > 2.0)
- Reconstruction MSE < 0.02
- 10+ clearly interpretable features per emotion

**Ideal**:
- 100+ selective features
- Reconstruction MSE < 0.01
- Selectivity > 4.0 for top features
- Features generalize to unseen prompts

---

## Next Steps After Phase 1

**Phase 2: Activation Steering** (Months 3-4)
- Use discovered features to control MusicGen
- Inject feature activations into T5 conditioning
- Generate music with steered emotions
- Validate with CLAP scores + human eval

**Phase 3: Causal Pathways** (Months 5-6)
- Map emotion acoustic features (tempo, mode, energy)
- Find causal pathways in MusicGen
- Compare to human music perception neuroscience

---

## ⏱️ Time Estimates

| Task | Time | Cumulative |
|------|------|------------|
| Setup (already done) | 0 min | 0 min |
| Baseline training | 10 min | 10 min |
| Feature analysis | 2 min | 12 min |
| Hyperparameter sweep (4 runs) | 40 min | 52 min |
| Best model analysis | 2 min | 54 min |
| **Week 1 Total** | **~1 hour** | |
| Scale up dataset | 30 min | |
| Retrain on 500 samples | 15 min | |
| **Week 2 Total** | **~1 hour** | |
| Manual interpretation | 2 hours | |
| Causal validation | 1 hour | |
| Report writing | 2 hours | |
| **Week 3 Total** | **~5 hours** | |
| **Phase 1 Total** | **~7 hours** | |

---

## Start Now!

```bash
cd "/Users/lending/Documents/AI PRJ/MusicGen"
source venv/bin/activate
python3 experiments/train_sae_on_t5_embeddings.py
```

Watch the progress bar. In 10 minutes, you'll have your first SAE trained! 

---

*For detailed technical information, see [PHASE1_ROADMAP.md](PHASE1_ROADMAP.md)*
*For Phase 0 results, see test_text_embeddings.py and extract_t5_embeddings_at_scale.py outputs*
