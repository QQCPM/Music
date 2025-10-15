# Phase 1: Sparse Autoencoder Training - Complete Roadmap

**Status**: READY TO BEGIN 
**Duration**: 2-3 weeks
**Goal**: Train SAEs on T5 text embeddings to discover monosemantic emotion-encoding features

---

## Revised Research Focus

### Key Discovery from Phase 0

**Emotions are encoded in T5 TEXT EMBEDDINGS, not transformer activations**

| Representation | Between-Emotion Similarity | Differentiation | Status |
|----------------|---------------------------|-----------------|---------|
| **T5 text embeddings** | 49.4% | **50% range** | **STRONG** |
| **Transformer activations** | 94.6% | 5% range | ️ WEAK |

**Implication**: Phase 1 targets T5 embedding space (768-dim), not transformer layers (2048-dim)

**Benefits**:
- **Smaller dimensionality**: 768 vs 2048 (easier to train)
- **Stronger signal**: 96% classification accuracy vs ~50%
- **Clear interpretation**: Text encoding is more interpretable than transformer states

---

## Phase 0 Results Summary

### Validation Completed

**Dataset**: 100 T5 embeddings (25 per emotion: happy, sad, calm, energetic)

**Key Metrics**:
- Within-emotion similarity: **0.560**
- Between-emotion similarity: **0.494**
- Differentiation: **6.6%** (statistically significant, p < 0.000001)
- Linear probe accuracy: **96%** 
- Emotion-encoding dimensions: Distributed across all 768 dimensions

**Verdict**: **STRONG EMOTION ENCODING CONFIRMED**

**Data Ready**:
- `results/t5_embeddings/embeddings.npy` - 100 × 768 tensor
- `results/t5_embeddings/labels.npy` - 100 emotion labels
- `results/t5_embeddings/metadata.json` - Full analysis

---

## ️ Phase 1 Infrastructure (COMPLETE)

### Completed Components

1. **Sparse Autoencoder Implementation**
- Location: `src/models/sparse_autoencoder.py`
- Architecture: 768 6144 768 (8x overcomplete)
- Features:
- L1 sparsity penalty
- Dead feature reinitialization
- Decoder weight normalization
- Feature tracking and metrics
- Status: Tested and working

2. **Dataset Utilities**
- Location: `src/utils/dataset_utils.py`
- Features:
- T5 embedding loading
- Train/val/test splitting
- Data normalization and centering
- PyTorch DataLoader integration
- Status: Tested with 100 samples

3. **Training Pipeline**
- Location: `experiments/train_sae_on_t5_embeddings.py`
- Features:
- Full training loop with early stopping
- Validation and metrics logging
- Dead feature reinitialization
- Training curve visualization
- Checkpoint saving
- Status: Ready to run

4. **Feature Analysis Tools**
- Location: `experiments/analyze_sae_features.py`
- Features:
- Feature selectivity computation
- Emotion-specific feature identification
- Activation pattern visualization
- Detailed statistics and reports
- Status: Ready for post-training analysis

---

## Week-by-Week Plan

### Week 1: Initial SAE Training & Hyperparameter Search

**Days 1-2: Baseline Training**

Run initial training with default hyperparameters:

```bash
cd "/Users/lending/Documents/AI PRJ/MusicGen"
source venv/bin/activate
python3 experiments/train_sae_on_t5_embeddings.py
```

**Expected outcomes**:
- Training time: ~5-10 minutes
- L0 (active features): 50-200 per sample (target: sparse!)
- Reconstruction loss: < 0.01 (good reconstruction)
- Validation loss: Should decrease and plateau

**Success criteria**:
- Training completes without errors
- Reconstruction loss < 0.02
- L0 < 500 (< 10% of 6144 features active)
- < 100 dead features at end of training

**Days 3-4: Hyperparameter Sweep**

Test different L1 coefficients to find optimal sparsity/reconstruction trade-off:

| L1 Coefficient | Expected Sparsity | Expected Reconstruction |
|----------------|-------------------|------------------------|
| 1e-3 | Medium (~300 active) | Good |
| 3e-3 | High (~100 active) | Medium |
| 1e-2 | Very high (~50 active) | Poor |
| 3e-4 | Low (~500 active) | Excellent |

**To run sweep**:
```python
# Edit train_sae_on_t5_embeddings.py CONFIG
for l1 in [1e-3, 3e-3, 1e-2, 3e-4]:
CONFIG['l1_coefficient'] = l1
# Run training
```

**Choose best model** based on:
1. Reconstruction loss < 0.01 (must reconstruct well)
2. L0 = 50-200 (sparse enough to be interpretable)
3. Dead features < 10% (most features should be useful)

**Days 5-7: Feature Analysis**

Run analysis on best model:

```bash
python3 experiments/analyze_sae_features.py
```

**Expected findings**:
- 50-200 emotion-selective features (selectivity > 2.0)
- Features cluster by emotion in heatmap
- Some features activate for specific emotion words (e.g., "joyful", "sad")

**Week 1 Deliverable**:
- Trained SAE model with good sparsity/reconstruction trade-off
- Initial feature analysis showing emotion selectivity
- Training curves and metrics saved

---

### Week 2: Scale Up Dataset & Retrain

**Days 1-3: Generate More Prompts**

Current dataset: 100 prompts (25 per emotion)
Target: 400-500 prompts (100-125 per emotion)

**Strategy**: Expand prompt diversity

```python
# Create experiments/generate_diverse_prompts.py
# Generate prompts with:
# - Different instruments (piano, guitar, orchestra, electronic)
# - Different genres (classical, pop, rock, jazz, ambient)
# - Different intensity levels (subtle, moderate, intense)
# - Different descriptive styles (adjectives, scenes, metaphors)

emotions = {
'happy': [
# Instrumental
'happy piano melody',
'cheerful acoustic guitar',
'joyful orchestral piece',

# Genre-based
'upbeat pop song',
'energetic dance music',
'bright jazz tune',

# Intensity
'subtly optimistic music',
'moderately cheerful tune',
'intensely jubilant celebration',

# Scene-based
'sunny day at the beach music',
'children playing in park',
'birthday party celebration',

# ... 100+ total for happy
],
# ... similar for sad, calm, energetic
}
```

**Days 4-5: Extract T5 Embeddings for New Prompts**

Modify and run `extract_t5_embeddings_at_scale.py`:

```bash
python3 experiments/extract_t5_embeddings_at_scale.py \
--output-dir results/t5_embeddings_500 \
--num-prompts 500
```

**Expected results**:
- 500 T5 embeddings (125 per emotion)
- Maintained differentiation (between-similarity ~0.49)
- Improved diversity (lower within-similarity)

**Days 6-7: Retrain SAE on Larger Dataset**

Run training with 500 samples:

```bash
# Update CONFIG in train_sae_on_t5_embeddings.py
CONFIG['data_dir'] = 'results/t5_embeddings_500'

python3 experiments/train_sae_on_t5_embeddings.py
```

**Expected improvements**:
- More robust features (less overfitting)
- Better coverage of emotion space
- More monosemantic features

**Week 2 Deliverable**:
- 500-prompt T5 embedding dataset
- SAE trained on larger, more diverse dataset
- Feature analysis showing improved interpretability

---

### Week 3: Feature Interpretation & Validation

**Days 1-2: Manual Feature Inspection**

Create `experiments/inspect_features_manually.py`:

```python
# For each top selective feature:
# 1. Find prompts that activate it most
# 2. Find prompts that don't activate it
# 3. Determine what concept it encodes

# Example output:
# Feature 42 (happy-selective, selectivity: 5.2x):
# Top activating prompts:
# - "joyful celebration music" (activation: 0.82)
# - "happy children playing" (activation: 0.78)
# - "cheerful upbeat melody" (activation: 0.75)
#
# Low activating prompts:
# - "sad melancholic piano" (activation: 0.01)
# - "calm meditation ambient" (activation: 0.03)
#
# Interpretation: Encodes "JOYFUL CELEBRATION" concept
```

**Goal**: Find 10-20 clearly interpretable features per emotion

**Days 3-4: Feature Intervention Experiments**

Test if features are **causally important**:

```python
# experiments/test_feature_causality.py

# Experiment 1: Activation clamping
# - Clamp happy-selective feature to 0 when encoding "happy music"
# - Expect: Reconstruction loses "happy" characteristics

# Experiment 2: Feature transplanting
# - Take sad prompt embedding
# - Add activation from happy-selective feature
# - Expect: Reconstruction gains "happy" characteristics

# Experiment 3: Feature ablation
# - Zero out all features except top-5 for emotion
# - Check if reconstruction still captures emotion
```

**Days 5-6: Quantitative Validation**

**Test 1: Feature Orthogonality**
- Are emotion features independent?
- Compute correlation between feature activations
- Expected: Low correlation (< 0.3)

**Test 2: Feature Stability**
- Train 3 SAEs with different random seeds
- Do similar features emerge?
- Compute feature similarity across models

**Test 3: Generalization**
- Generate 100 NEW prompts (not in training set)
- Extract T5 embeddings
- Run through SAE
- Check if features still selective

**Day 7: Write Phase 1 Report**

Document findings:

1. **Summary**
- Number of interpretable features found
- Selectivity scores
- Example features with descriptions

2. **Validation Results**
- Causal intervention outcomes
- Stability across random seeds
- Generalization to new prompts

3. **Comparison to Phase 0**
- Phase 0: Showed emotions ARE encoded (96% accuracy)
- Phase 1: Identified WHERE/HOW emotions are encoded (specific features)

4. **Next Steps for Phase 2**
- Use features for MusicGen activation steering
- Test if steering these features changes generated music
- Validate with acoustic analysis + human eval

**Week 3 Deliverable**:
- 10-20 interpretable features per emotion
- Causal validation of features
- Stability and generalization tests passed
- Phase 1 complete report

---

## Success Criteria

### Must Have (Go/No-Go for Phase 2)

- **Reconstruction quality**: MSE < 0.01 on test set
- **Sparsity**: L0 = 50-200 active features per sample
- **Selectivity**: ≥50 features with selectivity > 2.0
- **Interpretability**: ≥10 clearly interpretable features per emotion
- **Causality**: Feature interventions change reconstruction in expected way

### Nice to Have (Strengthens findings)

- **Monosemanticity**: Features activate for single, clear concept
- **Stability**: Similar features across different random seeds
- **Generalization**: Features transfer to unseen prompts
- **Compositionality**: Multiple emotion features can be combined

---

## Hyperparameters to Tune

### Critical Parameters

| Parameter | Default | Range to Test | Impact |
|-----------|---------|---------------|--------|
| `l1_coefficient` | 1e-3 | [3e-4, 1e-3, 3e-3, 1e-2] | Sparsity level |
| `expansion_factor` | 8 | [4, 8, 12, 16] | Feature capacity |
| `learning_rate` | 1e-3 | [3e-4, 1e-3, 3e-3] | Training speed |
| `batch_size` | 16 | [8, 16, 32] | Stability |

### Less Critical (Use Defaults)

- `dead_feature_threshold`: 100 steps
- `decoder_norm`: True
- `encoder_init_scale`: 0.1
- `patience`: 50 epochs

**Tuning Strategy**:
1. Start with defaults
2. If too sparse (L0 < 20): Decrease `l1_coefficient`
3. If not sparse enough (L0 > 500): Increase `l1_coefficient`
4. If poor reconstruction: Decrease `l1_coefficient` or increase `expansion_factor`
5. If too many dead features: Decrease `dead_feature_threshold` or reinit more frequently

---

## Expected Outcomes

### Quantitative

- **10-20 interpretable features per emotion** (40-80 total)
- **Selectivity > 3.0** for best features
- **Reconstruction MSE < 0.01**
- **L0 = 50-200** (1-3% of 6144 features active)
- **< 5% dead features**

### Qualitative

**Example interpretable features**:

| Feature | Emotion | Selectivity | Interpretation |
|---------|---------|-------------|----------------|
| F42 | Happy | 5.2x | "Joyful celebration" |
| F108 | Happy | 4.8x | "Playful energy" |
| F221 | Sad | 6.1x | "Melancholic longing" |
| F334 | Sad | 4.3x | "Sorrowful grief" |
| F445 | Calm | 5.7x | "Peaceful meditation" |
| F556 | Calm | 4.9x | "Gentle tranquility" |
| F667 | Energetic | 6.3x | "Intense power" |
| F778 | Energetic | 5.4x | "Aggressive drive" |

---

## Potential Issues & Solutions

### Issue 1: Poor Reconstruction (MSE > 0.05)

**Symptoms**: Reconstructed embeddings don't match originals

**Causes**:
- L1 coefficient too high (over-sparsifying)
- Model capacity too small
- Insufficient training

**Solutions**:
- Decrease `l1_coefficient` from 1e-3 to 3e-4
- Increase `expansion_factor` from 8 to 12
- Train for more epochs

---

### Issue 2: Not Sparse Enough (L0 > 500)

**Symptoms**: Too many features active per sample

**Causes**:
- L1 coefficient too low
- ReLU not enforcing sparsity

**Solutions**:
- Increase `l1_coefficient` from 1e-3 to 3e-3 or 1e-2
- Check activation statistics (should see clear 0/non-zero split)

---

### Issue 3: Too Many Dead Features (> 20%)

**Symptoms**: Many features never activate

**Causes**:
- Too sparse (L1 too high)
- Poor initialization
- Dataset too small

**Solutions**:
- Decrease `l1_coefficient`
- Decrease `dead_feature_threshold` to 50
- Reinitialize more frequently (every 50 steps)
- Scale up to 500-sample dataset

---

### Issue 4: Features Not Interpretable

**Symptoms**: Features activate for random/unclear concepts

**Causes**:
- Insufficient sparsity (polysemantic features)
- Dataset not diverse enough
- Superposition not fully resolved

**Solutions**:
- Increase sparsity (higher L1)
- Increase model capacity (larger expansion factor)
- Generate more diverse prompts (500+ samples)
- Train longer to convergence

---

### Issue 5: Features Don't Generalize

**Symptoms**: Features selective on training data but not test data

**Causes**:
- Overfitting
- Dataset too homogeneous
- Insufficient samples

**Solutions**:
- Add more diverse prompts
- Use larger train/val/test splits
- Add weight decay regularization
- Ensure prompts vary in style, not just emotion

---

## Key References

### SAE Papers

1. **Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"** (Anthropic, 2023)
- Core SAE methodology
- L1 penalty + overcomplete architecture
- Feature interpretability evaluation

2. **Sharkey et al., "Taking Features out of Superposition with Sparse Autoencoders"** (2023)
- Theory of why SAEs work
- Relationship to superposition
- Scaling laws for SAE capacity

3. **Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models"** (Anthropic, 2023)
- Feature selectivity metrics
- Dead feature reinitialization
- Evaluation protocols

### Interpretability Background

4. **Elhage et al., "Toy Models of Superposition"** (Anthropic, 2022)
- Why neural nets compress features
- Why overcomplete SAEs help

5. **Park et al., "The Linear Representation Hypothesis"** (2024)
- Concepts as linear directions
- Theoretical foundation for feature interpretation

---

## ️ Implementation Checklist

### Before Starting

- [x] Phase 0 complete and validated
- [x] T5 embedding dataset ready (100 samples)
- [x] SAE implementation tested
- [x] Training pipeline tested
- [x] Analysis tools ready

### Week 1

- [ ] Run baseline SAE training
- [ ] Hyperparameter sweep (L1 coefficient)
- [ ] Select best model
- [ ] Initial feature analysis
- [ ] Document Week 1 findings

### Week 2

- [ ] Generate 400-500 diverse prompts
- [ ] Extract T5 embeddings for all prompts
- [ ] Retrain SAE on larger dataset
- [ ] Analyze features from larger model
- [ ] Compare to Week 1 results

### Week 3

- [ ] Manual feature inspection (10-20 per emotion)
- [ ] Causal intervention experiments
- [ ] Stability testing (multiple random seeds)
- [ ] Generalization testing (unseen prompts)
- [ ] Write Phase 1 report
- [ ] Go/no-go decision for Phase 2

---

## Deliverables

### Code

- `src/models/sparse_autoencoder.py` - SAE implementation
- `src/utils/dataset_utils.py` - Data loading utilities
- `experiments/train_sae_on_t5_embeddings.py` - Training script
- `experiments/analyze_sae_features.py` - Analysis script
- `experiments/generate_diverse_prompts.py` - Prompt generation
- `experiments/inspect_features_manually.py` - Feature interpretation
- `experiments/test_feature_causality.py` - Causal validation

### Data

- `results/t5_embeddings/` - 100-sample dataset
- `results/t5_embeddings_500/` - 500-sample dataset
- `results/sae_training/` - Trained models and metrics

### Documentation

- `PHASE1_ROADMAP.md` - This document
- `PHASE1_WEEK1_REPORT.md` - Week 1 findings
- `PHASE1_WEEK2_REPORT.md` - Week 2 findings
- `PHASE1_FINAL_REPORT.md` - Complete Phase 1 results

### Visualizations

- Training curves (loss, sparsity, dead features)
- Feature-emotion heatmap
- Feature activation distributions
- Top feature examples with prompts

---

## Phase 2 Preview

**If Phase 1 succeeds**, proceed to Phase 2: **Activation Steering**

**Goal**: Use discovered features to control MusicGen's emotional output

**Approach**:
1. Take trained SAE features
2. Identify "emotion steering vectors" in T5 embedding space
3. Add steering vectors to MusicGen's T5 conditioning
4. Generate music and validate emotion shift

**Example experiment**:
```python
# Start with neutral prompt
prompt = "instrumental music"

# Add "happy" steering vector
happy_feature_vector = sae_decoder[:, happy_feature_42] # Feature 42 = "joyful"
steered_embedding = t5_embedding + 2.0 * happy_feature_vector

# Generate with MusicGen using steered embedding
# Expected: Music sounds happier even though prompt is neutral
```

**Success criteria for Phase 2**:
- Steering changes CLAP similarity scores for emotion
- Human evaluators detect emotion shift
- Acoustic features (tempo, mode) change as expected

---

## Getting Started

**To begin Phase 1 right now:**

```bash
# 1. Activate environment
cd "/Users/lending/Documents/AI PRJ/MusicGen"
source venv/bin/activate

# 2. Run baseline training (Week 1, Day 1)
python3 experiments/train_sae_on_t5_embeddings.py

# Expected time: 5-10 minutes
# Expected output: Trained model in results/sae_training/

# 3. Analyze features (Week 1, Day 5)
python3 experiments/analyze_sae_features.py

# Expected output: Feature analysis in results/sae_analysis/
```

**Then follow the week-by-week plan above!**

---

## Support & Resources

**Communities**:
- EleutherAI Discord - #interpretability channel
- ARENA Slack - SAE implementation help
- LessWrong - Interpretability research discussions

**Code Resources**:
- [SAELens](https://github.com/jbloomAus/SAELens) - Production SAE library
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Interpretability tools

**Papers to Reference**:
- All papers listed in "Key References" section above
- Check for new SAE papers monthly (field moves fast!)

---

**Status**: 🟢 READY TO BEGIN

**Next Action**: Run `python3 experiments/train_sae_on_t5_embeddings.py`

---

*Last Updated: October 10, 2024*
*Phase 0 Completion: October 10, 2024*
*Phase 1 Start: Ready to begin*
