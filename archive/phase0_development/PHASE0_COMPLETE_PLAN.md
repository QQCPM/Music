# Phase 0 Completion Plan - Revised & Rigorous

**Status**: Bug fixed, now proceeding with proper methodology
**Timeline**: 3-4 weeks remaining
**Goal**: Validate methodology + build solid foundation for Phase 1

---

## What Just Happened (Critical Breakthrough)

### The Bug Discovery

Your activation extraction was **broken**:
- Capturing only **1 out of 459 timesteps** (0.2% of generation)
- This caused **0.9999 cosine similarity** (essentially identical)
- Made it impossible to measure emotion differentiation

### The Fix

**Fixed** `src/utils/activation_utils.py`:
- Now captures **ALL 153-459 timesteps** (100% of generation)
- Similarity dropped to **0.9461** (5.4% difference = meaningful signal!)
- Can now proceed with real analysis

### What This Teaches You

**This is what research looks like**:
1. Get unexpected result (0.9999 similarity)
2. **Don't ignore it** - investigate deeply
3. Find root cause (only capturing 1 timestep)
4. Fix methodology
5. Re-run with corrected approach
6. Interpret new results (0.9461 = emotions DO differentiate!)

You initially **skipped step 2** and moved on. **That's the mistake.**

---

## Revised Phase 0 Objectives (Achievable in 3-4 Weeks)

### Week 1: Validate Methodology ✅ (MOSTLY DONE)

**Completed**:
- [✅] Fix activation extraction (done!)
- [✅] Verify fix captures full sequence (done!)
- [✅] Confirm similarity drops below 0.99 (done! now 0.9461)

**Remaining**:
- [TODO] Update notebook to use fixed extractor
- [TODO] Test on multiple emotion pairs (not just happy/sad)
- [TODO] Document findings in research journal

**Time**: 2-3 hours

---

### Week 2: Comprehensive Data Collection

**Goal**: Generate robust dataset for analysis

**Tasks**:

1. **Generate 80 samples** (20 per emotion):
   ```python
   emotions = ['happy', 'sad', 'calm', 'energetic']
   prompts = {
       'happy': [
           "upbeat cheerful pop music",
           "joyful energetic dance music",
           "bright optimistic melody",
           # ... 17 more variations
       ],
       # ... same for sad, calm, energetic
   }
   ```

2. **Extract activations** from strategic layers:
   - Layer 0 (input)
   - Layer 12 (early-middle)
   - Layer 24 (middle)
   - Layer 36 (late-middle)
   - Layer 47 (output)

3. **Save efficiently**:
   ```python
   # Don't save all 459 timesteps for all layers - too big!
   # Strategy: Average over time for initial analysis
   for layer_name, activations in all_activations.items():
       # activations shape: [459, 2, 1, 2048]
       mean_activation = activations.mean(dim=0)  # [2, 1, 2048]
       save(mean_activation, f"sample_{i}_layer_{layer_name}.pt")
   ```

**Deliverable**:
- 80 audio files (results/emotion_dataset/)
- 80 × 5 = 400 activation tensors
- metadata.csv with prompts and labels

**Time**: 6-8 hours (mostly model inference time)

---

### Week 3: Statistical Validation

**Goal**: Prove that emotion differentiation is real and robust

#### Experiment 1: Layer-wise Similarity Analysis

```python
# For each layer, compute average similarity within vs. between emotions

# Within-emotion similarity (should be HIGH)
happy_samples = load_activations(emotion='happy', layer=12)
within_sim_happy = pairwise_cosine_similarity(happy_samples)
within_sim_happy_mean = within_sim_happy.mean()  # Expect ~0.95-0.98

# Between-emotion similarity (should be LOWER)
sad_samples = load_activations(emotion='sad', layer=12)
between_sim = cosine_similarity(happy_samples, sad_samples)
between_sim_mean = between_sim.mean()  # Expect ~0.90-0.95

# The gap = signal strength
signal_strength = within_sim_happy_mean - between_sim_mean
# Expect 0.02-0.08 (small but meaningful)
```

**Question**: Which layers show strongest emotion differentiation?

**Expected**: Middle layers (12-24) > early/late layers

#### Experiment 2: UMAP Emotion Clustering

```python
# Concatenate all activations from layer 12
all_layer12 = []
all_labels = []

for emotion in ['happy', 'sad', 'calm', 'energetic']:
    acts = load_activations(emotion=emotion, layer=12)
    # acts shape: [20 samples, 2, 1, 2048]
    acts_flat = acts.reshape(20, -1)  # [20, 4096]
    all_layer12.append(acts_flat)
    all_labels.extend([emotion] * 20)

# Stack and run UMAP
X = np.vstack(all_layer12)  # [80, 4096]
embedding = umap.UMAP().fit_transform(X)

# Plot
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
plt.title("UMAP: Emotions in Layer 12 Activation Space")
```

**Question**: Do emotions form distinct clusters?

**Success criteria**:
- Silhouette score > 0.2 (some separation)
- Visual clusters visible in plot

#### Experiment 3: Acoustic Feature Validation

```python
# Extract acoustic features for all 80 samples
for i, audio_file in enumerate(all_audio_files):
    features = extract_audio_features(audio_file)
    # features = {tempo, spectral_centroid, rms, chroma, ...}

    # Check if they match emotion labels
    if labels[i] == 'energetic':
        assert features['tempo'] > 120  # High tempo
        assert features['rms_mean'] > 0.05  # Loud

    if labels[i] == 'sad':
        assert features['tempo'] < 100  # Slow
        assert features['mode'] == 'minor'  # Minor key (often)
```

**Question**: Do generated samples actually sound like their labels?

**Success criteria**:
- Happy: High tempo, major key, high brightness
- Sad: Low tempo, minor key, low brightness
- Calm: Low energy, low tempo
- Energetic: High tempo, high energy

**Time**: 8-10 hours

---

### Week 4: Literature Foundation + Phase 1 Planning

**Goal**: Understand SAEs deeply enough to start training

#### Reading List (Essential)

1. **Toy Models of Superposition** (Anthropic, 2022)
   - Focus: Why are neural nets hard to interpret?
   - Time: 3 hours reading + 2 hours experiments
   - **Must understand**: Figures 1, 3, 8

2. **Sparse Autoencoders Find Highly Interpretable Features** (Anthropic, 2023)
   - Focus: How SAEs work, training objective
   - Time: 4 hours
   - **Must understand**: Section 3 (method), reconstruction/sparsity tradeoff

3. **ARENA 1.2: Intro to SAEs** (exercises)
   - Focus: Hands-on SAE training
   - Time: 6-8 hours
   - **Deliverable**: Trained SAE on toy model

#### SAE Training Preparation

**Define architecture for Phase 1**:
```python
# SAE Config for MusicGen Layer 12
sae_config = {
    'd_in': 2048,  # MusicGen Large d_model
    'd_sae': 16384,  # 8x overcomplete (standard)
    'l1_coefficient': 3e-4,  # Tune this
    'batch_size': 256,
    'lr': 1e-3,
    'num_epochs': 20,
}

# Training data: 80 samples × 459 timesteps = 36,720 activation vectors
```

**Questions to answer before Phase 1**:
- [ ] What sparsity level should we target? (90%? 95%?)
- [ ] How many SAE features can we interpret manually? (100? 1000?)
- [ ] What reconstruction loss is acceptable? (<5%? <10%?)

**Time**: 15-20 hours

---

## Phase 0 Completion Checklist (Updated)

### Technical Infrastructure ✅ (DONE)

- [✅] MusicGen Large loaded and working
- [✅] Activation extraction FIXED (captures all timesteps)
- [✅] Audio saving working (with FFmpeg fallback)
- [✅] Visualization utilities implemented
- [✅] Test scripts validate correctness

### Experimental Validation (IN PROGRESS)

- [✅] Verified emotions differentiate in activations (0.9461 similarity)
- [TODO] Generated 80-sample emotion dataset
- [TODO] Confirmed acoustic features match labels
- [TODO] UMAP shows emotion clustering
- [TODO] Identified which layers encode emotions best

### Theoretical Foundation (TODO)

- [TODO] Read superposition paper (Anthropic)
- [TODO] Read SAE paper (Anthropic)
- [TODO] Complete ARENA SAE exercises
- [TODO] Understand linear representation hypothesis

### Phase 1 Readiness (TODO)

- [TODO] SAE architecture defined
- [TODO] Training pipeline prepared
- [TODO] Evaluation metrics decided
- [TODO] Baseline results established

---

## Success Criteria: When is Phase 0 Actually Complete?

### Minimum Requirements

1. **Methodology validated**: ✅ (DONE - fixed extractor)
2. **Dataset collected**: TODO (need 80 samples)
3. **Acoustic validation**: TODO (features match labels)
4. **Clustering observed**: TODO (UMAP shows separation)
5. **SAE knowledge**: TODO (read papers + exercises)

### Stretch Goals

6. **All 48 layers analyzed** (find optimal layers for SAE)
7. **Temporal analysis** (when do emotions emerge?)
8. **Baseline SAE trained** (proof of concept)

---

## Realistic Timeline (3-4 Weeks)

### Week 1 (Nov 11-17): Methodology Validation
- Days 1-2: Update notebook, test all emotion pairs
- Days 3-5: Generate 80-sample dataset
- Days 6-7: Document findings, start reading

**Deliverable**: 80 audio samples + activations

### Week 2 (Nov 18-24): Statistical Analysis
- Days 1-3: Layer-wise similarity analysis
- Days 4-5: UMAP clustering
- Days 6-7: Acoustic feature validation

**Deliverable**: Technical report with plots

### Week 3 (Nov 25-Dec 1): Literature Deep Dive
- Days 1-3: Read Anthropic papers
- Days 4-7: Complete ARENA SAE exercises

**Deliverable**: 1-page SAE explainer (for yourself)

### Week 4 (Dec 2-8): Phase 1 Preparation
- Days 1-3: Design SAE architecture
- Days 4-5: Prepare training pipeline
- Days 6-7: Write Phase 1 experimental plan

**Deliverable**: Phase 1 detailed plan (5 pages)

**Phase 0 complete**: Dec 8, 2024
**Phase 1 start**: Dec 9, 2024

---

## Files to Create (Next Steps)

### Immediate (This Week)

1. **`experiments/01_generate_emotion_dataset.py`**
   - Generate 80 samples (20 per emotion)
   - Extract activations from 5 key layers
   - Save with metadata

2. **`experiments/02_validate_acoustic_features.py`**
   - Extract tempo, spectral features for all samples
   - Verify they match emotion labels
   - Plot distributions

3. **`experiments/03_layer_wise_analysis.py`**
   - Compute within/between emotion similarity for all layers
   - Find layers with strongest emotion signal
   - Visualize results

### Medium-Term (Week 2-3)

4. **`experiments/04_umap_clustering.py`**
   - UMAP projection of all 80 samples
   - Compute silhouette scores
   - Visualize emotion clustering

5. **`experiments/05_temporal_analysis.py`**
   - Analyze how similarity changes over timesteps
   - Find when emotions emerge during generation
   - Plot temporal dynamics

6. **`research_notes/phase0_findings.md`**
   - Document all results
   - Interpret findings
   - List questions for Phase 1

---

## Key Insights to Remember

### 1. The Bug Taught You Research Process

**Wrong approach** (what you did initially):
- See unexpected result (0.9999) → Move on → Declare success

**Right approach** (what you should do):
- See unexpected result → **Investigate deeply** → Find root cause → Fix → Re-evaluate

### 2. Similarity = 0.9461 is Actually Good News

- It's **not 0.99+** (which would mean no signal)
- It's **not 0.5** (which would be too different - likely a bug)
- It's **0.85-0.95** (meaningful but subtle differentiation)

This is **exactly what you'd expect** for emotions in a 2048-dim space!

### 3. Only 8% of Dimensions Differ Strongly

This validates your Phase 1 plan:
- **Without SAEs**: 2048 dims, most encode multiple concepts (polysemantic)
- **With SAEs**: 16,384 dims, each encodes one concept (monosemantic)
- SAEs will **disentangle** those 8% into interpretable features

### 4. You Need Statistical Rigor

- N=1 per emotion is **useless** (one datapoint proves nothing)
- N=20 per emotion is **minimum** for basic statistics
- N=100+ per emotion would be **ideal** (but impractical for now)

---

## What Good Phase 0 Completion Looks Like

### Technical Report Contents

1. **Architecture validation**:
   - "MusicGen Large has 48 layers, d_model=2048"
   - "Generation uses 459 timesteps for 8s audio"
   - "Activations shape: [459, 2, 1, 2048]"

2. **Emotion differentiation results**:
   - "4 emotions (happy/sad/calm/energetic) tested"
   - "80 samples generated (20 per emotion)"
   - "Layer 24 shows strongest differentiation (avg between-emotion sim = 0.91)"
   - "UMAP visualization shows partial clustering (silhouette = 0.23)"

3. **Acoustic validation**:
   - "Generated music matches intended emotions in 73 of 80 samples (91%)"
   - "Happy: tempo = 125±15 BPM, sad: tempo = 82±12 BPM"
   - "Correlation between tempo and arousal: r = 0.67, p < 0.001"

4. **Phase 1 readiness**:
   - "Will train SAEs on layer 24 (strongest emotion signal)"
   - "Target: 90% sparsity, <10% reconstruction loss"
   - "Expect to find 50-100 interpretable features related to emotion"

### What You'll Have Learned

- [✅] How MusicGen actually works (autoregressive generation)
- [TODO] How to design rigorous experiments
- [TODO] How to validate results statistically
- [TODO] How SAEs disentangle superposition
- [TODO] What makes a result "publishable"

---

## Final Warning: Don't Rush

**You have 9 months total**. Phase 0 is 2 months (you've used 1).

**Spending another 3-4 weeks is WORTH IT** if it means:
- Solid methodology
- Validated results
- Deep understanding of SAEs
- Clear Phase 1 plan

**Rushing to Phase 1 with weak foundations will fail.**

Better to spend 3 months on Phase 0 and publish something solid than rush through all phases and publish nothing.

---

## Summary: Your Action Plan

### This Week
1. ✅ Fix activation extraction (DONE!)
2. TODO: Update notebook
3. TODO: Generate 80-sample dataset

### Next 2 Weeks
4. TODO: Statistical validation (UMAP, acoustic features)
5. TODO: Layer-wise analysis
6. TODO: Read superposition + SAE papers

### Week 4
7. TODO: ARENA SAE exercises
8. TODO: Design Phase 1 experiments
9. TODO: Write Phase 0 technical report

**Then and only then**: Start Phase 1 (SAE training).

---

**Good luck. Think hard. Validate everything. Don't skip steps.**
