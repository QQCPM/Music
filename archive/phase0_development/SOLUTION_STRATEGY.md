# Solution Strategy: How to Find Emotion Encoding in MusicGen

**Problem**: Current evidence shows weak/no emotion differentiation
**Goal**: Find WHERE and HOW emotions are encoded (if at all)

---

## Root Cause Analysis

### Why Is The Signal So Weak?

I see **5 possible explanations**:

#### 1. **We're Looking at the Wrong Layer** 🎯

**Evidence**:
- Only tested layer 12 (middle layer)
- MusicGen has 24-48 layers depending on size
- Different layers encode different information

**Hypothesis**:
- Early layers: Low-level audio features (pitch, rhythm)
- Middle layers: Musical structure (harmony, melody)
- Late layers: High-level semantics (emotion, style)
- **Emotion might be in late layers (18-24), not middle (12)**

**How to test**:
```python
# Extract from ALL layers
layers_to_test = list(range(0, 24))  # All layers
# Find which layer shows maximum differentiation
```

**Expected if true**: One layer shows 5-10% difference, others show <2%

---

#### 2. **Prompts Are Too Generic** 📝

**Evidence**:
- "happy music" vs "sad music" - very broad
- Model might not know what we mean
- Our acoustic analysis: tempo BARELY differs

**Current prompts**:
```
"happy upbeat cheerful music"  → Too generic
"sad melancholic sorrowful music"  → Too generic
```

**Better prompts** (more specific):
```
# Extreme happiness
"euphoric celebration music with fast tempo 140 BPM, major key, bright synths, energetic drums"

# Extreme sadness
"deeply sorrowful funeral dirge, slow tempo 60 BPM, minor key, solo cello, mournful"

# Try other dimensions
"aggressive angry metal with distorted guitars screaming vocals"
"peaceful zen meditation music, slow gentle ambient, no drums"
```

**How to test**: Generate with specific prompts, measure acoustic features

**Expected if true**: Specific prompts → larger tempo/energy differences

---

#### 3. **Wrong Similarity Metric** 📊

**Evidence**:
- Cosine similarity measures angle, not magnitude
- Might miss important differences in activation patterns

**Problem with cosine similarity**:
- If emotions shift ALL dimensions uniformly → similarity stays high
- Cosine only measures direction, not scale

**Alternative metrics to try**:

**A) Euclidean Distance**:
```python
dist = torch.norm(happy - sad)
# Measures actual magnitude of difference
```

**B) Centered Kernel Alignment (CKA)**:
```python
# Compares representational similarity
# More robust than cosine for neural activations
from utils import cka_similarity
```

**C) Maximum Mean Discrepancy (MMD)**:
```python
# Tests if two distributions differ
# Better for detecting distributional shifts
```

**D) Classification Accuracy**:
```python
# Train linear probe: can it predict emotion from activations?
# If accuracy > 75%, signal exists
```

**How to test**: Compute all metrics, see which is most sensitive

**Expected if true**: One metric shows clear separation where cosine doesn't

---

#### 4. **Looking at Wrong Aspect of Activations** 🔍

**Evidence**:
- Averaging over all timesteps might wash out signal
- Emotion might be in specific timesteps or attention patterns

**What we're doing wrong**:
```python
# Current: Average across ALL timesteps
happy_mean = happy_activations.mean(dim=0)
# This might cancel out important temporal dynamics!
```

**Better approaches**:

**A) Timestep-specific analysis**:
```python
# Look at first 10% (conditioning phase)
early = activations[:15, ...]
# vs last 10% (generation completion)
late = activations[-15:, ...]
# Emotion might be encoded at START, then executed
```

**B) Variance analysis**:
```python
# Happy music might have HIGH variance (dynamic)
# Sad music might have LOW variance (monotonous)
happy_var = happy_activations.var(dim=0)
sad_var = sad_activations.var(dim=0)
```

**C) Attention pattern analysis**:
```python
# Extract attention weights, not just activations
# Emotion might be in WHAT the model attends to
```

**D) Activation magnitude analysis**:
```python
# Not just direction (cosine), but STRENGTH
happy_magnitude = happy_activations.abs().mean()
sad_magnitude = sad_activations.abs().mean()
```

**How to test**: Analyze each aspect separately

**Expected if true**: Signal appears in specific timesteps or attention patterns

---

#### 5. **MusicGen Genuinely Doesn't Encode Emotions Much** 💭

**Evidence**:
- Acoustic features barely differ
- Statistical tests fail
- Even with perfect methodology, might not find signal

**This is a VALID research finding**:
- "MusicGen generates music via pattern matching, not emotional understanding"
- Compare to: Human composers (do encode emotion)
- Suggest: Architectural changes needed

**How to test**:
- Try other models (Jukebox, AudioLDM)
- Try other modalities (image, text - known to work)
- If others work but music doesn't → interesting finding!

**Expected if true**: No methodology finds strong signal in MusicGen

---

## Comprehensive Solution Strategy

### Phase 1: Systematic Layer Search (Week 1)

**Goal**: Find which layer(s) encode emotions

**Method**:
```python
# Test ALL layers
for layer in range(24):
    extractor = ActivationExtractor(model, layers=[layer])

    # Generate 5 happy, 5 sad
    happy_acts = [extractor.generate(["happy"]) for _ in range(5)]
    sad_acts = [extractor.generate(["sad"]) for _ in range(5)]

    # Compute within vs between similarity
    within_happy = compute_within_similarity(happy_acts)
    between = compute_between_similarity(happy_acts, sad_acts)

    signal_strength = within_happy - between
    print(f"Layer {layer}: signal = {signal_strength:.4f}")
```

**Expected outcome**:
- **If hypothesis 1 is true**: One layer shows signal_strength > 0.05
- **If not**: All layers show < 0.02 (no emotion encoding anywhere)

**Decision**:
- If found: Use that layer for Phase 2
- If not found: Proceed to Phase 2 anyway (try prompts)

---

### Phase 2: Prompt Engineering (Week 1-2)

**Goal**: Find prompts that produce clearly different music

**Method**:

**Step 1**: Design extreme prompts
```python
extreme_prompts = {
    'happy': [
        "euphoric dance party, 150 BPM, major key, celebration, bright",
        "joyful children laughing and playing, upbeat xylophone",
        "triumphant victory fanfare, brass section, energetic",
    ],
    'sad': [
        "funeral dirge, 50 BPM, solo cello, crying, mournful, dark",
        "rainy day melancholy, slow piano, somber, depressed",
        "heartbreak ballad, minor key, sorrowful violin, tears",
    ],
    'angry': [
        "aggressive death metal, screaming, distorted guitars, 180 BPM",
        "violent rage, pounding drums, chaotic, harsh",
    ],
    'calm': [
        "zen meditation, 40 BPM, soft flute, peaceful, silence",
        "gentle lullaby, slow, quiet, soothing, sleep",
    ]
}
```

**Step 2**: Validate acoustically
```python
for emotion, prompts in extreme_prompts.items():
    for prompt in prompts:
        audio = generate(prompt)
        features = extract_features(audio)

        # Check if acoustic features match intention
        if emotion == 'happy':
            assert features['tempo'] > 120  # Fast
            assert features['energy'] > 0.08  # Loud
            assert features['mode'] == 'major'
        # etc.
```

**Step 3**: Keep only prompts that work
- If acoustic features match → good prompt
- If not → revise prompt

**Expected outcome**:
- Find 3-5 prompts per emotion that reliably produce different music
- These become your ground truth

---

### Phase 3: Alternative Metrics (Week 2)

**Goal**: Find metrics more sensitive than cosine similarity

**Method**:
```python
metrics = {
    'cosine': cosine_similarity,
    'euclidean': euclidean_distance,
    'cka': cka_similarity,
    'mmd': maximum_mean_discrepancy,
    'linear_probe': train_linear_classifier,
}

for metric_name, metric_fn in metrics.items():
    score = metric_fn(happy_acts, sad_acts)
    print(f"{metric_name}: {score:.4f}")
```

**Linear probe is KEY**:
```python
# If a linear classifier can predict emotion from activations,
# then emotion IS encoded (linearly separable)

X = np.vstack([happy_acts, sad_acts])  # [40, d_model]
y = [0]*20 + [1]*20  # labels

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
scores = cross_val_score(clf, X, y, cv=5)

print(f"Classification accuracy: {scores.mean():.2%}")
# > 75% → emotion is encoded
# < 60% → no encoding
```

**Expected outcome**:
- If one metric works → use it going forward
- If linear probe works → emotions are linearly encoded (good for SAEs!)
- If nothing works → weak signal confirmed

---

### Phase 4: Temporal & Attention Analysis (Week 2-3)

**Goal**: Find if emotion is in specific parts of the sequence

**Method**:

**A) Timestep sweep**:
```python
# For each timestep, compute happy vs sad similarity
for t in range(num_timesteps):
    happy_t = happy_acts[:, t, ...]
    sad_t = sad_acts[:, t, ...]
    sim_t = cosine_similarity(happy_t, sad_t)

    plt.plot(t, sim_t)

# Look for timesteps where similarity DROPS
# Those timesteps encode emotion
```

**B) Variance analysis**:
```python
# Happy music: high variance (dynamic)
# Sad music: low variance (static)

happy_var = happy_acts.var(dim=0).mean()
sad_var = sad_acts.var(dim=0).mean()

print(f"Happy variance: {happy_var:.4f}")
print(f"Sad variance: {sad_var:.4f}")

# If happy_var > sad_var * 1.5 → signal!
```

**C) Spectral analysis**:
```python
# FFT of activation sequences
# Different emotions might have different temporal frequencies

from scipy.fft import fft
happy_fft = np.abs(fft(happy_acts, axis=0))
sad_fft = np.abs(fft(sad_acts, axis=0))

# Compare frequency spectra
```

**Expected outcome**:
- Find specific timesteps or frequencies where emotions differ
- This narrows down WHERE to look

---

### Phase 5: Multi-Model Comparison (Week 3)

**Goal**: Determine if this is MusicGen-specific or universal

**Method**:
```python
models_to_test = [
    'facebook/musicgen-small',
    'facebook/musicgen-medium',
    'facebook/musicgen-large',
    # 'facebook/musicgen-melody',  # Conditioned on melody
]

for model_name in models_to_test:
    model = load_model(model_name)
    signal = measure_emotion_signal(model)
    print(f"{model_name}: signal = {signal:.4f}")
```

**Expected outcome**:
- If larger models show stronger signal → size matters
- If all models show weak signal → architecture issue
- If melody model works → conditioning helps

---

## Concrete Experimental Plan

### Week 1: Rapid Exploration

**Monday-Tuesday**: Layer sweep
- Extract from all 24 layers
- Find layer with maximum differentiation
- **Output**: Best layer(s) identified

**Wednesday-Thursday**: Prompt engineering
- Test 20 extreme prompts
- Validate acoustic features
- **Output**: 5 prompts per emotion that work

**Friday**: Re-test with best layer + best prompts
- Generate 10 samples per emotion
- Compute within vs between similarity
- **Decision point**: Is signal > 2%?

### Week 2: Deep Dive (IF signal found)

**Monday-Tuesday**: Alternative metrics
- Test CKA, MMD, linear probe
- Find most sensitive metric
- **Output**: Best metric identified

**Wednesday-Thursday**: Temporal analysis
- Timestep-by-timestep similarity
- Variance analysis
- Attention patterns
- **Output**: When/where emotion appears

**Friday**: Synthesis
- Generate 20 samples with best everything
- Final within/between test
- **Decision**: Proceed to Phase 1 or pivot?

### Week 2-3: Pivot (IF signal NOT found)

**Alternative research questions**:

**A) "Why MusicGen Doesn't Encode Emotions"**
- Compare to human perception
- Analyze training data biases
- Propose architectural improvements

**B) "Where Music Models Fail at Emotion"**
- Test multiple models
- Identify systematic weaknesses
- Design better evaluation metrics

**C) "Emotion in Other Modalities"**
- Apply same techniques to text (BERT, GPT)
- Show it works there, not in music
- Explore why music is harder

All of these are **publishable negative results**.

---

## Critical Success Factors

### What "Success" Looks Like

**Minimal success** (publishable):
- Find at least one layer where between/within difference > 3%
- Linear probe accuracy > 70%
- Acoustic features consistently match labels

**Strong success**:
- One layer shows > 5% difference
- Linear probe > 80% accuracy
- Clear temporal pattern (emotion emerges at specific point)

**Failure** (but still publishable):
- No layer shows > 2% difference
- Linear probe < 60% (chance level)
- Pivot to "why not" paper

### Red Lines (When to Pivot)

**Pivot if**:
- After testing all 24 layers, best is < 2%
- After 20 extreme prompts, acoustic features don't differ
- After 100 samples, between ≈ within
- After trying 5 metrics, none show signal

**Don't waste time** beyond Week 3 if no signal.

---

## Implementation Priority

### Must Do (Week 1)

1. **Layer sweep** - Most likely to find signal
2. **Prompt engineering** - Cheap, high impact
3. **Within/between similarity** - Proper statistical test

### Should Do (Week 2)

4. **Linear probe** - Definitive test
5. **Alternative metrics** - Might be more sensitive
6. **Temporal analysis** - Where is emotion?

### Nice to Have (Week 3)

7. **Multi-model comparison**
8. **Attention analysis**
9. **Spectral analysis**

---

## Code Structure

### Main Script: `experiments/comprehensive_emotion_search.py`

```python
def layer_sweep(model, prompts):
    """Test all layers, return best"""
    results = {}
    for layer in range(24):
        signal = test_layer(model, layer, prompts)
        results[layer] = signal
    return max(results, key=results.get)

def prompt_validation(prompts):
    """Test which prompts produce different audio"""
    valid_prompts = {}
    for emotion, prompt_list in prompts.items():
        valid = []
        for prompt in prompt_list:
            audio = generate(prompt)
            features = extract_features(audio)
            if validate_features(emotion, features):
                valid.append(prompt)
        valid_prompts[emotion] = valid
    return valid_prompts

def within_vs_between_test(activations_by_emotion):
    """Proper statistical test"""
    within_sims = []
    between_sims = []

    for emotion, acts in activations_by_emotion.items():
        # Within-emotion similarity
        for i in range(len(acts)):
            for j in range(i+1, len(acts)):
                sim = cosine_similarity(acts[i], acts[j])
                within_sims.append(sim)

    # Between-emotion similarity
    for e1, acts1 in activations_by_emotion.items():
        for e2, acts2 in activations_by_emotion.items():
            if e1 >= e2: continue
            for a1 in acts1:
                for a2 in acts2:
                    sim = cosine_similarity(a1, a2)
                    between_sims.append(sim)

    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(within_sims, between_sims)

    mean_within = np.mean(within_sims)
    mean_between = np.mean(between_sims)

    return {
        'within': mean_within,
        'between': mean_between,
        'difference': mean_within - mean_between,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def main():
    # 1. Load model
    model = MusicGen.get_pretrained('facebook/musicgen-large')

    # 2. Layer sweep
    print("Phase 1: Layer sweep...")
    best_layer = layer_sweep(model, initial_prompts)
    print(f"Best layer: {best_layer}")

    # 3. Prompt validation
    print("Phase 2: Prompt validation...")
    valid_prompts = prompt_validation(extreme_prompts)

    # 4. Generate large dataset with best layer + prompts
    print("Phase 3: Generating dataset...")
    activations = generate_dataset(model, best_layer, valid_prompts, n=20)

    # 5. Within vs between test
    print("Phase 4: Statistical test...")
    results = within_vs_between_test(activations)

    print(f"\nResults:")
    print(f"  Within-emotion: {results['within']:.4f}")
    print(f"  Between-emotion: {results['between']:.4f}")
    print(f"  Difference: {results['difference']:.4f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Significant: {results['significant']}")

    # 6. Decision
    if results['significant'] and results['difference'] > 0.02:
        print("\n✅ SIGNAL FOUND! Proceed to Phase 1 (SAE training)")
    else:
        print("\n⚠️ WEAK/NO SIGNAL. Consider pivot.")
```

---

## Expected Timeline

### Optimistic (Signal Exists)

- **Week 1**: Find layer 18 shows 4% difference
- **Week 2**: Confirm with 20 samples, p < 0.01
- **Week 3**: Deep analysis, prepare Phase 1
- **Week 4**: Start SAE training
- **Result**: Original plan proceeds

### Realistic (Weak Signal)

- **Week 1**: Layer 20 shows 2.5% difference
- **Week 2**: Needs 50+ samples to reach significance
- **Week 3**: Expand dataset, p = 0.03
- **Week 4**: Decide if worth pursuing
- **Result**: Extended Phase 0, or pivot

### Pessimistic (No Signal)

- **Week 1**: All layers < 1.5% difference
- **Week 2**: Even extreme prompts show weak effect
- **Week 3**: Pivot to "why not" paper
- **Week 4**: Start new direction
- **Result**: Learned a lot, but different path

---

## Why This Will Work

### Systematic Approach
- Test ALL layers (not just one)
- Test MANY prompts (not just generic)
- Test MULTIPLE metrics (not just cosine)
- Test PROPER statistics (within vs between)

### Fail-Safe Design
- Even if MusicGen doesn't encode emotions, we learn something
- Negative results are publishable
- Methodology is reusable

### Data-Driven
- Every decision based on measurements
- Clear decision criteria
- No more "seems like it works"

---

## Final Thoughts

### The Real Problem

**We've been doing "wishful interpretability"**:
- Assumed emotions were encoded
- Found a small difference
- Convinced ourselves it was meaningful
- Didn't validate thoroughly

### The Solution

**Do "skeptical interpretability"**:
- Assume nothing
- Test systematically
- Validate at every step
- Follow the data

### The Mindset Shift

**Before**: "Let's find how emotions are encoded"

**Now**: "Let's find IF and WHERE emotions are encoded"

**This is proper science.**

---

## Action Plan (Starting Now)

1. **Read this document** (you are here)
2. **Decide**: Continue or pivot?
3. **If continue**: Start Week 1 experiments
4. **If pivot**: Choose alternative research question

**Either way, you have a clear path forward.**

**No more guessing. Just systematic exploration.**
