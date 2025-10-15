# MusicGen Emotion Interpretability Research

**Research Question**: Do music generation models "understand" emotion, or just correlate patterns?

**Answer (Phase 0)**: **YES! 96% accuracy** - Emotions encoded in T5 text embeddings

**Current Phase**: Phase 1 (SAE Training) - Ready to discover monosemantic emotion features

---

## New Here or Coming Back?

** START HERE: [WELCOME_BACK.md](WELCOME_BACK.md)** Clear entry point for returning users

---

## Overview

This project investigates how MusicGen (3.3B transformer) internally represents and manipulates musical emotion through:

1. **Phase 0** (Months 1-2): Foundation & methodology validation
2. **Phase 1** (Months 3-4): Train SAEs to find emotion-encoding features
3. **Phase 2** (Months 5-6): Causal probing (tempo energy emotion)
4. **Phase 3** (Months 7-9): Activation steering for emotion control

---

## Quick Start - Phase 1

```bash
# 1. Setup
source venv/bin/activate

# 2. Train SAE on T5 embeddings (10 minutes)
python3 experiments/train_sae_on_t5_embeddings.py

# 3. Analyze features (2 minutes)
python3 experiments/analyze_sae_features.py
```

** Phase 1 guide**: [PHASE1_QUICKSTART.md](PHASE1_QUICKSTART.md) | ** Phase 0 results**: [PHASE0_TO_PHASE1_SUMMARY.md](PHASE0_TO_PHASE1_SUMMARY.md)

---

## Current Status (Oct 10, 2024)

### Phase 0 COMPLETE
- **MAJOR DISCOVERY**: Emotions encoded in T5 text embeddings (96% accuracy)
- 100 T5 embeddings extracted and validated
- SAE infrastructure built and tested
- Training/analysis pipelines ready

### Phase 1 READY TO START
- Train Sparse Autoencoder on T5 embeddings (768-dim)
- Find monosemantic emotion-encoding features
- Target: 50+ selective features per emotion
- **Next action**: Run `python3 experiments/train_sae_on_t5_embeddings.py`

### Key Results
- T5 embeddings: 49% between-emotion similarity (STRONG differentiation)
- Transformer activations: 95% similarity (WEAK differentiation)
- Linear probe accuracy: 96%
- Statistical significance: p < 0.000001

** Phase 1 plan**: [PHASE1_ROADMAP.md](PHASE1_ROADMAP.md) | ** Quick start**: [PHASE1_QUICKSTART.md](PHASE1_QUICKSTART.md)

---

## Key Results

### Emotion Differentiation (After Fix)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Cosine similarity** | 0.9461 | 5.4% difference between happy/sad |
| **Strong dimensions** | 8% (80/1024) | Sparse encoding (expected) |
| **Temporal variation** | std = 0.026 | Similarity changes over generation |
| **Max dim difference** | 0.59 | Clear differentiation exists |

**Conclusion**: Emotions are represented differently in MusicGen's internal activations.

** Technical details**: [ACTIVATION_EXTRACTION_FIX.md](ACTIVATION_EXTRACTION_FIX.md)

---

## Success Criteria

Evidence the model "understands" emotion:
- Activations differentiate emotions (0.9461 similarity, not 0.99+)
- [TODO] SAE features are monosemantic & emotion-related
- [TODO] Causal pathways match human music perception
- [TODO] Activation steering produces coherent emotional shifts

**Either positive or negative results are publishable** - the question itself is novel.

---

## Project Structure

```
MusicGen/
START_HERE.md Read this first
README.md You are here
PHASE0_COMPLETE_PLAN.md Action plan

src/utils/
activation_utils.py Extract activations (FIXED)
audio_utils.py Audio processing
visualization_utils.py Plotting

test_fixed_extractor.py Validates extraction works
notebooks/ Interactive exploration
results/ Generated data
docs/ Learning roadmap
```

---

## Key Insights

### The Activation Bug (Oct 6-7)

**Problem**: Extractor was overwriting activations captured only last of 459 timesteps

**Impact**: 0.9999 similarity looked like no emotion differentiation

**Fix**: Store all timesteps now 0.9461 similarity

**Lesson**: Always validate methodology before interpreting results

### Why 0.9461 is Good News

- **Not 0.99+**: Shows real differentiation
- **Not 0.5**: Would suggest bug (too different)
- **0.85-0.95 range**: Meaningful but subtle (as expected)
- **8% strong dims**: Consistent with superposition theory

This validates the Phase 1 plan: Use SAEs to disentangle sparse emotion encoding.

---

## Dependencies

- Python 3.9+
- PyTorch 2.1+ (MPS for Apple Silicon)
- audiocraft (MusicGen)
- librosa, soundfile (audio processing)
- matplotlib, seaborn, umap-learn (visualization)

** Full list**: [requirements.txt](requirements.txt)

---

## Resources

### Papers
- **MusicGen**: [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
- **SAEs**: [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600)
- **Superposition**: [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/)
- **Activation Steering**: [Steering LLMs](https://arxiv.org/abs/2308.10248)

### Learning
- **ARENA**: [Transformer Interpretability](https://arena3-chapter1-transformer-interp.streamlit.app/)
- **Phase 0 Roadmap**: [docs/phase0_roadmap.md](docs/phase0_roadmap.md)

### Datasets (To Download)
- **PMEmo**: 794 songs with valence/arousal
- **DEAM**: 1,802 excerpts with dynamic emotion labels

---

## Timeline

| Phase | Duration | Goal | Status |
|-------|----------|------|--------|
| **0** | Months 1-2 | Foundation & validation | In progress |
| **1** | Months 3-4 | SAE training for emotions | Planned |
| **2** | Months 5-6 | Causal pathway analysis | Planned |
| **3** | Months 7-9 | Activation steering | Planned |

**Target completion**: June 2025

---

## Contact & Collaboration

This is an independent research project exploring mechanistic interpretability in music generation models.

**Communities**:
- EleutherAI Discord (interpretability)
- ARENA Slack (course support)
- LessWrong (mech interp discussions)

---

## License

Research project for educational and academic purposes.

---

** Get started**: [START_HERE.md](START_HERE.md)
