# MusicGen Emotion Interpretability Research

**Research Question**: Do music generation models "understand" emotion, or just correlate patterns?

---

## Overview

This project investigates how MusicGen (3.3B transformer) internally represents and manipulates musical emotion through:

1. **Phase 0**: Foundation & methodology validation
2. **Phase 1**: Train SAEs to find emotion-encoding features
3. **Phase 2**: Causal probing (tempo energy emotion)
4. **Phase 3**: Activation steering for emotion control

---

## Quick Start

```bash
# 1. Setup
source venv/bin/activate

# 2. Train SAE on T5 embeddings 
python3 experiments/train_sae_on_t5_embeddings.py

# 3. Analyze features 
python3 experiments/analyze_sae_features.py
```
---

## Current Status (Oct 10, 2024)

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
- Activations differentiate emotions (0.9461 similarity)
- [TODO] SAE features are monosemantic & emotion-related
- [TODO] Causal pathways match human music perception
- [TODO] Activation steering produces coherent emotional shifts

**Either positive or negative results are publishable** - the question itself is novel.


## Key Insights

### The Activation Bug 

**Problem**: Extractor was overwriting activations captured only last of 459 timesteps

**Impact**: 0.9999 similarity looked like no emotion differentiation

**Fix**: Store all timesteps now 0.9461 similarity

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
| **0** | Foundation & validation | In progress |
| **1** | SAE training for emotions | Planned |
| **2** | Causal pathway analysis | Planned |
| **3** | Activation steering | Planned |

**Target completion**: 2026

---

## Contact & Collaboration

This is an independent research project exploring mechanistic interpretability in music generation models.



---

## License

Research project for educational and academic purposes.

---

** Get started**: [START_HERE.md](START_HERE.md)
