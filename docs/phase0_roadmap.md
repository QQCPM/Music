# Phase 0: Foundation - Learning Roadmap

**Duration**: Months 1-2 (8 weeks)
**Goal**: Build deep understanding of mechanistic interpretability and set up infrastructure

---

## Week 1-2: Mechanistic Interpretability Fundamentals

### Core Concepts to Master

#### 1. **Superposition and Feature Geometry**
Understanding why neural networks are hard to interpret.

**Key Questions**:
- Why do neural networks compress many features into fewer dimensions?
- What is the "superposition hypothesis"?
- How do features interfere with each other in activation space?

**Resources**:
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022)
- **Time**: 2-3 hours reading + experiments
- **Focus**: Understand Figure 1, 3, 8 deeply
- [ARENA 3.1: Superposition exercises](https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Toy_Models_of_Superposition)
- **Time**: 4-6 hours coding
- **Deliverable**: Complete notebook with your own experiments

**Success Criteria**:
- Can explain why `d_model` neurons can represent > `d_model` features
- Understand ReLU's role in superposition
- Completed ARENA exercises with >80% understanding

---

#### 2. **Sparse Autoencoders (SAEs)**
The main tool for disentangling superposition.

**Key Questions**:
- How do SAEs recover monosemantic (single-concept) features?
- What is the L1 penalty doing geometrically?
- Why do we need SAEs to be "overcomplete" (more features than input dimensions)?

**Resources**:
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) (Anthropic, 2023)
- **Time**: 3-4 hours
- **Focus**: Section 3 (Method), Figure 2, Figure 5
- [An Intuitive Explanation of Sparse Autoencoders](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html) (Adam Karvonen, 2024)
- **Time**: 1 hour
- **Very accessible**: Start here before the full paper
- [ARENA 3.2: Sparse Autoencoders exercises](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_SAEs)
- **Time**: 6-8 hours
- **Deliverable**: Train your own SAE on a tiny language model

**Success Criteria**:
- Can explain the SAE training objective: reconstruction loss + L1 sparsity
- Understand what "monosemantic" means with concrete examples
- Trained a working SAE and visualized features

---

#### 3. **Linear Representation Hypothesis**
The theoretical foundation for activation steering.

**Key Questions**:
- Do neural networks represent concepts as linear directions in activation space?
- What is a "causal inner product"? (Don't worry if this is hard at first)
- How does this relate to controlling model behavior?

**Resources**:
- [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658) (Park et al., 2024)
- **Time**: 2-3 hours (read intro + Section 2, skim proofs)
- **Focus**: Understand Figures 1-3, skip heavy math for now
- [Actually, Othello-GPT Has A Linear Emergent World Representation](https://www.neelnanda.io/mechanistic-interpretability/othello) (Neel Nanda, 2023)
- **Time**: 2 hours
- **Concrete example**: See LRH in action on a board game model

**Success Criteria**:
- Understand the claim: "concepts = linear directions"
- Can explain why this matters for interpretability
- Know the difference between "representation" and "steering"

---

#### 4. **Activation Steering / Activation Engineering**
How to control model behavior by editing activations.

**Key Questions**:
- How do we steer a model without retraining?
- What is "activation addition"?
- How do we evaluate if steering worked?

**Resources**:
- [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248) (2023)
- **Time**: 2 hours
- **Focus**: Section 2 (Method), Figure 1
- [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248) (Turner et al., 2023)
- **Time**: 1-2 hours
- **Focus**: Algorithm 1, understand the simplicity of the method

**Success Criteria**:
- Understand the formula: `modified_activation = base_activation + α * steering_vector`
- Know how to compute a steering vector from contrastive examples
- Understand evaluation: CLAP scores, classifiers, human eval

---

### Week 1-2 Deliverable
**Write a 1-page summary** answering:
1. What is superposition and why does it matter?
2. How do SAEs help with interpretability?
3. What is the linear representation hypothesis?
4. How does activation steering work?

---

## Week 3-4: MusicGen Architecture & Infrastructure

### Understanding MusicGen

**Goal**: Know MusicGen's architecture deeply enough to probe it.

#### 1. **Read the MusicGen Paper**
- [MusicGen: Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) (Copet et al., 2023)
- **Time**: 3-4 hours
- **Focus**:
- Section 2 (Model architecture)
- Figure 1 (understand the autoregressive generation)
- Section 3.1 (EnCodec tokenization)

**Key Concepts**:
- **EnCodec**: Compresses audio discrete tokens (like words in text)
- **Transformer Decoder**: Autoregressively generates tokens
- **Delayed Pattern**: Generates multiple codebook levels in parallel
- **Text Conditioning**: T5 embeddings guide generation

**Success Criteria**:
- Can draw the architecture from memory
- Understand: audio EnCodec tokens Transformer audio
- Know where to extract activations (which layers, which tokens)

---

#### 2. **Hands-on with MusicGen**

**Task A: Basic Inference**
```python
# Goal: Generate your first music samples
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained('facebook/musicgen-large') # 3.3B model
prompts = [
"happy upbeat electronic dance music",
"sad melancholic piano piece",
"energetic rock guitar solo",
"calm ambient meditation music"
]
samples = model.generate(prompts, progress=True)
```

**Deliverables**:
- Generate 20-30 samples with varied prompts
- Listen critically: What does the model do well? Poorly?
- Save samples for later analysis

**Task B: Extract Activations**
```python
# Goal: Hook into transformer layers
import torch

activations = {}

def get_activation(name):
def hook(model, input, output):
activations[name] = output.detach()
return hook

# Register hooks
for i in [0, 6, 12, 18, 24]: # Sample layers
model.lm.layers[i].register_forward_hook(get_activation(f'layer_{i}'))

# Generate and capture activations
output = model.generate(["happy music"], progress=True)
print(activations['layer_12'].shape) # Understand the tensor shape
```

**Deliverables**:
- Working code to extract activations from any layer
- Saved activations for 10 samples (5 happy, 5 sad)
- Understand tensor shapes: `[batch, sequence, d_model]`

**Task C: Inspect Attention Patterns**
```python
# Goal: Visualize what the model attends to
# Use audiocraft's built-in tools or custom code
```

**Deliverables**:
- Attention pattern visualizations for 3-5 samples
- Understand: Does the model attend differently for different emotions?

---

### Week 3-4 Deliverable
**Technical Report** (2-3 pages):
1. MusicGen architecture diagram (hand-drawn or digital)
2. Summary of generation process: text T5 conditioning tokens audio
3. Activation extraction code (annotated)
4. Initial observations: Do activations look different for happy vs. sad music?

---

## Week 5-6: Dataset Preparation

### Acquire Emotion-Labeled Music Datasets

#### 1. **PMEmo Dataset**
- **Link**: [PMEmo on GitHub](https://github.com/HuiZhangDB/PMEmo)
- **Content**: 794 songs, valence/arousal annotations
- **Task**: Download, organize, create metadata CSV
- **Script**: `scripts/download_pmemo.py`

#### 2. **DEAM Dataset**
- **Link**: [DEAM on MediaEval](http://cvml.unige.ch/databases/DEAM/)
- **Content**: 1,802 music excerpts, dynamic valence/arousal
- **Task**: Download, extract 30-second clips
- **Script**: `scripts/download_deam.py`

#### 3. **Map to Discrete Emotions**
- **Valence + Arousal 4 quadrants**:
- High valence, High arousal **Happy/Energetic**
- High valence, Low arousal **Calm/Peaceful**
- Low valence, High arousal **Angry/Tense**
- Low valence, Low arousal **Sad/Melancholic**

**Deliverable**: `data/processed/emotion_labeled_dataset.csv`

---

### Extract Acoustic Features

Use `librosa` to extract features for later causal analysis:

**Features to Extract**:
1. **Tempo** (BPM)
2. **Key** (major/minor)
3. **Spectral centroid** (brightness)
4. **RMS energy** (loudness)
5. **Chroma features** (harmony)
6. **MFCCs** (timbre)

**Script**: `scripts/extract_audio_features.py`

**Deliverable**: `data/processed/audio_features.csv`

---

### Week 5-6 Deliverable
**Data Report**:
1. Dataset statistics (# samples per emotion)
2. Acoustic feature distributions (histograms)
3. Correlations: tempo vs. energy, key vs. valence, etc.
4. Ready-to-use dataset for Phase 1 experiments

---

## Week 7-8: Literature Deep Dive

### Critical Papers for Your Research

#### Must-Read (These directly apply to your project)

1. **"Discovering and Steering Interpretable Concepts in Large Generative Music Models"** (Singh et al., May 2025)
- **Why**: This is YOUR exact research direction
- **Time**: 4-5 hours
- **Task**: Implement their SAE evaluation pipeline (if code available)

2. **"Fine-Grained control over Music Generation with Activation Steering"** (June 2025)
- **Why**: Direct precedent for Phase 3
- **Time**: 3 hours
- **Task**: Understand their CLAP evaluation method

3. **"Sparse Autoencoders Make Audio Foundation Models more Explainable"** (Sept 2025)
- **Why**: SAEs for audio (not just text!)
- **Time**: 3 hours
- **Task**: Compare to your planned approach

#### Important Context

4. **"Measuring the Reliability of Causal Probing Methods"** (Aug 2024)
- **Why**: Avoid spurious findings in Phase 2
- **Time**: 2-3 hours
- **Focus**: Completeness & selectivity metrics

5. **EACL 2024 Tutorial: "Transformer-specific Interpretability"**
- **Why**: Causal intervention techniques
- **Time**: 4-5 hours
- **Task**: Run their notebooks

#### Background (Skim for now, deep-read when needed)

6. Music perception neuroscience papers:
- "Live music stimulates the affective brain..." (PNAS, 2024)
- "On joy and sorrow: Neuroimaging meta-analyses..." (MIT Press, 2024)
- **Why**: Ground truth for human causal pathways

---

### Week 7-8 Deliverable
**Literature Review** (5-7 pages):
1. **What we know**: Current state of music generation interpretability
2. **What we don't know**: Gaps in the literature (your contributions!)
3. **Methods you'll use**: SAEs, causal probing, activation steering
4. **Expected contributions**: Novel comparisons, first results in music domain

---

## Phase 0 Final Deliverable

### Comprehensive Setup Document
**Due**: End of Week 8

**Contents**:
1. Completed ARENA exercises (code + writeups)
2. MusicGen infrastructure working (can generate + extract activations)
3. Datasets prepared and documented
4. Literature review complete
5. Research questions refined based on literature
6. Phase 1 experimental plan detailed

**Success Criteria**:
- Can explain mechanistic interpretability concepts to a peer
- Can generate music and extract activations programmatically
- Have clean, documented datasets
- Know exactly what experiments to run in Phase 1

---

## Checkpoints & Self-Assessment

### End of Week 2
- [ ] Understand superposition conceptually
- [ ] Completed basic SAE training exercise
- [ ] Can explain linear representation hypothesis

### End of Week 4
- [ ] Generated 20+ music samples
- [ ] Extracted activations from multiple layers
- [ ] Understand MusicGen architecture

### End of Week 6
- [ ] Downloaded and processed datasets
- [ ] Extracted acoustic features
- [ ] Visualized emotion distributions

### End of Week 8
- [ ] Literature review complete
- [ ] All infrastructure ready
- [ ] Phase 1 experiments planned in detail

---

## Tips for Success

1. **Don't rush**: Understanding > speed
2. **Take notes**: Document everything you learn
3. **Code as you learn**: Implement concepts in small experiments
4. **Ask questions**: Join Discord communities (EleutherAI, ARENA, etc.)
5. **Visualize often**: Plot activations, features, distributions
6. **Stay grounded**: Connect concepts to concrete examples

---

## Resources & Communities

### Learning Communities
- [EleutherAI Discord](https://discord.gg/eleutherai) - Interpretability channel
- [ARENA Slack](https://arena-uk.slack.com) - Course support
- [LessWrong](https://www.lesswrong.com) - Mechanistic interpretability posts

### Code Repositories
- [SAELens](https://github.com/jbloomAus/SAELens) - SAE training library
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Interpretability tools
- [AudioCraft](https://github.com/facebookresearch/audiocraft) - MusicGen official repo

### Keep Updated
- [Alignment Newsletter](https://rohinshah.com/alignment-newsletter/)
- [Import AI](https://jack-clark.net) - Weekly AI news
- Follow researchers: Neel Nanda, Trenton Bricken, Arthur Conmy on Twitter/X

---

**Remember**: This is a marathon, not a sprint. Deep understanding now will make Phases 1-3 much smoother!
