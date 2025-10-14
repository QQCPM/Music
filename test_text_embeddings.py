#!/usr/bin/env python3
"""
THE SMOKING GUN TEST: Are emotions encoded in TEXT EMBEDDINGS?

This is the FIRST test we should have run.
If emotions differ in T5 embeddings (MusicGen's text encoder),
then emotion IS encoded - just in the INPUT, not the processing.

This test takes 10 minutes and could change everything.
"""

import sys
sys.path.append('.')

import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from src.utils.activation_utils import cosine_similarity

print("=" * 80)
print("THE SMOKING GUN TEST: Text Embedding Analysis")
print("=" * 80)
print()
print("Question: Are emotions already encoded in the TEXT before generation?")
print("If YES: Emotion encoding is STRONG (just not where we looked)")
print("If NO: Emotion might be in generation process or output")
print()

# Load T5 (this is what MusicGen uses for text conditioning)
print("Loading T5-base encoder (MusicGen uses this)...")
tokenizer = T5Tokenizer.from_pretrained('t5-base')
encoder = T5EncoderModel.from_pretrained('t5-base')
print("✅ T5 loaded")
print()

# Define emotion prompts (same as MusicGen experiments)
emotion_prompts = {
    'happy': [
        "happy cheerful upbeat music",
        "joyful energetic positive music",
        "euphoric celebration music",
    ],
    'sad': [
        "sad melancholic sorrowful music",
        "depressing mournful tragic music",
        "heartbreak funeral dirge music",
    ],
    'calm': [
        "calm peaceful relaxing music",
        "tranquil serene gentle music",
        "soothing meditation ambient music",
    ],
    'energetic': [
        "energetic intense powerful music",
        "high energy driving forceful music",
        "adrenaline pumping aggressive music",
    ],
}

print("=" * 80)
print("Phase 1: Extract T5 Embeddings")
print("=" * 80)
print()

# Extract embeddings for all prompts
embeddings = {}

for emotion, prompts in emotion_prompts.items():
    print(f"Processing {emotion} prompts...")
    embeddings[emotion] = []

    for prompt in prompts:
        # Tokenize
        tokens = tokenizer(prompt, return_tensors='pt', padding=True)

        # Get T5 embeddings
        with torch.no_grad():
            output = encoder(**tokens)

        # Mean pooling over sequence
        emb = output.last_hidden_state.mean(dim=1).squeeze()  # [768]
        embeddings[emotion].append(emb)

        print(f"  '{prompt[:40]}...' → embedding shape: {emb.shape}")

    print()

# Compute average embedding per emotion
avg_embeddings = {}
for emotion, emb_list in embeddings.items():
    avg_embeddings[emotion] = torch.stack(emb_list).mean(dim=0)
    print(f"{emotion:12s}: avg embedding computed")

print()

# ============================================================================
# Phase 2: Compute Similarity Matrix
# ============================================================================

print("=" * 80)
print("Phase 2: Similarity Matrix")
print("=" * 80)
print()

emotions = list(avg_embeddings.keys())
n = len(emotions)

# Compute all pairwise similarities
similarity_matrix = np.zeros((n, n))

print("Pairwise cosine similarities:")
print()

for i, e1 in enumerate(emotions):
    for j, e2 in enumerate(emotions):
        sim = cosine_similarity(avg_embeddings[e1], avg_embeddings[e2])
        similarity_matrix[i, j] = sim

        if i < j:  # Only print upper triangle
            print(f"{e1:12s} vs {e2:12s}: {sim:.4f}")

print()

# ============================================================================
# Phase 3: Analysis
# ============================================================================

print("=" * 80)
print("Phase 3: Analysis")
print("=" * 80)
print()

# Within-category vs between-category
within_sims = []  # Same emotion, different prompts
between_sims = []  # Different emotions

for emotion, emb_list in embeddings.items():
    # Within: compare different prompts of same emotion
    for i in range(len(emb_list)):
        for j in range(i+1, len(emb_list)):
            sim = cosine_similarity(emb_list[i], emb_list[j])
            within_sims.append(sim)

# Between: compare average embeddings of different emotions
for i in range(n):
    for j in range(i+1, n):
        between_sims.append(similarity_matrix[i, j])

mean_within = np.mean(within_sims)
mean_between = np.mean(between_sims)
std_within = np.std(within_sims)
std_between = np.std(between_sims)

print(f"Within-emotion similarity (same emotion, different prompts):")
print(f"  Mean: {mean_within:.4f} ± {std_within:.4f}")
print()

print(f"Between-emotion similarity (different emotions):")
print(f"  Mean: {mean_between:.4f} ± {std_between:.4f}")
print()

difference = mean_within - mean_between
print(f"Difference: {difference:.4f}")
print()

# Statistical test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(within_sims, between_sims)
print(f"T-test: t = {t_stat:.4f}, p = {p_value:.4f}")
print()

# ============================================================================
# Phase 4: Interpretation
# ============================================================================

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

print("Comparison with our transformer activation results:")
print(f"  Transformer activations: 0.946 similarity (happy vs sad)")
print(f"  Text embeddings: {similarity_matrix[0, 1]:.4f} similarity (happy vs sad)")
print()

# Decision logic
if mean_between < 0.85:
    print("✅✅✅ EMOTIONS ARE STRONGLY ENCODED IN TEXT EMBEDDINGS!")
    print()
    print("Key findings:")
    print(f"  - Between-emotion similarity: {mean_between:.4f} (< 0.85)")
    print(f"  - Within-emotion similarity: {mean_within:.4f}")
    print(f"  - Difference: {difference:.4f}")
    print(f"  - P-value: {p_value:.6f}")
    print()
    print("What this means:")
    print("  → Emotions ARE encoded, but in the TEXT REPRESENTATION")
    print("  → Transformer just executes the plan (hence similar activations)")
    print("  → We were looking in the WRONG PLACE")
    print()
    print("Revised research strategy:")
    print("  1. Phase 1 should target TEXT EMBEDDINGS, not transformer activations")
    print("  2. Train SAEs on T5 embeddings (768-dim) not transformer (2048-dim)")
    print("  3. Find emotion-encoding features in text space")
    print("  4. This is actually EASIER (smaller dimensionality)")
    print()
    print("Next steps:")
    print("  - Extract T5 embeddings for 100+ prompts")
    print("  - Train SAEs on T5 embedding space")
    print("  - Find monosemantic features for emotions")
    print()
    verdict = "FOUND_IN_TEXT"

elif mean_between < 0.92:
    print("✅ EMOTIONS ARE MODERATELY ENCODED IN TEXT EMBEDDINGS")
    print()
    print("Key findings:")
    print(f"  - Between-emotion similarity: {mean_between:.4f} (0.85-0.92)")
    print(f"  - Some differentiation exists, but not as strong as hoped")
    print()
    print("What this means:")
    print("  → Emotion encoding is SPLIT between text and generation")
    print("  → Need to look at BOTH text embeddings AND transformer activations")
    print("  → Or emotion is in the INTERACTION (cross-attention)")
    print()
    print("Next steps:")
    print("  - Test cross-attention patterns (how text conditions generation)")
    print("  - Combine text + transformer features")
    print("  - Look at attention mechanisms")
    print()
    verdict = "PARTIAL_IN_TEXT"

else:
    print("❌ EMOTIONS ARE NOT STRONGLY ENCODED IN TEXT EMBEDDINGS")
    print()
    print("Key findings:")
    print(f"  - Between-emotion similarity: {mean_between:.4f} (> 0.92)")
    print(f"  - Text embeddings are too similar")
    print()
    print("What this means:")
    print("  → T5 doesn't distinguish emotions much")
    print("  → Emotion must be in GENERATION process or OUTPUT")
    print("  → Need to look at:")
    print("     1. Transformer activations (all layers)")
    print("     2. Audio tokens (output)")
    print("     3. Temporal dynamics (changes over time)")
    print()
    print("Next steps:")
    print("  - Continue with comprehensive layer sweep")
    print("  - Test audio token distributions")
    print("  - Analyze attention patterns")
    print()
    verdict = "NOT_IN_TEXT"

print()

# ============================================================================
# Phase 5: Hypothesis Validation
# ============================================================================

print("=" * 80)
print("HYPOTHESIS VALIDATION")
print("=" * 80)
print()

print("Testing our hypothesis:")
print("  'Emotions are in INPUT (text), not PROCESS (transformer)'")
print()

if verdict == "FOUND_IN_TEXT":
    print("✅ HYPOTHESIS CONFIRMED")
    print()
    print("Evidence:")
    print(f"  1. Text embeddings show {difference:.4f} differentiation")
    print(f"  2. Transformer activations show 0.005 differentiation")
    print(f"  3. Ratio: {difference / 0.005:.1f}x more signal in text")
    print()
    print("Conclusion:")
    print("  We found emotion encoding! Just not where we were looking.")
    print("  Research is BACK ON TRACK with revised focus.")
    print()

elif verdict == "PARTIAL_IN_TEXT":
    print("⚠️  HYPOTHESIS PARTIALLY CONFIRMED")
    print()
    print("Emotion encoding is distributed across multiple levels.")
    print("Need multi-level analysis.")
    print()

else:
    print("❌ HYPOTHESIS REJECTED")
    print()
    print("Emotion is not primarily in text embeddings.")
    print("Must be in generation process or output.")
    print()

# ============================================================================
# Phase 6: Specific Emotion Pairs
# ============================================================================

print("=" * 80)
print("DETAILED ANALYSIS: Specific Emotion Pairs")
print("=" * 80)
print()

# Theoretical expectations
expected_similar = [('happy', 'energetic'), ('sad', 'calm')]
expected_different = [('happy', 'sad'), ('energetic', 'calm')]

print("Emotionally SIMILAR pairs (should have HIGH similarity):")
for e1, e2 in expected_similar:
    i, j = emotions.index(e1), emotions.index(e2)
    sim = similarity_matrix[i, j]
    print(f"  {e1} - {e2}: {sim:.4f}")
print()

print("Emotionally DIFFERENT pairs (should have LOW similarity):")
for e1, e2 in expected_different:
    i, j = emotions.index(e1), emotions.index(e2)
    sim = similarity_matrix[i, j]
    print(f"  {e1} - {e2}: {sim:.4f}")
print()

# Check if pattern matches expectations
similar_avg = np.mean([similarity_matrix[emotions.index(e1), emotions.index(e2)]
                       for e1, e2 in expected_similar])
different_avg = np.mean([similarity_matrix[emotions.index(e1), emotions.index(e2)]
                         for e1, e2 in expected_different])

print(f"Average similarity (expected similar): {similar_avg:.4f}")
print(f"Average similarity (expected different): {different_avg:.4f}")
print()

if similar_avg > different_avg + 0.02:
    print("✅ Pattern matches expectations!")
    print("   T5 embeddings capture emotional relationships correctly.")
else:
    print("⚠️  Pattern unclear or reversed.")
    print("   T5 might not understand emotional semantics well.")

# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("Key Results:")
print(f"  1. Within-emotion similarity: {mean_within:.4f}")
print(f"  2. Between-emotion similarity: {mean_between:.4f}")
print(f"  3. Difference: {difference:.4f}")
print(f"  4. Statistical significance: p = {p_value:.6f}")
print(f"  5. Verdict: {verdict}")
print()

print("What this means for your research:")
if verdict == "FOUND_IN_TEXT":
    print("  ✅ Emotion encoding IS STRONG")
    print("  ✅ Located in TEXT EMBEDDINGS")
    print("  ✅ Research can proceed (revised target)")
    print("  ✅ Phase 1: SAEs on T5 embeddings")
elif verdict == "PARTIAL_IN_TEXT":
    print("  ⚠️  Emotion encoding is DISTRIBUTED")
    print("  ⚠️  Need multi-level analysis")
    print("  ⚠️  Combine text + transformer + attention")
else:
    print("  ⚠️  Emotion encoding NOT in text")
    print("  ⚠️  Must be in generation or output")
    print("  ⚠️  Continue comprehensive search")

print()
print("This test took 10 minutes and provided crucial insight.")
print("We should have run this FIRST.")
print()

# Save results
output = {
    'within_mean': float(mean_within),
    'within_std': float(std_within),
    'between_mean': float(mean_between),
    'between_std': float(std_between),
    'difference': float(difference),
    'p_value': float(p_value),
    'verdict': verdict,
    'similarity_matrix': similarity_matrix.tolist(),
    'emotions': emotions,
}

import json
from pathlib import Path
output_file = "results/text_embedding_analysis.json"
Path("results").mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: {output_file}")
print()
print("=" * 80)
