#!/usr/bin/env python3
"""
CRITICAL VALIDATION: Are the emotion differentiation results real or artifacts?

This script performs skeptical analysis to verify:
1. Control test: Identical prompts → ~1.0 similarity (sanity check)
2. Audio analysis: Do files actually sound different?
3. Randomness test: Random activations → ~0.0 similarity
4. Dimension analysis: Is 8% differentiation statistically significant?
5. Temporal consistency: Are differences consistent across time?
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from audiocraft.models import MusicGen
from src.utils.activation_utils import ActivationExtractor, cosine_similarity
from src.utils.audio_utils import extract_audio_features
from pathlib import Path
import json

print("=" * 80)
print("CRITICAL VALIDATION: Are Emotion Differentiation Results Real?")
print("=" * 80)
print()

# Load model
print("Loading MusicGen Small (faster for testing)...")
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=3)
print("✅ Model loaded")
print()

# ============================================================================
# TEST 1: CONTROL - Identical Prompts Should Give ~1.0 Similarity
# ============================================================================
print("=" * 80)
print("TEST 1: Control - Identical Prompts (Sanity Check)")
print("=" * 80)
print()
print("Hypothesis: Same prompt twice → similarity ≈ 1.0")
print("If this fails, our similarity metric is broken!")
print()

extractor = ActivationExtractor(model, layers=[12])

# Generate same prompt twice
prompt_control = "upbeat happy energetic music"
print(f"Generating: '{prompt_control}' (attempt 1)")
wav1 = extractor.generate([prompt_control])
acts1 = extractor.get_activations(concatenate=True)

extractor.clear_activations()
print(f"Generating: '{prompt_control}' (attempt 2)")
wav2 = extractor.generate([prompt_control])
acts2 = extractor.get_activations(concatenate=True)

sim_control = cosine_similarity(acts1['layer_12'], acts2['layer_12'])
print()
print(f"Similarity (same prompt, 2 generations): {sim_control:.6f}")
print()

# Interpret
if sim_control > 0.99:
    print("✅ PASS: Identical prompts give high similarity (>0.99)")
    print("   This validates our similarity metric is working correctly.")
elif sim_control > 0.95:
    print("⚠️  BORDERLINE: Similarity is 0.95-0.99")
    print("   Some stochasticity expected, but should be higher.")
    print("   MusicGen generation may be non-deterministic.")
elif sim_control > 0.90:
    print("⚠️  WARNING: Similarity only 0.90-0.95 for identical prompts")
    print("   This suggests high variance in generation.")
    print("   Our emotion similarity (0.9461) may not be meaningful!")
else:
    print("🔴 FAIL: Identical prompts give low similarity (<0.90)")
    print("   This invalidates our entire methodology!")
    print("   Something is fundamentally wrong.")

print()
print(f"Context: Our happy vs. sad similarity was 0.9461")
print(f"Difference from control: {abs(sim_control - 0.9461):.4f}")
print()

if sim_control > 0.95:
    control_passed = True
    print("✅ Control test PASSED - metric is valid")
else:
    control_passed = False
    print("❌ Control test FAILED - results may not be meaningful")

print()

# ============================================================================
# TEST 2: RANDOM BASELINE - Random Activations Should Give ~0.0 Similarity
# ============================================================================
print("=" * 80)
print("TEST 2: Random Baseline")
print("=" * 80)
print()
print("Hypothesis: Random tensors → similarity ≈ 0.0")
print("This establishes the 'no relationship' baseline.")
print()

# Create random tensors with same shape as our activations
shape = acts1['layer_12'].shape
random1 = torch.randn(shape)
random2 = torch.randn(shape)

sim_random = cosine_similarity(random1, random2)
print(f"Similarity (random tensors): {sim_random:.6f}")
print()

if abs(sim_random) < 0.1:
    print("✅ PASS: Random tensors give near-zero similarity")
    print("   This is expected for uncorrelated data.")
else:
    print("⚠️  Unexpected: Random tensors have high similarity")
    print("   This might indicate an issue with our similarity computation.")

print()
print(f"Range: Random baseline ({sim_random:.3f}) to Control ({sim_control:.3f})")
print(f"Our result (0.9461) is {(0.9461 - sim_random)/(sim_control - sim_random)*100:.1f}% toward identical")
print()

# ============================================================================
# TEST 3: ACOUSTIC FEATURE VALIDATION - Do Audio Files Actually Differ?
# ============================================================================
print("=" * 80)
print("TEST 3: Acoustic Feature Validation")
print("=" * 80)
print()
print("Checking if generated audio files actually sound different...")
print("If happy and sad music have similar acoustic features,")
print("then our activation differences might not matter!")
print()

# Check existing audio files
audio_files = {
    'happy': 'results/sample_happy.wav',
    'sad': 'results/sample_sad.wav',
    'calm': 'results/sample_calm.wav',
    'energetic': 'results/sample_energetic.wav'
}

acoustic_features = {}
for emotion, filepath in audio_files.items():
    if Path(filepath).exists():
        print(f"Analyzing {emotion} music: {filepath}")
        features = extract_audio_features(filepath, sr=32000, duration=8.0)
        acoustic_features[emotion] = features

        print(f"  Tempo: {features['tempo']:.1f} BPM")
        print(f"  Spectral centroid: {features['spectral_centroid_mean']:.1f} Hz")
        print(f"  RMS energy: {features['rms_mean']:.4f}")
        print(f"  Mode: {features['mode']}")
        print()
    else:
        print(f"⚠️  File not found: {filepath}")

# Compare happy vs. sad
if 'happy' in acoustic_features and 'sad' in acoustic_features:
    happy = acoustic_features['happy']
    sad = acoustic_features['sad']

    print("Comparison: Happy vs. Sad")
    print(f"  Tempo difference: {happy['tempo'] - sad['tempo']:.1f} BPM")
    print(f"  Energy difference: {happy['rms_mean'] - sad['rms_mean']:.4f}")
    print(f"  Brightness difference: {happy['spectral_centroid_mean'] - sad['spectral_centroid_mean']:.1f} Hz")
    print()

    # Check if differences are meaningful
    tempo_diff = abs(happy['tempo'] - sad['tempo'])
    energy_diff = abs(happy['rms_mean'] - sad['rms_mean'])

    if tempo_diff > 10 or energy_diff > 0.01:
        print("✅ PASS: Acoustic features show clear differences")
        print("   The audio files actually sound different!")
        acoustic_valid = True
    else:
        print("🔴 FAIL: Acoustic features are too similar")
        print("   MusicGen might not be generating different emotions!")
        print("   Our activation differences might be meaningless.")
        acoustic_valid = False
else:
    print("⚠️  Cannot validate - audio files missing")
    acoustic_valid = None

print()

# ============================================================================
# TEST 4: STATISTICAL SIGNIFICANCE - Is 8% Differentiation Real?
# ============================================================================
print("=" * 80)
print("TEST 4: Statistical Significance of Dimension Differences")
print("=" * 80)
print()
print("Question: Is 8% of dimensions showing >0.1 difference statistically meaningful?")
print()

# Generate new happy and sad samples
extractor.clear_activations()
wav_happy = extractor.generate(["happy upbeat cheerful music"])
acts_happy = extractor.get_activations(concatenate=True)

extractor.clear_activations()
wav_sad = extractor.generate(["sad melancholic sorrowful music"])
acts_sad = extractor.get_activations(concatenate=True)

# Compare dimension-wise
happy_mean = acts_happy['layer_12'].mean(dim=0).flatten()  # [d_model]
sad_mean = acts_sad['layer_12'].mean(dim=0).flatten()

diff = (happy_mean - sad_mean).abs()

# Thresholds
threshold_01 = (diff > 0.1).float().mean().item()
threshold_02 = (diff > 0.2).float().mean().item()
threshold_03 = (diff > 0.3).float().mean().item()

print(f"Percentage of dimensions with difference:")
print(f"  > 0.1: {threshold_01*100:.2f}%")
print(f"  > 0.2: {threshold_02*100:.2f}%")
print(f"  > 0.3: {threshold_03*100:.2f}%")
print()

print(f"Mean difference across all dimensions: {diff.mean().item():.4f}")
print(f"Std of differences: {diff.std().item():.4f}")
print(f"Max difference: {diff.max().item():.4f}")
print()

# Statistical test: Is this better than random?
# Compare to random permutation
happy_permuted = happy_mean[torch.randperm(len(happy_mean))]
diff_random = (happy_permuted - sad_mean).abs()
threshold_random = (diff_random > 0.1).float().mean().item()

print(f"Control: Random permutation gives {threshold_random*100:.2f}% above 0.1")
print(f"Actual: We observe {threshold_01*100:.2f}% above 0.1")
print()

if threshold_01 > threshold_random * 1.5:
    print("✅ PASS: Significantly more differentiation than random")
    stats_valid = True
elif threshold_01 > threshold_random * 1.2:
    print("⚠️  BORDERLINE: Slightly more than random, but weak signal")
    stats_valid = None
else:
    print("🔴 FAIL: Not significantly different from random permutation")
    stats_valid = False

print()

# ============================================================================
# TEST 5: TEMPORAL CONSISTENCY - Are Differences Consistent Over Time?
# ============================================================================
print("=" * 80)
print("TEST 5: Temporal Consistency")
print("=" * 80)
print()
print("Question: Do differences persist across the generation sequence?")
print("If differences are only at start/end, they might be artifacts.")
print()

# Get full tensors
happy_full = acts_happy['layer_12']  # [timesteps, codebooks, batch, d_model]
sad_full = acts_sad['layer_12']

num_timesteps = happy_full.shape[0]
print(f"Total timesteps: {num_timesteps}")
print()

# Sample 10 timesteps evenly
sample_indices = np.linspace(0, num_timesteps-1, 10, dtype=int)
print(f"Sampling timesteps: {sample_indices.tolist()}")
print()

similarities_over_time = []
for t in sample_indices:
    happy_t = happy_full[t]
    sad_t = sad_full[t]
    sim_t = cosine_similarity(happy_t, sad_t)
    similarities_over_time.append(sim_t)
    print(f"  t={t:3d}: similarity = {sim_t:.6f}")

print()
mean_sim = np.mean(similarities_over_time)
std_sim = np.std(similarities_over_time)
min_sim = np.min(similarities_over_time)
max_sim = np.max(similarities_over_time)

print(f"Mean similarity over time: {mean_sim:.6f}")
print(f"Std similarity: {std_sim:.6f}")
print(f"Range: [{min_sim:.6f}, {max_sim:.6f}]")
print()

# Is it consistent?
if std_sim < 0.02:
    print("⚠️  Very consistent (std < 0.02) - differences might be initial state")
    temporal_valid = None
elif std_sim > 0.05:
    print("✅ PASS: High variation (std > 0.05) - differences evolve over time")
    temporal_valid = True
else:
    print("✅ MODERATE: Some variation - differences present throughout")
    temporal_valid = True

# Check if early timesteps are more similar (conditioning artifact)
early_sim = np.mean(similarities_over_time[:3])
late_sim = np.mean(similarities_over_time[-3:])
print()
print(f"Early timesteps (0-33%): {early_sim:.6f}")
print(f"Late timesteps (67-100%): {late_sim:.6f}")
print(f"Difference: {abs(early_sim - late_sim):.6f}")

if abs(early_sim - late_sim) < 0.01:
    print("  → Consistent throughout generation")
elif early_sim > late_sim:
    print("  → Emotions DIVERGE during generation (good!)")
else:
    print("  → Emotions CONVERGE during generation (unexpected)")

print()

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("=" * 80)
print("FINAL VERDICT: Are the Results Real?")
print("=" * 80)
print()

results_summary = {
    'control_test': control_passed,
    'acoustic_validation': acoustic_valid,
    'statistical_significance': stats_valid,
    'temporal_consistency': temporal_valid,
    'control_similarity': sim_control,
    'emotion_similarity': 0.9461,  # From earlier test
    'random_baseline': sim_random,
}

print("Test Results:")
print(f"  1. Control (same prompt):          {control_passed if control_passed is not None else 'N/A'}")
print(f"  2. Acoustic validation:             {acoustic_valid if acoustic_valid is not None else 'N/A'}")
print(f"  3. Statistical significance:        {stats_valid if stats_valid is not None else 'N/A'}")
print(f"  4. Temporal consistency:            {temporal_valid if temporal_valid is not None else 'N/A'}")
print()

# Count passes
tests = [control_passed, acoustic_valid, stats_valid, temporal_valid]
passed = sum(1 for t in tests if t is True)
total = sum(1 for t in tests if t is not None)

print(f"Tests Passed: {passed}/{total}")
print()

# Overall verdict
if passed >= 3 and control_passed:
    print("✅✅✅ VERDICT: Results are LIKELY REAL")
    print()
    print("Evidence:")
    print("- Control test passes (sanity check works)")
    print("- Multiple validation tests pass")
    print("- Emotion differentiation (0.9461) is measurably different from control")
    print()
    print("Confidence: HIGH")
    print("Recommendation: Proceed with Phase 0 data collection")

elif passed >= 2:
    print("⚠️⚠️ VERDICT: Results are POSSIBLY REAL but WEAK")
    print()
    print("Concerns:")
    print("- Some validation tests fail or borderline")
    print("- Signal may be present but subtle")
    print("- Need more samples (N=1 is insufficient)")
    print()
    print("Confidence: MODERATE")
    print("Recommendation: Generate 20+ samples per emotion before concluding")

else:
    print("🔴🔴🔴 VERDICT: Results are LIKELY ARTIFACTS or METHODOLOGY IS FLAWED")
    print()
    print("Critical Issues:")
    print("- Control test fails OR")
    print("- Multiple validations fail")
    print("- Differences may not be meaningful")
    print()
    print("Confidence: LOW")
    print("Recommendation: Debug methodology before proceeding")

print()

# Save results
output_file = "results/critical_validation_results.json"
Path("results").mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"Results saved to: {output_file}")

print()
print("=" * 80)
print("CRITICAL QUESTIONS TO ANSWER:")
print("=" * 80)
print()
print("1. Why is control similarity not 1.0?")
print("   → MusicGen generation is non-deterministic")
print("   → This is expected (sampling/stochasticity)")
print()
print("2. Is 0.9461 meaningfully different from control?")
print(f"   → Control: {sim_control:.4f}")
print(f"   → Emotion: 0.9461")
print(f"   → Difference: {abs(sim_control - 0.9461):.4f}")
print(f"   → This is {'SMALL' if abs(sim_control - 0.9461) < 0.02 else 'MEANINGFUL'}")
print()
print("3. Do we need more samples?")
print("   → YES! N=1 per emotion is insufficient")
print("   → Need 20+ samples to compute means and variances")
print("   → Current results are suggestive, not conclusive")
print()
print("4. What's the next critical test?")
print("   → Generate 5 happy samples + 5 sad samples")
print("   → Compute within-emotion vs. between-emotion similarities")
print("   → If between < within, signal is real")
print()
