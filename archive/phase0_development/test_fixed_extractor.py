#!/usr/bin/env python3
"""
Test the FIXED ActivationExtractor to verify it captures full sequences.
"""

import sys
sys.path.append('.')

import torch
from audiocraft.models import MusicGen
from src.utils.activation_utils import ActivationExtractor, cosine_similarity
import numpy as np

print("=" * 80)
print("Testing FIXED ActivationExtractor")
print("=" * 80)
print()

# Load model (use small for faster testing)
print("Loading MusicGen Small...")
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=3)  # 3 seconds
print("✅ Model loaded")
print()

# Create extractor
print("Creating ActivationExtractor for layers [0, 12, 23]...")
extractor = ActivationExtractor(model, layers=[0, 12, 23])
print("✅ Extractor created")
print()

# Generate TWO different emotions
print("=" * 80)
print("Test 1: Generate Happy Music")
print("=" * 80)
prompt_happy = "upbeat cheerful happy energetic music"
print(f"Prompt: '{prompt_happy}'")
wav_happy = extractor.generate([prompt_happy])
activations_happy = extractor.get_activations(concatenate=True)

print()
print("Captured activations (HAPPY):")
for name, act in activations_happy.items():
    print(f"  {name}: {act.shape}")
    print(f"    Mean: {act.mean():.4f}, Std: {act.std():.4f}")
    print(f"    Min: {act.min():.4f}, Max: {act.max():.4f}")

print()
print("=" * 80)
print("Test 2: Generate Sad Music")
print("=" * 80)
extractor.clear_activations()
prompt_sad = "melancholic sad sorrowful depressing music"
print(f"Prompt: '{prompt_sad}'")
wav_sad = extractor.generate([prompt_sad])
activations_sad = extractor.get_activations(concatenate=True)

print()
print("Captured activations (SAD):")
for name, act in activations_sad.items():
    print(f"  {name}: {act.shape}")
    print(f"    Mean: {act.mean():.4f}, Std: {act.std():.4f}")
    print(f"    Min: {act.min():.4f}, Max: {act.max():.4f}")

print()
print("=" * 80)
print("Test 3: Compute Similarity Between Emotions")
print("=" * 80)

# Compare layer 12 (middle layer)
happy_layer12 = activations_happy['layer_12']
sad_layer12 = activations_sad['layer_12']

print(f"Happy layer 12 shape: {happy_layer12.shape}")
print(f"Sad layer 12 shape: {sad_layer12.shape}")
print()

# Compute cosine similarity
similarity = cosine_similarity(happy_layer12, sad_layer12)
print(f"Cosine similarity (layer 12): {similarity:.6f}")
print()

# Interpret result
if similarity > 0.99:
    print("🔴 STILL TOO SIMILAR (> 0.99)")
    print("   Possible reasons:")
    print("   1. Model representations truly don't differentiate emotions much")
    print("   2. Need to look at specific timesteps, not average over all")
    print("   3. Need to analyze WHICH dimensions differ")
elif similarity > 0.95:
    print("⚠️  MODERATELY SIMILAR (0.95-0.99)")
    print("   Some differentiation exists but weak")
elif similarity > 0.8:
    print("✅ SOMEWHAT DIFFERENT (0.8-0.95)")
    print("   Emotions are represented differently!")
else:
    print("✅✅ VERY DIFFERENT (< 0.8)")
    print("   Strong emotion differentiation!")

print()
print("=" * 80)
print("Test 4: Analyze Temporal Dynamics")
print("=" * 80)

# Look at similarity across time
print("Computing similarity at different timesteps...")

num_timesteps = happy_layer12.shape[0]
print(f"Total timesteps: {num_timesteps}")

# Sample 5 timesteps evenly
timestep_indices = np.linspace(0, num_timesteps-1, 5, dtype=int)
print(f"Sampling timesteps: {timestep_indices.tolist()}")
print()

timestep_similarities = []
for t in timestep_indices:
    happy_t = happy_layer12[t]  # Shape: [num_codebooks, batch, d_model]
    sad_t = sad_layer12[t]
    sim_t = cosine_similarity(happy_t, sad_t)
    timestep_similarities.append(sim_t)
    print(f"  Timestep {t:3d}: similarity = {sim_t:.6f}")

print()
print(f"Mean similarity across sampled timesteps: {np.mean(timestep_similarities):.6f}")
print(f"Std of similarity across timesteps: {np.std(timestep_similarities):.6f}")

if np.std(timestep_similarities) > 0.01:
    print("✅ Similarity VARIES across time - this is expected!")
else:
    print("⚠️  Similarity is CONSTANT across time - may indicate issue")

print()
print("=" * 80)
print("Test 5: Analyze Which Dimensions Differ Most")
print("=" * 80)

# Compute element-wise difference
# Average over timesteps first
happy_mean = happy_layer12.mean(dim=0)  # [num_codebooks, batch, d_model]
sad_mean = sad_layer12.mean(dim=0)

# Compute absolute difference for each dimension
diff = (happy_mean - sad_mean).abs()  # [num_codebooks, batch, d_model]

# Flatten to [d_model]
diff_flat = diff.flatten()

# Find top 10 most different dimensions
top_k = 10
top_diffs, top_indices = torch.topk(diff_flat, k=top_k)

print(f"Top {top_k} most different dimensions:")
for i, (idx, val) in enumerate(zip(top_indices, top_diffs)):
    print(f"  {i+1}. Dimension {idx.item()}: difference = {val.item():.4f}")

print()
print(f"Mean difference across all dimensions: {diff_flat.mean().item():.4f}")
print(f"Max difference: {diff_flat.max().item():.4f}")
print(f"% of dimensions with diff > 0.1: {(diff_flat > 0.1).float().mean().item()*100:.2f}%")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()
print("Key Findings:")
print(f"1. Activation shape: {happy_layer12.shape}")
print(f"   - Timesteps: {happy_layer12.shape[0]} (was 459, depends on duration)")
print(f"   - Codebooks: {happy_layer12.shape[1]}")
print(f"   - Batch: {happy_layer12.shape[2]}")
print(f"   - d_model: {happy_layer12.shape[3]}")
print()
print(f"2. Overall similarity: {similarity:.6f}")
print(f"3. Temporal variation in similarity: {np.std(timestep_similarities):.6f}")
print(f"4. Max dimension difference: {diff_flat.max().item():.4f}")
print()

if similarity < 0.95:
    print("✅ SUCCESS: Emotions are represented differently!")
    print("   The fix worked - we're now capturing meaningful activations.")
else:
    print("⚠️  Emotions still very similar")
    print("   This might be a real finding - MusicGen may encode emotions weakly")
    print("   OR we need to:")
    print("   - Look at different layers")
    print("   - Analyze specific timesteps")
    print("   - Use more sophisticated metrics (e.g., CCA, SVCCA)")

print()
print("Next steps:")
print("1. Generate more samples (20+ per emotion)")
print("2. Analyze ALL layers, not just layer 12")
print("3. Use UMAP to visualize emotion clustering")
print("4. Extract acoustic features to validate audio quality")
print()

# Save results
output_file = "results/fixed_extractor_test_results.txt"
with open(output_file, 'w') as f:
    f.write("Fixed ActivationExtractor Test Results\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Happy activation shape: {happy_layer12.shape}\n")
    f.write(f"Sad activation shape: {sad_layer12.shape}\n")
    f.write(f"\nCosine similarity (layer 12): {similarity:.6f}\n")
    f.write(f"Mean similarity across timesteps: {np.mean(timestep_similarities):.6f}\n")
    f.write(f"Std similarity: {np.std(timestep_similarities):.6f}\n")
    f.write(f"\nMax dimension difference: {diff_flat.max().item():.4f}\n")
    f.write(f"Mean dimension difference: {diff_flat.mean().item():.4f}\n")

print(f"Results saved to: {output_file}")
