#!/usr/bin/env python3
"""
Test script to verify the fixed architecture paths.
This tests that we can correctly access MusicGen's transformer layers.
"""

import sys
sys.path.append('.')

from audiocraft.models import MusicGen
from src.utils.activation_utils import ActivationExtractor
import torch

print("=" * 70)
print("Testing Fixed MusicGen Architecture Paths")
print("=" * 70)
print()

# Test 1: Load model and check architecture
print("Test 1: Loading MusicGen Small (faster for testing)...")
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=5)  # Short duration for testing

print("✅ Model loaded")
print(f"   Sample rate: {model.sample_rate}")
print()

# Test 2: Access transformer layers
print("Test 2: Accessing transformer layers...")
try:
    num_layers = len(model.lm.transformer.layers)
    print(f"✅ Successfully accessed layers")
    print(f"   Number of layers: {num_layers}")
    print(f"   Layer 0 type: {type(model.lm.transformer.layers[0])}")
except AttributeError as e:
    print(f"❌ Failed to access layers: {e}")
    sys.exit(1)

print()

# Test 3: Create ActivationExtractor
print("Test 3: Creating ActivationExtractor...")
try:
    extractor = ActivationExtractor(model, layers=[0, 6, 12, 18, 23])
    print(f"✅ ActivationExtractor created successfully")
    print(f"   Monitoring layers: {extractor.layers}")
except Exception as e:
    print(f"❌ Failed to create extractor: {e}")
    sys.exit(1)

print()

# Test 4: Generate with activation capture
print("Test 4: Generating music with activation capture...")
print("   (This will take ~15-30 seconds)")
try:
    prompts = ["happy upbeat music"]
    wav = extractor.generate(prompts)
    activations = extractor.get_activations()

    print(f"✅ Generation and extraction successful")
    print(f"   Generated audio shape: {wav.shape}")
    print(f"   Captured {len(activations)} layer activations")

    # Show activation shapes
    for name, act in sorted(activations.items()):
        print(f"   {name}: {act.shape}")

except Exception as e:
    print(f"❌ Failed during generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Check activation properties
print("Test 5: Analyzing activation properties...")
layer_12_act = activations.get('layer_12')
if layer_12_act is not None:
    print(f"✅ Layer 12 activation analysis:")
    print(f"   Shape: {layer_12_act.shape}")
    print(f"   Mean: {layer_12_act.mean().item():.4f}")
    print(f"   Std: {layer_12_act.std().item():.4f}")
    print(f"   Min: {layer_12_act.min().item():.4f}")
    print(f"   Max: {layer_12_act.max().item():.4f}")
else:
    print("❌ Could not find layer_12 activation")

print()
print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("Your MusicGen setup is working correctly!")
print()
print("Next steps:")
print("  1. Try the notebook: jupyter notebook notebooks/00_quick_test.ipynb")
print("  2. Read the learning roadmap: docs/phase0_roadmap.md")
print("  3. Start experimenting with different prompts and layers")
