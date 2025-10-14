#!/usr/bin/env python3
"""
Deep diagnostic script to understand what's actually happening during MusicGen generation.
This will help us understand why activations show 0.9999 similarity.
"""

import sys
sys.path.append('.')

import torch
from audiocraft.models import MusicGen
from pathlib import Path
import numpy as np

print("=" * 80)
print("DEEP DIAGNOSTIC: MusicGen Activation Extraction")
print("=" * 80)
print()

# Load model
print("Step 1: Loading MusicGen Large...")
model = MusicGen.get_pretrained('facebook/musicgen-small')  # Use small for faster testing
model.set_generation_params(duration=3)  # Short duration for debugging
print(f"✅ Model loaded: {model.__class__.__name__}")
print()

# Investigate architecture
print("Step 2: Architecture Investigation")
print("-" * 80)
# Get device from the actual model parameters (lm model)
if hasattr(model, 'lm'):
    print(f"Model device: {next(model.lm.parameters()).device}")
print()

# Check the actual structure
print("Checking model.lm structure:")
print(f"  model.lm type: {type(model.lm)}")
print(f"  model.lm.transformer type: {type(model.lm.transformer)}")

# Count actual layers
if hasattr(model.lm, 'transformer'):
    if hasattr(model.lm.transformer, 'layers'):
        num_layers = len(model.lm.transformer.layers)
        print(f"  Number of transformer layers: {num_layers}")
        print(f"  Layer 0 type: {type(model.lm.transformer.layers[0])}")
    else:
        print("  ❌ No 'layers' attribute found in transformer")
        print(f"  Available attributes: {dir(model.lm.transformer)}")
else:
    print("  ❌ No 'transformer' attribute found in lm")
    print(f"  Available attributes: {dir(model.lm)}")

print()

# Investigate what happens during generation
print("Step 3: Tracing Generation Process")
print("-" * 80)

# Track ALL forward passes
forward_passes = []
activation_shapes = {}

def make_debug_hook(name):
    """Create a hook that logs every forward pass"""
    def hook(module, input, output):
        # Record shape and statistics
        if isinstance(output, torch.Tensor):
            shape = output.shape
            forward_passes.append({
                'name': name,
                'shape': shape,
                'mean': output.detach().float().mean().item(),
                'std': output.detach().float().std().item(),
                'device': str(output.device)
            })

            # Store for later analysis
            if name not in activation_shapes:
                activation_shapes[name] = []
            activation_shapes[name].append(shape)
        elif isinstance(output, tuple):
            # Some layers return tuples
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    forward_passes.append({
                        'name': f"{name}_output_{i}",
                        'shape': out.shape,
                        'mean': out.detach().float().mean().item(),
                        'std': out.detach().float().std().item(),
                        'device': str(out.device)
                    })
    return hook

# Register hooks on multiple layers
layers_to_monitor = [0, 6, 12] if num_layers > 12 else [0, num_layers//2, num_layers-1]
hooks = []

print(f"Registering hooks on layers: {layers_to_monitor}")
for layer_idx in layers_to_monitor:
    hook = model.lm.transformer.layers[layer_idx].register_forward_hook(
        make_debug_hook(f'layer_{layer_idx}')
    )
    hooks.append(hook)

print()

# Generate TWO different prompts
print("Generating music with TWO different prompts...")
print()

prompts = [
    "happy upbeat cheerful music",
    "sad melancholic sorrowful music"
]

results = {}

for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    forward_passes.clear()  # Clear previous passes

    with torch.no_grad():
        wav = model.generate([prompt], progress=True)

    # Analyze what happened
    print(f"  Total forward passes: {len(forward_passes)}")

    # Get unique shapes seen
    unique_shapes = {}
    for fp in forward_passes:
        shape_str = str(fp['shape'])
        if shape_str not in unique_shapes:
            unique_shapes[shape_str] = 0
        unique_shapes[shape_str] += 1

    print(f"  Unique activation shapes seen:")
    for shape, count in unique_shapes.items():
        print(f"    {shape}: {count} times")

    # Store statistics
    results[prompt] = {
        'forward_passes': len(forward_passes),
        'shapes': unique_shapes,
        'first_activation': forward_passes[0] if forward_passes else None,
        'last_activation': forward_passes[-1] if forward_passes else None,
        'audio_shape': wav.shape
    }

    print(f"  Generated audio shape: {wav.shape}")
    print()

# Remove hooks
for hook in hooks:
    hook.remove()

print()
print("Step 4: Analysis of Results")
print("-" * 80)

# Compare the two generations
prompt1, prompt2 = prompts[0], prompts[1]
result1, result2 = results[prompt1], results[prompt2]

print(f"Comparison:")
print(f"  {prompt1}:")
print(f"    Forward passes: {result1['forward_passes']}")
print(f"    Audio shape: {result1['audio_shape']}")
print()
print(f"  {prompt2}:")
print(f"    Forward passes: {result2['forward_passes']}")
print(f"    Audio shape: {result2['audio_shape']}")
print()

# Critical insight: Are we seeing multiple forward passes?
if result1['forward_passes'] == result2['forward_passes']:
    print(f"✅ Both prompts triggered {result1['forward_passes']} forward passes")
    if result1['forward_passes'] == 1:
        print("⚠️  WARNING: Only 1 forward pass detected!")
        print("   This suggests we're only capturing ONE timestep/token")
        print("   This explains the 0.9999 similarity!")
    elif result1['forward_passes'] < 10:
        print(f"⚠️  WARNING: Only {result1['forward_passes']} forward passes")
        print("   Expected many more for autoregressive generation")
else:
    print(f"❌ Different number of forward passes: {result1['forward_passes']} vs {result2['forward_passes']}")

print()
print("Step 5: Understanding Activation Shapes")
print("-" * 80)

print("Looking at activation_shapes collected:")
for layer_name, shapes in activation_shapes.items():
    print(f"\n{layer_name}:")
    print(f"  Number of times forward() called: {len(shapes)}")
    print(f"  Shapes seen: {set(str(s) for s in shapes)}")

    # Are shapes changing over time? (autoregressive)
    if len(shapes) > 1:
        first_shape = shapes[0]
        last_shape = shapes[-1]
        if first_shape != last_shape:
            print(f"  ✅ Shape CHANGES during generation: {first_shape} → {last_shape}")
            print(f"     This indicates autoregressive token generation")
        else:
            print(f"  ⚠️  Shape SAME throughout: {first_shape}")
    else:
        print(f"  ⚠️  Only called once!")

print()
print("=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print()

# Key findings
print("Key Findings:")
print()

if result1['forward_passes'] == 1:
    print("🔴 CRITICAL ISSUE: Only 1 forward pass per generation")
    print("   Your hooks are capturing a SINGLE snapshot, not the full sequence")
    print("   This is why activations are 99.99% similar!")
    print()
    print("   The Problem:")
    print("   - MusicGen generates autoregressively (token by token)")
    print("   - You're only seeing ONE token's activation")
    print("   - Different prompts likely have similar initial states")
    print()
    print("   The Solution:")
    print("   - Need to capture activations at EACH generation step")
    print("   - Store them in a list or concatenate them")
    print("   - Analyze the FULL sequence, not just one timestep")
else:
    print(f"✅ Multiple forward passes detected: {result1['forward_passes']}")
    print("   But we need to verify we're STORING all of them...")

print()
print("Next Steps:")
print("1. Fix ActivationExtractor to capture ALL timesteps, not just the last one")
print("2. Re-run experiment with corrected extraction")
print("3. Compute similarity on FULL sequences")
print()

# Save diagnostic results
output_file = "results/diagnostic_report.txt"
Path("results").mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    f.write("MusicGen Activation Extraction Diagnostic Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Model: MusicGen Small\n")
    f.write(f"Layers monitored: {layers_to_monitor}\n\n")

    for prompt, result in results.items():
        f.write(f"Prompt: {prompt}\n")
        f.write(f"  Forward passes: {result['forward_passes']}\n")
        f.write(f"  Audio shape: {result['audio_shape']}\n")
        f.write(f"  Shapes seen: {result['shapes']}\n\n")

    f.write("\nConclusion:\n")
    if result1['forward_passes'] == 1:
        f.write("ISSUE: Only capturing 1 timestep. Need to fix ActivationExtractor.\n")
    else:
        f.write(f"Capturing {result1['forward_passes']} timesteps. Verify storage mechanism.\n")

print(f"Diagnostic report saved to: {output_file}")
