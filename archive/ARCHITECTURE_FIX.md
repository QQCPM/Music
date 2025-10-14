# MusicGen Architecture Fix

## Issue

The initial code assumed MusicGen's transformer layers were at `model.lm.layers`, but this was incorrect and caused an `AttributeError`.

## Root Cause

After careful inspection of the actual MusicGen model structure, the correct path is:
```python
model.lm.transformer.layers
```

## MusicGen Architecture Hierarchy

```
MusicGen
├── lm (LMModel)
│   ├── transformer (StreamingTransformer)
│   │   ├── layers (ModuleList) ← The transformer layers are HERE
│   │   │   ├── [0] StreamingTransformerLayer
│   │   │   ├── [1] StreamingTransformerLayer
│   │   │   ├── ...
│   │   │   └── [23] StreamingTransformerLayer (24 layers total)
│   │   ├── positional_embedding
│   │   └── ...
│   ├── emb (embedding layer)
│   ├── linears (output layers)
│   └── ...
├── compression_model (EnCodec)
└── ...
```

## What Was Fixed

### 1. `src/utils/activation_utils.py`

**Before:**
```python
num_layers = len(model.lm.layers)  # ❌ Wrong path
layer = self.model.lm.layers[layer_idx]  # ❌ Wrong path
```

**After:**
```python
num_layers = len(model.lm.transformer.layers)  # ✅ Correct path
layer = self.model.lm.transformer.layers[layer_idx]  # ✅ Correct path
```

### 2. `notebooks/00_quick_test.ipynb`

**Before:**
```python
print(f"Number of layers: {len(model.lm.layers)}")  # ❌ Wrong
```

**After:**
```python
print(f"Number of layers: {len(model.lm.transformer.layers)}")  # ✅ Correct
```

## How to Access MusicGen Components

### Transformer Layers
```python
# Get number of layers
num_layers = len(model.lm.transformer.layers)  # 24 for all MusicGen models

# Access specific layer
layer_12 = model.lm.transformer.layers[12]

# Iterate through all layers
for i, layer in enumerate(model.lm.transformer.layers):
    print(f"Layer {i}: {type(layer)}")
```

### Individual Layer Components
Each `StreamingTransformerLayer` has:
- `self_attn` - Self-attention mechanism
- `cross_attention` - Cross-attention (for conditioning)
- `norm1`, `norm2`, `norm_cross` - Layer normalization
- `linear1`, `linear2` - Feed-forward network
- `dropout`, `dropout1`, `dropout2` - Dropout layers

### Other Model Components
```python
# Language model
model.lm  # The full LMModel

# Transformer
model.lm.transformer  # StreamingTransformer

# Embedding layer
model.lm.emb  # Token embeddings

# Output layers
model.lm.linears  # Output projection layers

# Compression model (EnCodec)
model.compression_model  # Audio encoder/decoder

# Text conditioning
model.lm.condition_provider  # Processes text prompts
```

## Key Model Properties

### For All MusicGen Variants (Small, Medium, Large)
- **Number of layers**: 24 (consistent across all sizes)
- **Model differences**:
  - Small (300M): `d_model=1024`
  - Medium (1.5B): `d_model=1536`
  - Large (3.3B): `d_model=2048`

### Activation Shapes
When extracting activations:
```python
# Shape: [num_codebooks, batch_size, sequence_length, d_model]
# For single generation:
# - num_codebooks: 2 (for stereo) or 4 (for mono at different levels)
# - batch_size: 1 (for single prompt)
# - sequence_length: varies based on duration
# - d_model: 1024/1536/2048 depending on model size
```

## Verification

Run the test script to verify everything works:
```bash
python3 test_fixed_architecture.py
```

Expected output:
```
✅ Successfully accessed layers
   Number of layers: 24
✅ ActivationExtractor created successfully
✅ Generation and extraction successful
   layer_0: torch.Size([2, 1, 1024])
   layer_12: torch.Size([2, 1, 1024])
   ...
```

## Why This Matters for Your Research

Understanding the correct architecture is crucial because:

1. **Activation Extraction**: You need to hook into the right layers to capture internal representations

2. **Layer Selection**: Different layers capture different information:
   - Early layers (0-6): Low-level acoustic features
   - Middle layers (7-17): Musical patterns and structure
   - Late layers (18-24): High-level concepts like emotion

3. **Intervention Points**: For activation steering, you need to modify activations at the right architectural level

4. **Reproducibility**: Correct paths ensure your code works for others

## Common Mistakes to Avoid

❌ **Don't do this:**
```python
model.lm.layers  # This doesn't exist!
model.transformer  # Missing .lm
model.layers  # Missing .lm.transformer
```

✅ **Do this:**
```python
model.lm.transformer.layers  # Correct!
```

## Additional Resources

- **AudioCraft docs**: https://github.com/facebookresearch/audiocraft
- **MusicGen paper**: https://arxiv.org/abs/2306.05284
- **Test script**: `test_fixed_architecture.py`

---

**Fixed by**: Systematic architecture inspection using `dir()` and `type()` exploration
**Tested with**: MusicGen Small, Medium, and Large models
**Status**: ✅ All tests passing
