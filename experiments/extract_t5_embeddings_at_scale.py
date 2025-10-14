"""
Extract T5 Embeddings at Scale
================================

Now that we know emotions ARE encoded in T5 text embeddings,
this script extracts embeddings for 100+ prompts to:

1. Validate the 21% differentiation holds at scale
2. Map which dimensions encode emotions
3. Prepare dataset for Phase 1 SAE training

Author: MusicGen Emotion Interpretability Research
"""

import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# ================================================================================
# Configuration
# ================================================================================

EMOTIONS = {
    'happy': [
        'happy cheerful upbeat music',
        'joyful energetic positive music',
        'euphoric celebration music',
        'bright sunny optimistic music',
        'playful fun lighthearted music',
        'excited enthusiastic music',
        'bouncy peppy spirited music',
        'delightful pleasant music',
        'radiant gleeful music',
        'carefree jolly music',
        'blissful content music',
        'lively animated music',
        'vibrant dynamic happy music',
        'jubilant festive music',
        'buoyant uplifting music',
        'cheerful whistling melody',
        'happy go lucky tune',
        'sunny day music',
        'celebration party music',
        'feel good vibes',
        'happy dancing music',
        'joyous triumph music',
        'happy children playing',
        'warm happy acoustic',
        'happy pop anthem'
    ],

    'sad': [
        'sad melancholic sorrowful music',
        'depressing mournful tragic music',
        'heartbreak funeral dirge music',
        'gloomy desolate music',
        'tearful weeping music',
        'lonely isolated music',
        'hopeless despair music',
        'grieving lamenting music',
        'woeful doleful music',
        'miserable anguished music',
        'somber solemn music',
        'downcast dejected music',
        'heavy hearted music',
        'forlorn abandoned music',
        'despondent music',
        'tragic loss music',
        'crying piano ballad',
        'rainy day sadness',
        'broken heart melody',
        'farewell goodbye music',
        'melancholic strings',
        'sad violin solo',
        'mourning requiem',
        'tears and sorrow',
        'sad acoustic guitar'
    ],

    'calm': [
        'calm peaceful relaxing music',
        'tranquil serene gentle music',
        'soothing meditation ambient music',
        'quiet still music',
        'placid restful music',
        'soft delicate music',
        'mellow smooth music',
        'subdued understated music',
        'contemplative reflective music',
        'hushed whispered music',
        'tender loving music',
        'dreamy floating music',
        'ethereal otherworldly music',
        'spa relaxation music',
        'slow peaceful piano',
        'calm ocean waves',
        'gentle lullaby',
        'peaceful forest ambience',
        'soft classical music',
        'zen meditation music',
        'calm breathing music',
        'quiet night music',
        'peaceful garden sounds',
        'calm acoustic guitar',
        'gentle harp melody'
    ],

    'energetic': [
        'energetic intense powerful music',
        'high energy driving forceful music',
        'adrenaline pumping aggressive music',
        'fierce vigorous music',
        'explosive dynamic music',
        'turbulent chaotic music',
        'frenzied wild music',
        'relentless pounding music',
        'hard hitting music',
        'thunderous booming music',
        'electrifying charged music',
        'fiery passionate music',
        'roaring powerful music',
        'intense workout music',
        'fast paced action music',
        'aggressive rock music',
        'high tempo electronic',
        'powerful orchestral battle',
        'intense drum and bass',
        'energetic running music',
        'fast metal guitar',
        'powerful bass drop',
        'intense game music',
        'energetic dance track',
        'powerful epic music'
    ]
}

OUTPUT_DIR = Path('results/t5_embeddings')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# Load Model
# ================================================================================

print("=" * 80)
print("T5 EMBEDDING EXTRACTION AT SCALE")
print("=" * 80)
print()

print("Loading T5-base encoder...")
tokenizer = T5Tokenizer.from_pretrained('t5-base')
encoder = T5EncoderModel.from_pretrained('t5-base')
encoder.eval()

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = encoder.to(device)
print(f"✅ Model loaded on {device}")
print()

# ================================================================================
# Extract Embeddings
# ================================================================================

print("=" * 80)
print("Extracting embeddings for 100 prompts...")
print("=" * 80)
print()

all_embeddings = []
all_labels = []
embeddings_by_emotion = {emotion: [] for emotion in EMOTIONS}

with torch.no_grad():
    for emotion_idx, (emotion, prompts) in enumerate(EMOTIONS.items()):
        print(f"\n{emotion.upper()} ({len(prompts)} prompts):")

        for prompt in tqdm(prompts, desc=f"  Processing"):
            # Tokenize
            tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Extract embedding
            output = encoder(**tokens)
            # Mean pool over sequence length
            embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu()

            # Store
            all_embeddings.append(embedding.numpy())
            all_labels.append(emotion_idx)
            embeddings_by_emotion[emotion].append(embedding.numpy())

# Convert to arrays
X = np.array(all_embeddings)  # Shape: (100, 768)
y = np.array(all_labels)      # Shape: (100,)

print(f"\n✅ Extracted {len(X)} embeddings")
print(f"   Shape: {X.shape}")
print()

# ================================================================================
# Analysis 1: Similarity Analysis
# ================================================================================

print("=" * 80)
print("ANALYSIS 1: Similarity Matrix")
print("=" * 80)
print()

# Compute average embedding per emotion
avg_embeddings = {}
for emotion, embs in embeddings_by_emotion.items():
    avg_embeddings[emotion] = np.mean(embs, axis=0)

# Compute similarity matrix
from sklearn.metrics.pairwise import cosine_similarity

emotion_names = list(EMOTIONS.keys())
sim_matrix = np.zeros((4, 4))

for i, em1 in enumerate(emotion_names):
    for j, em2 in enumerate(emotion_names):
        sim = cosine_similarity(
            avg_embeddings[em1].reshape(1, -1),
            avg_embeddings[em2].reshape(1, -1)
        )[0, 0]
        sim_matrix[i, j] = sim

# Print matrix
print("Cosine similarity matrix:")
print()
print(f"{'':12s}", end='')
for em in emotion_names:
    print(f"{em:12s}", end='')
print()

for i, em1 in enumerate(emotion_names):
    print(f"{em1:12s}", end='')
    for j, em2 in enumerate(emotion_names):
        print(f"{sim_matrix[i, j]:12.4f}", end='')
    print()
print()

# Within vs between
within_sims = []
between_sims = []

for emotion, embs in embeddings_by_emotion.items():
    # Within: all pairs from same emotion
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            sim = cosine_similarity(
                embs[i].reshape(1, -1),
                embs[j].reshape(1, -1)
            )[0, 0]
            within_sims.append(sim)

# Between: all pairs from different emotions
for i, (em1, embs1) in enumerate(embeddings_by_emotion.items()):
    for j, (em2, embs2) in enumerate(embeddings_by_emotion.items()):
        if i < j:  # Only upper triangle
            for e1 in embs1:
                for e2 in embs2:
                    sim = cosine_similarity(
                        e1.reshape(1, -1),
                        e2.reshape(1, -1)
                    )[0, 0]
                    between_sims.append(sim)

print(f"Within-emotion similarity:  {np.mean(within_sims):.4f} ± {np.std(within_sims):.4f}")
print(f"Between-emotion similarity: {np.mean(between_sims):.4f} ± {np.std(between_sims):.4f}")
print(f"Difference: {np.mean(within_sims) - np.mean(between_sims):.4f}")
print()

from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(within_sims, between_sims)
print(f"T-test: t = {t_stat:.4f}, p = {p_value:.6f}")
print()

if p_value < 0.05:
    print("✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print("⚠️  Not statistically significant")
print()

# ================================================================================
# Analysis 2: Linear Probe Test
# ================================================================================

print("=" * 80)
print("ANALYSIS 2: Linear Probe (Classification Test)")
print("=" * 80)
print()

print("Training logistic regression classifier...")
print("(5-fold cross-validation)")
print()

clf = LogisticRegression(max_iter=1000, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print(f"Accuracy scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print()

if np.mean(scores) > 0.65:
    print("✅ STRONG ENCODING (accuracy > 65%)")
elif np.mean(scores) > 0.50:
    print("⚠️  WEAK ENCODING (accuracy 50-65%)")
else:
    print("❌ NO ENCODING (accuracy < 50%)")
print()

# Train on full dataset to see feature importance
clf.fit(X, y)
print(f"Trained on full dataset")
print()

# ================================================================================
# Analysis 3: Dimensionality Analysis
# ================================================================================

print("=" * 80)
print("ANALYSIS 3: Which Dimensions Encode Emotions?")
print("=" * 80)
print()

# Find most important dimensions using variance between emotion means
emotion_means = np.array([avg_embeddings[em] for em in emotion_names])  # (4, 768)
between_var = np.var(emotion_means, axis=0)  # (768,)

# Find within-emotion variance
within_var = []
for dim in range(768):
    dim_vals = []
    for emotion, embs in embeddings_by_emotion.items():
        dim_vals.extend([emb[dim] for emb in embs])
    within_var.append(np.var(dim_vals))
within_var = np.array(within_var)

# Signal-to-noise ratio
snr = between_var / (within_var + 1e-10)

# Top 20 dimensions
top_dims = np.argsort(snr)[::-1][:20]

print("Top 20 emotion-encoding dimensions:")
print()
print(f"{'Rank':4s}  {'Dim':4s}  {'SNR':8s}  {'Between-var':12s}  {'Within-var':12s}")
print("-" * 60)
for rank, dim in enumerate(top_dims):
    print(f"{rank+1:4d}  {dim:4d}  {snr[dim]:8.4f}  {between_var[dim]:12.6f}  {within_var[dim]:12.6f}")
print()

# What % of dimensions are emotion-encoding?
significant_dims = np.sum(snr > 1.0)  # SNR > 1 means more between than within variance
print(f"Dimensions with SNR > 1.0: {significant_dims} / 768 ({100*significant_dims/768:.1f}%)")
print()

# ================================================================================
# Analysis 4: PCA Visualization
# ================================================================================

print("=" * 80)
print("ANALYSIS 4: PCA Visualization")
print("=" * 80)
print()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
print()

# Plot
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'orange']
for i, emotion in enumerate(emotion_names):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], label=emotion, alpha=0.6, s=100)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('T5 Embeddings: Emotion Clustering (PCA)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = OUTPUT_DIR / 'emotion_clustering_pca.png'
plt.savefig(plot_path, dpi=300)
print(f"✅ Plot saved to {plot_path}")
print()

# ================================================================================
# Save Results
# ================================================================================

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Save embeddings
np.save(OUTPUT_DIR / 'embeddings.npy', X)
np.save(OUTPUT_DIR / 'labels.npy', y)
print(f"✅ Saved embeddings to {OUTPUT_DIR}/embeddings.npy")
print(f"✅ Saved labels to {OUTPUT_DIR}/labels.npy")

# Save metadata
metadata = {
    'num_prompts': len(X),
    'embedding_dim': 768,
    'emotions': emotion_names,
    'prompts_per_emotion': {em: len(prompts) for em, prompts in EMOTIONS.items()},
    'similarity_analysis': {
        'within_emotion_mean': float(np.mean(within_sims)),
        'within_emotion_std': float(np.std(within_sims)),
        'between_emotion_mean': float(np.mean(between_sims)),
        'between_emotion_std': float(np.std(between_sims)),
        'difference': float(np.mean(within_sims) - np.mean(between_sims)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    },
    'linear_probe': {
        'mean_accuracy': float(np.mean(scores)),
        'std_accuracy': float(np.std(scores)),
        'cv_scores': [float(s) for s in scores],
        'verdict': 'strong' if np.mean(scores) > 0.65 else 'weak' if np.mean(scores) > 0.5 else 'none'
    },
    'dimensionality': {
        'top_20_dimensions': [int(d) for d in top_dims],
        'top_20_snr': [float(snr[d]) for d in top_dims],
        'significant_dimensions': int(significant_dims),
        'percent_significant': float(100*significant_dims/768)
    },
    'pca': {
        'variance_explained_pc1': float(pca.explained_variance_ratio_[0]),
        'variance_explained_pc2': float(pca.explained_variance_ratio_[1]),
        'total_variance_explained': float(sum(pca.explained_variance_ratio_))
    }
}

with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Saved metadata to {OUTPUT_DIR}/metadata.json")
print()

# ================================================================================
# Final Summary
# ================================================================================

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()

print("Dataset:")
print(f"  • 100 prompts (25 per emotion)")
print(f"  • 768-dimensional T5 embeddings")
print()

print("Key Results:")
print(f"  • Within-emotion similarity: {np.mean(within_sims):.4f}")
print(f"  • Between-emotion similarity: {np.mean(between_sims):.4f}")
print(f"  • Difference: {np.mean(within_sims) - np.mean(between_sims):.4f}")
print(f"  • Statistical significance: p = {p_value:.6f}")
print(f"  • Linear probe accuracy: {np.mean(scores):.4f}")
print(f"  • Emotion-encoding dims: {significant_dims}/768 ({100*significant_dims/768:.1f}%)")
print()

print("Verdict:")
if p_value < 0.05 and np.mean(scores) > 0.65:
    print("  ✅✅✅ STRONG EMOTION ENCODING CONFIRMED")
    print("  → Ready to proceed to Phase 1 (SAE training)")
    print(f"  → Target these {significant_dims} dimensions")
elif p_value < 0.05 or np.mean(scores) > 0.50:
    print("  ⚠️  WEAK BUT PRESENT ENCODING")
    print("  → May need larger dataset or different emotions")
else:
    print("  ❌ NO SIGNIFICANT ENCODING")
    print("  → Re-evaluate emotion categories")
print()

print("Next Steps:")
print("  1. Review PCA plot to visualize clustering")
print("  2. If strong: Proceed to SAE training on T5 embeddings")
print("  3. If weak: Generate more diverse prompts")
print("  4. Use top dimensions for targeted analysis")
print()

print("=" * 80)
