"""
Visualization utilities for activation analysis and interpretability.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional
import umap


def plot_activation_statistics(
    activations: Dict[str, torch.Tensor],
    save_path: Optional[str] = None
):
    """
    Plot statistics (mean, std, sparsity) across layers.

    Args:
        activations: Dictionary mapping layer names to activation tensors
        save_path: Optional path to save figure
    """
    layer_names = sorted(activations.keys())
    stats = {
        'mean': [],
        'std': [],
        'sparsity': []
    }

    for name in layer_names:
        act = activations[name]
        stats['mean'].append(act.mean().item())
        stats['std'].append(act.std().item())
        stats['sparsity'].append((act == 0).float().mean().item())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean activation
    axes[0].plot(range(len(layer_names)), stats['mean'], marker='o')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Activation')
    axes[0].set_title('Average Activation by Layer')
    axes[0].grid(True, alpha=0.3)

    # Std activation
    axes[1].plot(range(len(layer_names)), stats['std'], marker='o', color='orange')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Std Activation')
    axes[1].set_title('Activation Variance by Layer')
    axes[1].grid(True, alpha=0.3)

    # Sparsity
    axes[2].plot(range(len(layer_names)), stats['sparsity'], marker='o', color='green')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Sparsity (% zeros)')
    axes[2].set_title('Activation Sparsity by Layer')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_umap_projection(
    activations: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "UMAP Projection of Activations",
    save_path: Optional[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1
):
    """
    Plot UMAP projection of high-dimensional activations.

    Args:
        activations: Array of shape [n_samples, d_model]
        labels: Optional list of labels for coloring
        title: Plot title
        save_path: Optional path to save figure
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    # Ensure 2D
    if activations.ndim > 2:
        # Flatten extra dimensions
        n_samples = activations.shape[0]
        activations = activations.reshape(n_samples, -1)

    # UMAP projection
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    embedding = reducer.fit_transform(activations)

    # Plot
    plt.figure(figsize=(10, 8))

    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.6,
                s=50
            )
        plt.legend()
    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=0.6,
            s=50
        )

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf(), embedding


def plot_activation_heatmap(
    activation: torch.Tensor,
    title: str = "Activation Heatmap",
    save_path: Optional[str] = None,
    max_tokens: int = 100,
    max_features: int = 100
):
    """
    Plot activation heatmap.

    Args:
        activation: Activation tensor [seq_len, d_model] or [batch, seq_len, d_model]
        title: Plot title
        save_path: Optional path to save figure
        max_tokens: Maximum tokens to show
        max_features: Maximum features to show
    """
    # Convert to numpy
    if torch.is_tensor(activation):
        activation = activation.cpu().numpy()

    # Handle batch dimension
    if activation.ndim == 3:
        activation = activation[0]  # Take first in batch

    # Limit size for visualization
    seq_len, d_model = activation.shape
    activation = activation[:min(seq_len, max_tokens), :min(d_model, max_features)]

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        activation.T,
        cmap='viridis',
        cbar_kws={'label': 'Activation'},
        xticklabels=False,
        yticklabels=False
    )
    plt.xlabel('Token Position')
    plt.ylabel('Feature Dimension')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_steering_effect(
    original_features: Dict[str, float],
    steered_features: Dict[str, float],
    feature_names: Optional[List[str]] = None,
    title: str = "Effect of Activation Steering",
    save_path: Optional[str] = None
):
    """
    Compare features before and after steering.

    Args:
        original_features: Features from original generation
        steered_features: Features from steered generation
        feature_names: Optional list of feature names to plot
        title: Plot title
        save_path: Optional path to save figure
    """
    if feature_names is None:
        feature_names = ['tempo', 'spectral_centroid_mean', 'rms_mean', 'chroma_mean']

    # Extract values
    original_vals = [original_features.get(name, 0) for name in feature_names]
    steered_vals = [steered_features.get(name, 0) for name in feature_names]

    # Normalize for comparison
    max_vals = [max(o, s) for o, s in zip(original_vals, steered_vals)]
    original_norm = [o / m if m > 0 else 0 for o, m in zip(original_vals, max_vals)]
    steered_norm = [s / m if m > 0 else 0 for s, m in zip(steered_vals, max_vals)]

    # Plot
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, original_norm, width, label='Original', alpha=0.8)
    ax.bar(x + width/2, steered_norm, width, label='Steered', alpha=0.8)

    ax.set_xlabel('Feature')
    ax.set_ylabel('Normalized Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_layer_wise_similarity(
    activations1: Dict[str, torch.Tensor],
    activations2: Dict[str, torch.Tensor],
    metric: str = 'cosine',
    title: str = "Layer-wise Similarity",
    save_path: Optional[str] = None
):
    """
    Plot similarity between two sets of activations across layers.

    Args:
        activations1, activations2: Dictionaries of activations
        metric: Similarity metric ('cosine' or 'euclidean')
        title: Plot title
        save_path: Optional path to save figure
    """
    layer_names = sorted(activations1.keys())
    similarities = []

    for name in layer_names:
        act1 = activations1[name].flatten()
        act2 = activations2[name].flatten()

        if metric == 'cosine':
            sim = torch.nn.functional.cosine_similarity(
                act1.unsqueeze(0),
                act2.unsqueeze(0)
            ).item()
        elif metric == 'euclidean':
            dist = torch.nn.functional.pairwise_distance(
                act1.unsqueeze(0),
                act2.unsqueeze(0)
            ).item()
            # Convert to similarity
            sim = 1 / (1 + dist)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        similarities.append(sim)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(layer_names)), similarities, marker='o', linewidth=2)
    plt.xlabel('Layer')
    plt.ylabel(f'{metric.capitalize()} Similarity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_emotion_clusters(
    activations: np.ndarray,
    emotions: List[str],
    layer_name: str = "",
    save_path: Optional[str] = None
):
    """
    Plot UMAP projection colored by emotion labels.

    Args:
        activations: Activation array [n_samples, d_model]
        emotions: List of emotion labels
        layer_name: Name of layer for title
        save_path: Optional path to save figure
    """
    title = f"Emotion Clustering in {layer_name}" if layer_name else "Emotion Clustering"

    fig, embedding = plot_umap_projection(
        activations=activations,
        labels=emotions,
        title=title,
        save_path=save_path
    )

    # Compute cluster separation metrics
    from sklearn.metrics import silhouette_score

    # Encode emotions as numbers
    unique_emotions = sorted(set(emotions))
    emotion_to_idx = {e: i for i, e in enumerate(unique_emotions)}
    emotion_indices = [emotion_to_idx[e] for e in emotions]

    silhouette = silhouette_score(embedding, emotion_indices)

    # Add silhouette score to plot
    plt.text(
        0.02, 0.98,
        f'Silhouette Score: {silhouette:.3f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, silhouette
