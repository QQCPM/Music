"""
Analyze SAE Features for Emotion Interpretability
==================================================

This script analyzes the learned features from the trained SAE to determine:
1. Which features are monosemantic (activate for one concept)
2. Which features encode specific emotions
3. Feature activation patterns across emotions

Analysis pipeline:
1. Load trained SAE model
2. Run all T5 embeddings through the SAE
3. Track which features activate for which emotions
4. Compute feature selectivity scores
5. Identify emotion-specific features
6. Visualize top features

Author: MusicGen Emotion Interpretability Research
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from typing import Dict, List, Tuple

from models.sparse_autoencoder import SparseAutoencoder
from utils.dataset_utils import load_t5_embeddings_with_metadata


# ================================================================================
# Configuration
# ================================================================================

CONFIG = {
    'model_path': 'results/sae_training/best_model.pt',  # Will be updated
    'data_dir': 'results/t5_embeddings',
    'output_dir': 'results/sae_analysis',
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',

    # Analysis params
    'activation_threshold': 0.01,  # Minimum activation to consider feature "on"
    'top_k_features': 20,          # Number of top features to analyze in detail
}


# ================================================================================
# Feature Analysis
# ================================================================================

class FeatureAnalyzer:
    """Analyzes learned SAE features for interpretability"""

    def __init__(
        self,
        model: SparseAutoencoder,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        emotion_names: List[str],
        activation_threshold: float = 0.01,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.embeddings = embeddings.to(device)
        self.labels = labels
        self.emotion_names = emotion_names
        self.activation_threshold = activation_threshold
        self.device = device
        self.num_features = model.config.hidden_dim
        self.num_emotions = len(emotion_names)

        # Computed attributes
        self.feature_activations = None
        self.feature_emotion_stats = None

    @torch.no_grad()
    def compute_feature_activations(self):
        """Compute feature activations for all embeddings"""
        print("Computing feature activations...")

        all_activations = []

        # Process in batches to avoid OOM
        batch_size = 32
        for i in range(0, len(self.embeddings), batch_size):
            batch = self.embeddings[i:i+batch_size]
            output = self.model(batch, return_loss=False, return_aux=False)
            all_activations.append(output['hidden'].cpu())

        self.feature_activations = torch.cat(all_activations, dim=0)
        print(f"✅ Computed activations: {self.feature_activations.shape}")

    def compute_feature_emotion_statistics(self):
        """Compute statistics about which features activate for which emotions"""
        print("Computing feature-emotion statistics...")

        if self.feature_activations is None:
            self.compute_feature_activations()

        stats = {}

        for feature_idx in range(self.num_features):
            feature_acts = self.feature_activations[:, feature_idx]

            # Identify when feature is active
            is_active = feature_acts > self.activation_threshold

            # Statistics per emotion
            emotion_stats = {}
            for emotion_idx, emotion_name in enumerate(self.emotion_names):
                mask = self.labels == emotion_idx

                # Activation rate for this emotion
                activation_rate = is_active[mask].float().mean().item()

                # Mean activation when active
                active_values = feature_acts[mask & is_active]
                mean_activation = active_values.mean().item() if len(active_values) > 0 else 0.0

                emotion_stats[emotion_name] = {
                    'activation_rate': activation_rate,
                    'mean_activation': mean_activation,
                    'num_samples': mask.sum().item(),
                    'num_active': is_active[mask].sum().item()
                }

            # Overall statistics
            stats[feature_idx] = {
                'emotions': emotion_stats,
                'overall_activation_rate': is_active.float().mean().item(),
                'overall_mean_activation': feature_acts[is_active].mean().item() if is_active.any() else 0.0,
                'max_activation': feature_acts.max().item(),
            }

        self.feature_emotion_stats = stats
        print(f"✅ Computed statistics for {self.num_features} features")

    def compute_feature_selectivity(self, feature_idx: int) -> float:
        """
        Compute selectivity score for a feature.

        Selectivity = max_emotion(activation_rate) / mean_all_emotions(activation_rate)
        High selectivity (> 2) means feature is selective for specific emotion.

        Args:
            feature_idx: Index of feature to analyze

        Returns:
            selectivity: Selectivity score (higher = more selective)
        """
        if self.feature_emotion_stats is None:
            self.compute_feature_emotion_statistics()

        stats = self.feature_emotion_stats[feature_idx]['emotions']
        rates = [s['activation_rate'] for s in stats.values()]

        if max(rates) == 0:
            return 0.0

        selectivity = max(rates) / (np.mean(rates) + 1e-8)
        return selectivity

    def find_emotion_specific_features(self, min_selectivity: float = 2.0) -> Dict[str, List[int]]:
        """
        Find features that are selective for each emotion.

        Args:
            min_selectivity: Minimum selectivity score

        Returns:
            emotion_features: Dict mapping emotion names to list of feature indices
        """
        print(f"Finding emotion-specific features (min selectivity: {min_selectivity})...")

        if self.feature_emotion_stats is None:
            self.compute_feature_emotion_statistics()

        emotion_features = {emotion: [] for emotion in self.emotion_names}

        for feature_idx in range(self.num_features):
            selectivity = self.compute_feature_selectivity(feature_idx)

            if selectivity >= min_selectivity:
                # Find which emotion has highest activation rate
                stats = self.feature_emotion_stats[feature_idx]['emotions']
                best_emotion = max(stats.items(), key=lambda x: x[1]['activation_rate'])[0]

                emotion_features[best_emotion].append((feature_idx, selectivity))

        # Sort by selectivity
        for emotion in self.emotion_names:
            emotion_features[emotion] = sorted(
                emotion_features[emotion],
                key=lambda x: x[1],
                reverse=True
            )

        # Print summary
        for emotion in self.emotion_names:
            n_features = len(emotion_features[emotion])
            print(f"  {emotion:12s}: {n_features} selective features")

        return emotion_features

    def get_top_features_by_activation(self, k: int = 20) -> List[Tuple[int, float]]:
        """Get top k features by overall activation frequency"""
        if self.feature_activations is None:
            self.compute_feature_activations()

        activation_rates = (self.feature_activations > self.activation_threshold).float().mean(dim=0)
        top_indices = torch.topk(activation_rates, k).indices.tolist()
        top_rates = [activation_rates[i].item() for i in top_indices]

        return list(zip(top_indices, top_rates))

    def visualize_feature_emotion_matrix(self, top_k: int = 50, output_path: Path = None):
        """
        Visualize feature activation rates across emotions as a heatmap.

        Args:
            top_k: Number of top features to show
            output_path: Where to save plot
        """
        if self.feature_emotion_stats is None:
            self.compute_feature_emotion_statistics()

        # Get top k features by selectivity
        selectivities = [(i, self.compute_feature_selectivity(i)) for i in range(self.num_features)]
        selectivities = sorted(selectivities, key=lambda x: x[1], reverse=True)
        top_features = [i for i, _ in selectivities[:top_k]]

        # Build matrix
        matrix = np.zeros((top_k, self.num_emotions))
        for row_idx, feature_idx in enumerate(top_features):
            stats = self.feature_emotion_stats[feature_idx]['emotions']
            for col_idx, emotion in enumerate(self.emotion_names):
                matrix[row_idx, col_idx] = stats[emotion]['activation_rate']

        # Plot
        plt.figure(figsize=(10, max(12, top_k // 3)))
        sns.heatmap(
            matrix,
            xticklabels=self.emotion_names,
            yticklabels=[f"F{i}" for i in top_features],
            cmap='YlOrRd',
            cbar_kws={'label': 'Activation Rate'},
            vmin=0,
            vmax=1
        )
        plt.title(f'Top {top_k} Features by Selectivity\nActivation Rate per Emotion')
        plt.xlabel('Emotion')
        plt.ylabel('Feature Index')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved heatmap to {output_path}")

        plt.close()


def analyze_sae(config: dict):
    """Main analysis function"""

    print("=" * 80)
    print("SAE FEATURE ANALYSIS")
    print("=" * 80)
    print()

    # Find latest model if specific path not provided
    model_path = Path(config['model_path'])
    if not model_path.exists():
        # Look for most recent experiment
        training_dir = Path('results/sae_training')
        if training_dir.exists():
            experiments = sorted([d for d in training_dir.iterdir() if d.is_dir()])
            if experiments:
                latest_exp = experiments[-1]
                model_path = latest_exp / 'best_model.pt'
                print(f"Using latest experiment: {latest_exp.name}")

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please train a model first: python experiments/train_sae_on_t5_embeddings.py")
        return

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = SparseAutoencoder.load(str(model_path), device=config['device'])
    print(f"✅ Model loaded")
    print(f"   Input dim: {model.config.input_dim}")
    print(f"   Hidden dim: {model.config.hidden_dim}")
    print(f"   L1 coefficient: {model.config.l1_coefficient}")
    print()

    # Load data
    print("Loading T5 embeddings...")
    embeddings, labels, metadata = load_t5_embeddings_with_metadata(config['data_dir'])
    emotion_names = metadata['emotions']
    print(f"✅ Loaded {len(embeddings)} embeddings")
    print(f"   Emotions: {emotion_names}")
    print()

    # Create analyzer
    print("Creating feature analyzer...")
    analyzer = FeatureAnalyzer(
        model=model,
        embeddings=embeddings,
        labels=labels,
        emotion_names=emotion_names,
        activation_threshold=config['activation_threshold'],
        device=config['device']
    )
    print()

    # Compute activations
    analyzer.compute_feature_activations()
    print()

    # Compute statistics
    analyzer.compute_feature_emotion_statistics()
    print()

    # Find emotion-specific features
    print("=" * 80)
    print("EMOTION-SPECIFIC FEATURES")
    print("=" * 80)
    print()

    emotion_features = analyzer.find_emotion_specific_features(min_selectivity=2.0)
    print()

    # Print detailed analysis for each emotion
    for emotion in emotion_names:
        features = emotion_features[emotion]
        if len(features) > 0:
            print(f"\n{emotion.upper()} Features (Top 5):")
            print(f"{'Feature':<10} {'Selectivity':<12} {'Activation Rate':<18} {'Mean Activation':<18}")
            print("-" * 60)

            for feature_idx, selectivity in features[:5]:
                stats = analyzer.feature_emotion_stats[feature_idx]['emotions'][emotion]
                print(f"F{feature_idx:<9} {selectivity:<12.2f} {stats['activation_rate']:<18.2%} {stats['mean_activation']:<18.4f}")

    # Overall statistics
    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print()

    total_selective_features = sum(len(features) for features in emotion_features.values())
    print(f"Total selective features: {total_selective_features} / {model.config.hidden_dim}")
    print(f"Percentage selective: {100 * total_selective_features / model.config.hidden_dim:.1f}%")
    print()

    # Top features by activation
    print("Top 10 features by activation frequency:")
    top_features = analyzer.get_top_features_by_activation(k=10)
    print(f"{'Feature':<10} {'Activation Rate':<18}")
    print("-" * 30)
    for feature_idx, rate in top_features:
        print(f"F{feature_idx:<9} {rate:<18.2%}")
    print()

    # Visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    analyzer.visualize_feature_emotion_matrix(
        top_k=config['top_k_features'],
        output_path=output_dir / 'feature_emotion_heatmap.png'
    )

    # Save detailed results
    print()
    print("Saving detailed results...")

    results = {
        'emotion_specific_features': {
            emotion: [(int(idx), float(sel)) for idx, sel in features]
            for emotion, features in emotion_features.items()
        },
        'total_selective_features': total_selective_features,
        'percentage_selective': float(100 * total_selective_features / model.config.hidden_dim),
        'top_features_by_activation': [
            {'feature_idx': int(idx), 'activation_rate': float(rate)}
            for idx, rate in top_features
        ]
    }

    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output_dir}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Analysis complete!")
    print(f"Found {total_selective_features} emotion-selective features")
    print(f"Results saved to: {output_dir}")
    print()
    print("Key findings:")
    for emotion in emotion_names:
        n = len(emotion_features[emotion])
        if n > 0:
            top_selectivity = emotion_features[emotion][0][1]
            print(f"  • {emotion}: {n} features (best selectivity: {top_selectivity:.2f}x)")
    print()
    print("Next steps:")
    print("  1. Review feature_emotion_heatmap.png to see activation patterns")
    print("  2. Investigate specific features in detail")
    print("  3. Use features for activation steering experiments")
    print()
    print("=" * 80)


# ================================================================================
# Main
# ================================================================================

if __name__ == '__main__':
    analyze_sae(CONFIG)
