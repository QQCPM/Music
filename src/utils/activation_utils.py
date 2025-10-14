"""
Utilities for extracting and analyzing activations from MusicGen.
"""

import torch
from typing import Dict, List, Optional, Callable
from pathlib import Path
import numpy as np


class ActivationExtractor:
    """
    Extract activations from specific layers of MusicGen during generation.

    Usage:
        extractor = ActivationExtractor(model, layers=[0, 6, 12, 18, 24])
        wav = extractor.generate(["happy music"])
        activations = extractor.get_activations()
    """

    def __init__(
        self,
        model,
        layers: Optional[List[int]] = None,
        store_on_cpu: bool = True
    ):
        """
        Args:
            model: MusicGen model instance
            layers: List of layer indices to extract from (default: all layers)
            store_on_cpu: Move activations to CPU to save GPU memory
        """
        self.model = model
        self.store_on_cpu = store_on_cpu
        self.activations = {}
        self.hooks = []

        # Determine which layers to hook
        if layers is None:
            # Extract from all layers
            num_layers = len(model.lm.transformer.layers)
            self.layers = list(range(num_layers))
        else:
            self.layers = layers

        # Register hooks
        self._register_hooks()

    def _make_hook(self, name: str) -> Callable:
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # Detach and optionally move to CPU
            activation = output.detach()
            if self.store_on_cpu:
                activation = activation.cpu()

            # FIXED: Append to list instead of overwriting
            # MusicGen generates autoregressively (459 timesteps for 3s audio)
            # We need to capture ALL timesteps, not just the last one
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(activation)
        return hook

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        for layer_idx in self.layers:
            layer = self.model.lm.transformer.layers[layer_idx]
            hook = layer.register_forward_hook(
                self._make_hook(f'layer_{layer_idx}')
            )
            self.hooks.append(hook)

    def generate(self, descriptions: List[str], **kwargs):
        """
        Generate music with activation capture.

        Args:
            descriptions: Text prompts for generation
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Generated audio waveforms
        """
        # Clear previous activations
        self.activations = {}

        # Generate (hooks will capture activations)
        with torch.no_grad():
            wav = self.model.generate(descriptions, **kwargs)

        return wav

    def get_activations(self, concatenate: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get captured activations.

        Args:
            concatenate: If True, concatenate list of timesteps into single tensor.
                        If False, return raw list of activations.

        Returns:
            Dictionary mapping layer names to activation tensors or lists.
            If concatenate=True, tensors have shape [num_timesteps, num_codebooks, batch, d_model]
        """
        if not concatenate:
            return self.activations

        # Concatenate activation lists into tensors
        concatenated = {}
        for name, act_list in self.activations.items():
            if isinstance(act_list, list) and len(act_list) > 0:
                # Stack along timestep dimension
                # Each element has shape [num_codebooks, batch, d_model]
                # Result: [num_timesteps, num_codebooks, batch, d_model]
                concatenated[name] = torch.stack(act_list, dim=0)
            else:
                concatenated[name] = act_list

        return concatenated

    def get_layer_activation(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get activation for a specific layer."""
        key = f'layer_{layer_idx}'
        return self.activations.get(key)

    def save_activations(self, filepath: str):
        """Save activations to disk."""
        torch.save(self.activations, filepath)
        print(f"Saved activations to {filepath}")

    def clear_activations(self):
        """Clear stored activations to free memory."""
        self.activations = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class ActivationDataset:
    """
    Manage a collection of activations with metadata.

    Useful for storing activations from multiple generations
    along with labels, prompts, and other metadata.
    """

    def __init__(self):
        self.activations = []
        self.metadata = []

    def add(
        self,
        activations: Dict[str, torch.Tensor],
        prompt: str,
        label: Optional[str] = None,
        **extra_metadata
    ):
        """
        Add a set of activations with metadata.

        Args:
            activations: Dictionary of layer activations
            prompt: Text prompt used for generation
            label: Optional label (e.g., emotion category)
            **extra_metadata: Any additional metadata
        """
        self.activations.append(activations)
        metadata = {
            'prompt': prompt,
            'label': label,
            **extra_metadata
        }
        self.metadata.append(metadata)

    def save(self, filepath: str):
        """Save dataset to disk."""
        data = {
            'activations': self.activations,
            'metadata': self.metadata
        }
        torch.save(data, filepath)
        print(f"Saved activation dataset to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk."""
        data = torch.load(filepath)
        dataset = cls()
        dataset.activations = data['activations']
        dataset.metadata = data['metadata']
        return dataset

    def get_by_label(self, label: str):
        """Get all activations with a specific label."""
        indices = [
            i for i, meta in enumerate(self.metadata)
            if meta.get('label') == label
        ]
        return [self.activations[i] for i in indices]

    def get_layer_activations(
        self,
        layer_idx: int,
        as_array: bool = True
    ) -> np.ndarray:
        """
        Get all activations for a specific layer.

        Args:
            layer_idx: Layer index
            as_array: If True, stack into numpy array

        Returns:
            Array or list of activations
        """
        key = f'layer_{layer_idx}'
        layer_acts = [
            act[key] for act in self.activations
            if key in act
        ]

        if as_array and layer_acts:
            # Convert to numpy and stack
            layer_acts = [
                act.cpu().numpy() if torch.is_tensor(act) else act
                for act in layer_acts
            ]
            # Stack along first dimension
            return np.vstack(layer_acts)

        return layer_acts

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.metadata[idx]


def compute_steering_vector(
    activations_positive: torch.Tensor,
    activations_negative: torch.Tensor,
    method: str = 'mean_diff'
) -> torch.Tensor:
    """
    Compute a steering vector from contrastive examples.

    Args:
        activations_positive: Activations from "positive" examples
            Shape: [n_positive, seq_len, d_model] or [n_positive, d_model]
        activations_negative: Activations from "negative" examples
            Shape: [n_negative, seq_len, d_model] or [n_negative, d_model]
        method: Method for computing vector
            - 'mean_diff': mean(positive) - mean(negative)
            - 'pca': First principal component of difference

    Returns:
        Steering vector of shape [d_model] or [seq_len, d_model]
    """
    if method == 'mean_diff':
        # Average over batch dimension
        pos_mean = activations_positive.mean(dim=0)
        neg_mean = activations_negative.mean(dim=0)
        steering_vector = pos_mean - neg_mean

    elif method == 'pca':
        # Compute first PC of the difference
        from sklearn.decomposition import PCA

        # Flatten if needed
        if activations_positive.ndim == 3:
            # Average over sequence
            pos_mean = activations_positive.mean(dim=1)
            neg_mean = activations_negative.mean(dim=1)
        else:
            pos_mean = activations_positive
            neg_mean = activations_negative

        # Compute differences
        diffs = pos_mean - neg_mean

        # PCA
        pca = PCA(n_components=1)
        pca.fit(diffs.cpu().numpy())
        steering_vector = torch.from_numpy(
            pca.components_[0]
        ).to(activations_positive.device)

    else:
        raise ValueError(f"Unknown method: {method}")

    return steering_vector


def apply_steering_vector(
    activation: torch.Tensor,
    steering_vector: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Apply a steering vector to an activation.

    Args:
        activation: Current activation tensor
        steering_vector: Vector to add
        alpha: Scaling factor for the steering vector

    Returns:
        Modified activation
    """
    return activation + alpha * steering_vector


def analyze_activation_statistics(
    activations: torch.Tensor,
    return_dict: bool = True
):
    """
    Compute statistics about activations.

    Args:
        activations: Tensor of activations
        return_dict: If True, return as dictionary

    Returns:
        Dictionary or tuple of statistics
    """
    stats = {
        'mean': activations.mean().item(),
        'std': activations.std().item(),
        'min': activations.min().item(),
        'max': activations.max().item(),
        'sparsity': (activations == 0).float().mean().item(),
        'l1_norm': activations.abs().mean().item(),
        'l2_norm': (activations ** 2).mean().sqrt().item(),
    }

    if return_dict:
        return stats
    else:
        return tuple(stats.values())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two tensors.

    Args:
        a, b: Tensors of same shape

    Returns:
        Cosine similarity as float
    """
    a_flat = a.flatten()
    b_flat = b.flatten()

    similarity = torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0),
        b_flat.unsqueeze(0)
    )

    return similarity.item()
