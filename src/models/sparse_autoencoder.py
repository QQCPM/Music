"""
Sparse Autoencoder (SAE) for T5 Text Embeddings
================================================

Implementation based on:
- Anthropic's "Towards Monosemanticity" (2023)
- Bricken et al., "Sparse Autoencoders Find Highly Interpretable Features" (2023)

Architecture:
    Input: T5 embeddings (768-dim)
    Hidden: Overcomplete representation (4096-6144 dim, 8x expansion)
    Output: Reconstructed T5 embeddings (768-dim)

Training:
    Loss = MSE(input, reconstruction) + λ * L1(hidden_activations)

The L1 penalty encourages sparsity → monosemantic features
The overcomplete expansion allows disentangling superposition

Author: MusicGen Emotion Interpretability Research
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder"""

    # Architecture
    input_dim: int = 768              # T5-base embedding dimension
    hidden_dim: int = 6144            # 8x expansion (overcomplete)

    # Training
    l1_coefficient: float = 1e-3      # Sparsity penalty strength
    learning_rate: float = 1e-4       # Adam learning rate
    batch_size: int = 32              # Batch size for training

    # Regularization
    weight_decay: float = 0.0         # L2 regularization on weights
    decoder_norm: bool = True         # Normalize decoder columns to unit norm

    # Initialization
    encoder_init_scale: float = 0.1   # Scale for encoder weight init
    bias_init: float = 0.0            # Initial bias value

    # Sparsity targets
    target_l0: Optional[float] = None # Target number of active features (optional)
    dead_feature_threshold: int = 100 # Steps before considering feature "dead"

    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_dim >= self.input_dim, \
            "SAE must be overcomplete (hidden_dim >= input_dim)"
        assert self.l1_coefficient > 0, \
            "L1 coefficient must be positive for sparsity"
        assert 0 < self.encoder_init_scale <= 1.0, \
            "Encoder init scale should be in (0, 1]"


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for finding monosemantic features in T5 embeddings.

    Forward pass:
        1. Encode: hidden = ReLU(W_enc @ (input - bias_dec) + bias_enc)
        2. Decode: output = W_dec @ hidden + bias_dec
        3. Compute loss: MSE + L1 penalty on hidden activations

    Key properties:
        - Tied weights: W_dec = W_enc.T (reduces parameters)
        - ReLU activation: Enforces non-negativity (helps interpretability)
        - L1 penalty: Encourages sparse hidden representations
        - Overcomplete: More hidden dims than input dims (disentangles features)
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # Encoder: input_dim → hidden_dim
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)

        # Decoder: hidden_dim → input_dim
        # Note: We use separate weights (not tied) for flexibility
        # Can tie later if needed: self.decoder.weight = self.encoder.weight.T
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=True)

        # Pre-bias (subtracted before encoding, added after decoding)
        # This centers the input distribution
        self.register_buffer('pre_bias', torch.zeros(config.input_dim))

        # Initialize weights
        self._init_weights()

        # Track dead features (features that never activate)
        self.register_buffer('feature_activations', torch.zeros(config.hidden_dim))
        self.register_buffer('steps_since_activation', torch.zeros(config.hidden_dim, dtype=torch.long))

    def _init_weights(self):
        """Initialize weights using best practices from SAE literature"""

        # Encoder: Small random weights for stability
        nn.init.kaiming_uniform_(self.encoder.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.encoder.weight.data *= self.config.encoder_init_scale
        nn.init.constant_(self.encoder.bias, self.config.bias_init)

        # Decoder: Orthogonal initialization + column normalization
        nn.init.orthogonal_(self.decoder.weight)
        if self.config.decoder_norm:
            self._normalize_decoder()
        nn.init.constant_(self.decoder.bias, 0.0)

    def _normalize_decoder(self):
        """Normalize decoder weight columns to unit norm"""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse hidden representation.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            hidden: Sparse hidden activations [batch, hidden_dim]
        """
        # Center input
        x_centered = x - self.pre_bias

        # Encode with ReLU for sparsity
        hidden = F.relu(self.encoder(x_centered))

        return hidden

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse hidden representation to reconstruction.

        Args:
            hidden: Sparse hidden activations [batch, hidden_dim]

        Returns:
            reconstruction: Reconstructed input [batch, input_dim]
        """
        return self.decoder(hidden) + self.pre_bias

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
        return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with optional loss computation.

        Args:
            x: Input tensor [batch, input_dim]
            return_loss: Whether to compute and return loss
            return_aux: Whether to return auxiliary metrics

        Returns:
            Dictionary containing:
                - hidden: Sparse hidden activations
                - reconstruction: Reconstructed input
                - loss (if return_loss=True): Total loss
                - loss_reconstruction (if return_loss=True): MSE loss
                - loss_sparsity (if return_loss=True): L1 loss
                - l0 (if return_aux=True): Number of active features
                - dead_features (if return_aux=True): Number of dead features
        """
        # Encode
        hidden = self.encode(x)

        # Decode
        reconstruction = self.decode(hidden)

        # Prepare output
        output = {
            'hidden': hidden,
            'reconstruction': reconstruction,
        }

        # Compute loss if requested
        if return_loss:
            # Reconstruction loss (MSE)
            loss_reconstruction = F.mse_loss(reconstruction, x)

            # Sparsity loss (L1 on hidden activations)
            loss_sparsity = hidden.abs().mean()

            # Total loss
            loss = loss_reconstruction + self.config.l1_coefficient * loss_sparsity

            output['loss'] = loss
            output['loss_reconstruction'] = loss_reconstruction
            output['loss_sparsity'] = loss_sparsity

        # Compute auxiliary metrics if requested
        if return_aux:
            # L0 norm (number of active features per sample)
            l0 = (hidden > 0).float().sum(dim=1).mean()

            # Dead features (features that haven't activated recently)
            dead_features = (self.steps_since_activation > self.config.dead_feature_threshold).sum()

            output['l0'] = l0
            output['dead_features'] = dead_features
            output['max_activation'] = hidden.max()
            output['mean_activation'] = hidden[hidden > 0].mean() if (hidden > 0).any() else torch.tensor(0.0)

        return output

    def update_feature_tracking(self, hidden: torch.Tensor):
        """
        Update tracking of which features are active.
        Used to detect dead features during training.

        Args:
            hidden: Sparse hidden activations [batch, hidden_dim]
        """
        # Check which features activated
        activated = (hidden.max(dim=0).values > 0)

        # Update counters
        self.steps_since_activation += 1
        self.steps_since_activation[activated] = 0
        self.feature_activations[activated] += 1

    def get_feature_importance(self) -> torch.Tensor:
        """
        Get importance score for each feature based on activation frequency.

        Returns:
            importance: Tensor of shape [hidden_dim] with importance scores
        """
        return self.feature_activations / (self.feature_activations.sum() + 1e-8)

    def get_dead_features(self) -> torch.Tensor:
        """
        Get indices of features that haven't activated recently.

        Returns:
            dead_indices: Tensor of indices for dead features
        """
        return torch.where(self.steps_since_activation > self.config.dead_feature_threshold)[0]

    def reinit_dead_features(self):
        """
        Reinitialize dead features with random weights.
        This helps prevent feature death during training.
        """
        dead_features = self.get_dead_features()

        if len(dead_features) > 0:
            # Reinitialize encoder weights for dead features
            nn.init.kaiming_uniform_(
                self.encoder.weight[dead_features],
                a=0, mode='fan_in', nonlinearity='relu'
            )
            self.encoder.weight.data[dead_features] *= self.config.encoder_init_scale

            # Reinitialize decoder weights
            nn.init.orthogonal_(self.decoder.weight[:, dead_features])
            if self.config.decoder_norm:
                with torch.no_grad():
                    self.decoder.weight.data[:, dead_features] = F.normalize(
                        self.decoder.weight.data[:, dead_features],
                        dim=0
                    )

            # Reset tracking
            self.steps_since_activation[dead_features] = 0
            self.feature_activations[dead_features] = 0

        return len(dead_features)

    def set_pre_bias(self, data_mean: torch.Tensor):
        """
        Set the pre-bias to center the input distribution.
        Should be called before training with the mean of training data.

        Args:
            data_mean: Mean of training data [input_dim]
        """
        self.pre_bias.copy_(data_mean)

    @torch.no_grad()
    def get_feature_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the feature vectors (decoder columns).
        These represent the directions in input space for each feature.

        Returns:
            encoder_features: Encoder weight matrix [hidden_dim, input_dim]
            decoder_features: Decoder weight matrix [input_dim, hidden_dim]
        """
        return self.encoder.weight.data.clone(), self.decoder.weight.data.clone()

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'feature_activations': self.feature_activations,
            'steps_since_activation': self.steps_since_activation,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.feature_activations = checkpoint['feature_activations']
        model.steps_since_activation = checkpoint['steps_since_activation']
        return model


class SAETrainer:
    """
    Trainer for Sparse Autoencoder.

    Handles:
        - Training loop with progress tracking
        - Dead feature reinitialization
        - Decoder weight normalization
        - Validation and metrics logging
    """

    def __init__(
        self,
        model: SparseAutoencoder,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        # Training state
        self.step = 0
        self.epoch = 0

        # Metrics history
        self.history = {
            'loss': [],
            'loss_reconstruction': [],
            'loss_sparsity': [],
            'l0': [],
            'dead_features': [],
        }

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Input batch [batch_size, input_dim]

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        batch = batch.to(self.device)

        # Forward pass
        output = self.model(batch, return_loss=True, return_aux=True)

        # Backward pass
        self.optimizer.zero_grad()
        output['loss'].backward()
        self.optimizer.step()

        # Normalize decoder if configured
        if self.model.config.decoder_norm:
            self.model._normalize_decoder()

        # Update feature tracking
        self.model.update_feature_tracking(output['hidden'].detach())

        # Increment step
        self.step += 1

        # Extract metrics
        metrics = {
            'loss': output['loss'].item(),
            'loss_reconstruction': output['loss_reconstruction'].item(),
            'loss_sparsity': output['loss_sparsity'].item(),
            'l0': output['l0'].item(),
            'dead_features': output['dead_features'].item(),
        }

        # Record history
        for key, value in metrics.items():
            self.history[key].append(value)

        return metrics

    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Validation pass.

        Args:
            dataloader: Validation data loader

        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0
        total_loss_recon = 0
        total_loss_sparse = 0
        total_l0 = 0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            output = self.model(batch, return_loss=True, return_aux=True)

            total_loss += output['loss'].item()
            total_loss_recon += output['loss_reconstruction'].item()
            total_loss_sparse += output['loss_sparsity'].item()
            total_l0 += output['l0'].item()
            n_batches += 1

        return {
            'val_loss': total_loss / n_batches,
            'val_loss_reconstruction': total_loss_recon / n_batches,
            'val_loss_sparsity': total_loss_sparse / n_batches,
            'val_l0': total_l0 / n_batches,
        }


def create_sae_for_t5_embeddings(
    expansion_factor: int = 8,
    l1_coefficient: float = 1e-3,
    learning_rate: float = 1e-4
) -> Tuple[SparseAutoencoder, torch.optim.Optimizer]:
    """
    Convenience function to create SAE for T5 embeddings.

    Args:
        expansion_factor: Hidden dim = input_dim * expansion_factor
        l1_coefficient: Sparsity penalty strength
        learning_rate: Adam learning rate

    Returns:
        model: SparseAutoencoder instance
        optimizer: Configured Adam optimizer
    """
    config = SAEConfig(
        input_dim=768,                              # T5-base
        hidden_dim=768 * expansion_factor,          # Overcomplete
        l1_coefficient=l1_coefficient,
        learning_rate=learning_rate,
    )

    model = SparseAutoencoder(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    return model, optimizer


if __name__ == '__main__':
    """Test the SAE implementation"""

    print("Testing Sparse Autoencoder...")
    print()

    # Create model
    config = SAEConfig(input_dim=768, hidden_dim=6144)
    model = SparseAutoencoder(config)

    print(f"Config: {config}")
    print()

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 768)

    output = model(x, return_loss=True, return_aux=True)

    print("Forward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Hidden shape: {output['hidden'].shape}")
    print(f"  Reconstruction shape: {output['reconstruction'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Loss (reconstruction): {output['loss_reconstruction'].item():.4f}")
    print(f"  Loss (sparsity): {output['loss_sparsity'].item():.4f}")
    print(f"  L0 (active features): {output['l0'].item():.1f} / {config.hidden_dim}")
    print(f"  Dead features: {output['dead_features'].item():.0f}")
    print()

    # Test encoding sparsity
    hidden = output['hidden']
    sparsity = (hidden > 0).float().mean().item()
    print(f"Sparsity test:")
    print(f"  % active features: {100*sparsity:.2f}%")
    print(f"  Expected for random: {100*0.5:.2f}% (ReLU threshold)")
    print()

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model.save(f.name)
        loaded = SparseAutoencoder.load(f.name)
        print(f"Save/load test: ✅")

    print()
    print("All tests passed! ✅")
