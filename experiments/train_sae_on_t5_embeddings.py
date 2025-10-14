"""
Train Sparse Autoencoder on T5 Text Embeddings
===============================================

This script trains an SAE to discover monosemantic emotion-encoding
features in T5 text embeddings.

Training strategy:
1. Load 100 T5 embeddings (25 per emotion)
2. Train overcomplete SAE (768 → 6144 → 768)
3. Monitor sparsity, dead features, reconstruction
4. Save best model based on validation loss

Expected outcome:
- L0 norm: 10-50 active features per sample (sparse!)
- Reconstruction loss: < 0.01 (good reconstruction)
- Interpretable features that activate for specific emotions

Usage:
    python experiments/train_sae_on_t5_embeddings.py

Author: MusicGen Emotion Interpretability Research
Date: October 2024
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models.sparse_autoencoder import (
    SparseAutoencoder,
    SAEConfig,
    SAETrainer,
    create_sae_for_t5_embeddings
)
from utils.dataset_utils import create_dataloaders, load_t5_embeddings_with_metadata


# ================================================================================
# Configuration
# ================================================================================

CONFIG = {
    # Data
    'data_dir': 'results/t5_embeddings',
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,

    # Model architecture
    'expansion_factor': 8,        # 768 * 8 = 6144 hidden dimensions
    'l1_coefficient': 3e-3,       # Sparsity penalty (higher = sparser)

    # Training
    'batch_size': 16,
    'learning_rate': 1e-3,
    'num_epochs': 500,
    'patience': 50,               # Early stopping patience

    # Feature management
    'reinit_dead_features_every': 100,   # Steps between dead feature reinitialization
    'dead_feature_threshold': 100,       # Steps before feature considered dead

    # Logging
    'log_every': 10,              # Log metrics every N steps
    'validate_every': 50,         # Validate every N steps
    'save_every': 100,            # Save checkpoint every N steps

    # Output
    'output_dir': 'results/sae_training',
    'experiment_name': None,      # Auto-generated if None

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
}


# ================================================================================
# Training Loop
# ================================================================================

def train_sae(config: dict):
    """Main training function"""

    print("=" * 80)
    print("TRAINING SPARSE AUTOENCODER ON T5 EMBEDDINGS")
    print("=" * 80)
    print()

    # Create output directory
    if config['experiment_name'] is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['experiment_name'] = f"sae_t5_exp{config['expansion_factor']}_l1{config['l1_coefficient']:.0e}_{timestamp}"

    output_dir = Path(config['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Load data
    print("Loading data...")
    embeddings, labels, metadata = load_t5_embeddings_with_metadata(config['data_dir'])
    print(f"✅ Loaded {len(embeddings)} T5 embeddings")
    print(f"   Emotions: {metadata['emotions']}")
    print()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        embeddings_path=f"{config['data_dir']}/embeddings.npy",
        labels_path=f"{config['data_dir']}/labels.npy",
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        val_split=config['val_split'],
        test_split=config['test_split'],
        center=True,          # Center the data
        normalize=False,      # Don't normalize (preserve magnitude info)
        shuffle=True
    )
    print(f"✅ Train: {len(train_loader)} batches")
    print(f"✅ Val: {len(val_loader)} batches")
    print(f"✅ Test: {len(test_loader)} batches")
    print()

    # Compute data mean for pre-bias
    print("Computing data statistics...")
    all_embeddings = []
    for batch in train_loader:
        if isinstance(batch, tuple):
            batch_emb, _ = batch
        else:
            batch_emb = batch
        all_embeddings.append(batch_emb)
    train_mean = torch.cat(all_embeddings, dim=0).mean(dim=0)
    print(f"✅ Train mean computed")
    print()

    # Create model
    print("Creating model...")
    model, optimizer = create_sae_for_t5_embeddings(
        expansion_factor=config['expansion_factor'],
        l1_coefficient=config['l1_coefficient'],
        learning_rate=config['learning_rate']
    )

    # Set pre-bias to training mean
    model.set_pre_bias(train_mean)

    # Move to device
    device = config['device']
    model = model.to(device)
    print(f"✅ Model created")
    print(f"   Input dim: {model.config.input_dim}")
    print(f"   Hidden dim: {model.config.hidden_dim}")
    print(f"   Expansion factor: {config['expansion_factor']}x")
    print(f"   L1 coefficient: {config['l1_coefficient']}")
    print(f"   Device: {device}")
    print()

    # Create trainer
    trainer = SAETrainer(model, optimizer, device=device)

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    # Metrics storage
    train_metrics = {
        'loss': [],
        'loss_reconstruction': [],
        'loss_sparsity': [],
        'l0': [],
        'dead_features': [],
        'steps': []
    }
    val_metrics = {
        'loss': [],
        'loss_reconstruction': [],
        'loss_sparsity': [],
        'l0': [],
        'steps': []
    }

    # ============================================================================
    # Training Loop
    # ============================================================================

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")

        # Training
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Extract embeddings
            if isinstance(batch, tuple):
                batch_emb, _ = batch
            else:
                batch_emb = batch

            # Training step
            metrics = trainer.train_step(batch_emb)
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'recon': f"{metrics['loss_reconstruction']:.4f}",
                'l0': f"{metrics['l0']:.0f}",
                'dead': f"{metrics['dead_features']:.0f}"
            })

            # Log metrics
            if global_step % config['log_every'] == 0:
                for key in ['loss', 'loss_reconstruction', 'loss_sparsity', 'l0', 'dead_features']:
                    train_metrics[key].append(metrics[key])
                train_metrics['steps'].append(global_step)

            # Reinitialize dead features
            if global_step % config['reinit_dead_features_every'] == 0:
                n_dead = model.reinit_dead_features()
                if n_dead > 0:
                    print(f"\n  ⚠️  Reinitialized {n_dead} dead features")

            # Validation
            if global_step % config['validate_every'] == 0:
                val_results = trainer.validate(val_loader)

                # Log validation metrics
                for key in ['val_loss', 'val_loss_reconstruction', 'val_loss_sparsity', 'val_l0']:
                    short_key = key.replace('val_', '')
                    if short_key not in val_metrics:
                        val_metrics[short_key] = []
                    val_metrics[short_key].append(val_results[key])
                if 'steps' not in val_metrics:
                    val_metrics['steps'] = []
                val_metrics['steps'].append(global_step)

                print(f"\n  Validation - Loss: {val_results['val_loss']:.4f}, "
                      f"Recon: {val_results['val_loss_reconstruction']:.4f}, "
                      f"L0: {val_results['val_l0']:.0f}")

                # Early stopping check
                if val_results['val_loss'] < best_val_loss:
                    best_val_loss = val_results['val_loss']
                    patience_counter = 0

                    # Save best model
                    model.save(output_dir / 'best_model.pt')
                    print(f"  ✅ New best model saved (val_loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        print(f"\n⚠️  Early stopping triggered (patience: {config['patience']})")
                        break

            # Save checkpoint
            if global_step % config['save_every'] == 0:
                model.save(output_dir / f'checkpoint_step{global_step}.pt')

        # Check for early stopping
        if patience_counter >= config['patience']:
            break

        print()

    # ============================================================================
    # Save Final Results
    # ============================================================================

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()

    # Save final model
    model.save(output_dir / 'final_model.pt')
    print(f"✅ Final model saved")

    # Save metrics
    with open(output_dir / 'train_metrics.json', 'w') as f:
        json.dump({k: [float(v) if torch.is_tensor(v) else v for v in vals]
                  for k, vals in train_metrics.items()}, f, indent=2)

    with open(output_dir / 'val_metrics.json', 'w') as f:
        json.dump({k: [float(v) if torch.is_tensor(v) else v for v in vals]
                  for k, vals in val_metrics.items()}, f, indent=2)

    print(f"✅ Metrics saved")

    # Plot training curves
    print()
    print("Generating plots...")
    plot_training_curves(train_metrics, val_metrics, output_dir)
    print(f"✅ Plots saved to {output_dir}")

    # Final evaluation on test set
    print()
    print("Evaluating on test set...")
    test_results = trainer.validate(test_loader)
    print(f"Test Results:")
    print(f"  Loss: {test_results['val_loss']:.4f}")
    print(f"  Reconstruction: {test_results['val_loss_reconstruction']:.4f}")
    print(f"  Sparsity: {test_results['val_loss_sparsity']:.4f}")
    print(f"  L0 (active features): {test_results['val_l0']:.0f} / {model.config.hidden_dim}")
    print(f"  Sparsity: {100 * test_results['val_l0'] / model.config.hidden_dim:.1f}%")

    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)

    print()
    print(f"All results saved to: {output_dir}")
    print()
    print("=" * 80)

    return model, train_metrics, val_metrics, test_results


def plot_training_curves(train_metrics, val_metrics, output_dir):
    """Plot and save training curves"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(train_metrics['steps'], train_metrics['loss'], label='Train', alpha=0.7)
    if 'steps' in val_metrics and len(val_metrics['steps']) > 0:
        axes[0, 0].plot(val_metrics['steps'], val_metrics['loss'], label='Val', marker='o')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction Loss
    axes[0, 1].plot(train_metrics['steps'], train_metrics['loss_reconstruction'], label='Train', alpha=0.7)
    if 'steps' in val_metrics and len(val_metrics['steps']) > 0:
        axes[0, 1].plot(val_metrics['steps'], val_metrics['loss_reconstruction'], label='Val', marker='o')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reconstruction Loss (MSE)')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Sparsity (L0)
    axes[1, 0].plot(train_metrics['steps'], train_metrics['l0'], label='Train', alpha=0.7)
    if 'steps' in val_metrics and len(val_metrics['steps']) > 0:
        axes[1, 0].plot(val_metrics['steps'], val_metrics['l0'], label='Val', marker='o')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('L0 (Active Features)')
    axes[1, 0].set_title('Sparsity (Lower = More Sparse)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Dead Features
    axes[1, 1].plot(train_metrics['steps'], train_metrics['dead_features'], alpha=0.7)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Number of Dead Features')
    axes[1, 1].set_title('Dead Features')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300)
    plt.close()


# ================================================================================
# Main
# ================================================================================

if __name__ == '__main__':
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "TRAIN SAE ON T5 TEXT EMBEDDINGS" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This script trains a Sparse Autoencoder to discover monosemantic")
    print("emotion-encoding features in T5 text embeddings.")
    print()
    print("Expected training time: 5-10 minutes")
    print("Expected outcome: Sparse features that activate for specific emotions")
    print()

    # Train the model
    model, train_metrics, val_metrics, test_results = train_sae(CONFIG)

    print()
    print("✅ TRAINING COMPLETE!")
    print()
    print("Next steps:")
    print("  1. Analyze learned features with: experiments/analyze_sae_features.py")
    print("  2. Visualize feature activations for different emotions")
    print("  3. Identify monosemantic emotion features")
    print()
