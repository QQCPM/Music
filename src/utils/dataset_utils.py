"""
Dataset utilities for T5 embedding SAE training
================================================

Handles:
- Loading T5 embeddings from disk
- Train/val/test splits
- Batching and shuffling
- Data normalization

Author: MusicGen Emotion Interpretability Research
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict
import json


class T5EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for T5 embeddings.

    Loads embeddings and labels from .npy files.
    """

    def __init__(
        self,
        embeddings_path: str,
        labels_path: Optional[str] = None,
        normalize: bool = False,
        center: bool = True
    ):
        """
        Args:
            embeddings_path: Path to embeddings.npy file [N, 768]
            labels_path: Path to labels.npy file [N] (optional)
            normalize: Whether to L2-normalize embeddings
            center: Whether to center embeddings (subtract mean)
        """
        # Load data
        self.embeddings = torch.from_numpy(
            np.load(embeddings_path)
        ).float()

        if labels_path is not None:
            self.labels = torch.from_numpy(
                np.load(labels_path)
            ).long()
        else:
            self.labels = None

        # Store original statistics
        self.mean = self.embeddings.mean(dim=0)
        self.std = self.embeddings.std(dim=0)

        # Apply transformations
        if center:
            self.embeddings = self.embeddings - self.mean

        if normalize:
            self.embeddings = torch.nn.functional.normalize(
                self.embeddings, p=2, dim=1
            )

        self.normalize = normalize
        self.center = center

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.embeddings[idx], self.labels[idx]
        else:
            return self.embeddings[idx]

    def get_stats(self) -> Dict[str, torch.Tensor]:
        """Get dataset statistics"""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.embeddings.min(dim=0).values,
            'max': self.embeddings.max(dim=0).values,
            'num_samples': len(self),
            'embedding_dim': self.embeddings.shape[1],
        }


def create_dataloaders(
    embeddings_path: str,
    labels_path: Optional[str] = None,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    normalize: bool = False,
    center: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from embedding files.

    Args:
        embeddings_path: Path to embeddings.npy
        labels_path: Path to labels.npy (optional)
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        normalize: Whether to L2-normalize embeddings
        center: Whether to center embeddings
        shuffle: Whether to shuffle training data
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"

    # Load full dataset
    dataset = T5EmbeddingDataset(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        normalize=normalize,
        center=center
    )

    # Calculate split sizes
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def load_t5_embeddings_with_metadata(
    data_dir: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load T5 embeddings with metadata.

    Args:
        data_dir: Directory containing embeddings.npy, labels.npy, metadata.json

    Returns:
        embeddings: Tensor of shape [N, 768]
        labels: Tensor of shape [N]
        metadata: Dictionary with dataset info
    """
    data_dir = Path(data_dir)

    # Load embeddings and labels
    embeddings = torch.from_numpy(
        np.load(data_dir / 'embeddings.npy')
    ).float()

    labels = torch.from_numpy(
        np.load(data_dir / 'labels.npy')
    ).long()

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    return embeddings, labels, metadata


if __name__ == '__main__':
    """Test dataset utilities"""

    print("Testing T5 Embedding Dataset utilities...")
    print()

    # Path to our extracted T5 embeddings
    embeddings_path = 'results/t5_embeddings/embeddings.npy'
    labels_path = 'results/t5_embeddings/labels.npy'

    # Create dataset
    print("Loading dataset...")
    dataset = T5EmbeddingDataset(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        center=True,
        normalize=False
    )

    print(f"✅ Loaded {len(dataset)} samples")
    print()

    # Check a sample
    emb, label = dataset[0]
    print(f"Sample 0:")
    print(f"  Embedding shape: {emb.shape}")
    print(f"  Label: {label}")
    print()

    # Get stats
    stats = dataset.get_stats()
    print("Dataset statistics:")
    print(f"  Mean (first 5 dims): {stats['mean'][:5]}")
    print(f"  Std (first 5 dims): {stats['std'][:5]}")
    print(f"  Num samples: {stats['num_samples']}")
    print(f"  Embedding dim: {stats['embedding_dim']}")
    print()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        batch_size=16,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        center=True,
        shuffle=True
    )

    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    print()

    # Test a batch
    batch_emb, batch_labels = next(iter(train_loader))
    print(f"Sample batch:")
    print(f"  Batch embeddings shape: {batch_emb.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    print(f"  Labels in batch: {batch_labels.tolist()}")
    print()

    # Load with metadata
    print("Loading with metadata...")
    embeddings, labels, metadata = load_t5_embeddings_with_metadata(
        'results/t5_embeddings'
    )
    print(f"✅ Loaded {len(embeddings)} embeddings")
    print(f"✅ Metadata keys: {list(metadata.keys())}")
    print(f"✅ Emotions: {metadata['emotions']}")
    print()

    print("All tests passed! ✅")
