#!/usr/bin/env python3
"""
Prepare emotion-labeled music datasets for MusicGen interpretability research.

Datasets supported:
- PMEmo: 794 songs with valence/arousal
- DEAM: 1,802 excerpts with dynamic emotion labels
- Custom: Your own labeled dataset
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def create_sample_dataset():
    """
    Create a small sample dataset for testing.

    This creates synthetic metadata that you can replace with real data.
    """
    print("Creating sample dataset for testing...")

    # Sample prompts organized by emotion
    emotions_prompts = {
        'happy': [
            "upbeat cheerful pop music with bright melody",
            "energetic happy dance music",
            "joyful acoustic guitar with positive vibes",
            "bright and optimistic electronic music",
            "fun and playful piano melody"
        ],
        'sad': [
            "melancholic piano ballad",
            "sad emotional strings",
            "somber and reflective ambient music",
            "sorrowful violin melody",
            "depressing slow tempo minor key music"
        ],
        'calm': [
            "peaceful ambient meditation music",
            "relaxing gentle acoustic guitar",
            "serene spa background music",
            "tranquil nature sounds with soft piano",
            "calm and soothing atmospheric music"
        ],
        'energetic': [
            "high energy rock music",
            "intense electronic dance music",
            "fast paced drum and bass",
            "powerful upbeat techno",
            "energetic workout music"
        ]
    }

    # Create dataset
    data = []
    for emotion, prompts in emotions_prompts.items():
        for i, prompt in enumerate(prompts):
            # Estimate valence/arousal based on emotion
            if emotion == 'happy':
                valence, arousal = 0.8, 0.7
            elif emotion == 'sad':
                valence, arousal = 0.2, 0.3
            elif emotion == 'calm':
                valence, arousal = 0.6, 0.2
            elif emotion == 'energetic':
                valence, arousal = 0.7, 0.9
            else:
                valence, arousal = 0.5, 0.5

            # Add some noise
            valence += np.random.uniform(-0.1, 0.1)
            arousal += np.random.uniform(-0.1, 0.1)
            valence = np.clip(valence, 0, 1)
            arousal = np.clip(arousal, 0, 1)

            data.append({
                'id': f'{emotion}_{i}',
                'prompt': prompt,
                'emotion': emotion,
                'valence': valence,
                'arousal': arousal,
            })

    df = pd.DataFrame(data)

    # Save
    output_path = Path('data/processed/sample_emotion_dataset.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Created sample dataset: {output_path}")
    print(f"   {len(df)} samples across {len(emotions_prompts)} emotions")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())

    return df


def download_pmemo_info():
    """
    Provide information about downloading PMEmo dataset.
    """
    print("\n" + "=" * 70)
    print("PMEmo Dataset Information")
    print("=" * 70)
    print()
    print("PMEmo is a dataset of 794 songs with valence/arousal annotations.")
    print()
    print("To download PMEmo:")
    print("1. Visit: https://github.com/HuiZhangDB/PMEmo")
    print("2. Follow their instructions to request access")
    print("3. Download the dataset files")
    print("4. Extract to: data/raw/pmemo/")
    print()
    print("Expected structure:")
    print("  data/raw/pmemo/")
    print("  ├── audio/          # Audio files")
    print("  └── annotations.csv # Emotion annotations")
    print()


def download_deam_info():
    """
    Provide information about downloading DEAM dataset.
    """
    print("\n" + "=" * 70)
    print("DEAM Dataset Information")
    print("=" * 70)
    print()
    print("DEAM contains 1,802 music excerpts with valence/arousal ratings.")
    print()
    print("To download DEAM:")
    print("1. Visit: http://cvml.unige.ch/databases/DEAM/")
    print("2. Fill out the download form")
    print("3. Download the dataset")
    print("4. Extract to: data/raw/deam/")
    print()
    print("Expected structure:")
    print("  data/raw/deam/")
    print("  ├── audio/          # Audio files")
    print("  └── annotations/    # Annotation files")
    print()


def process_pmemo(input_dir: str, output_dir: str):
    """
    Process PMEmo dataset into standard format.

    Args:
        input_dir: Path to raw PMEmo data
        output_dir: Path to save processed data
    """
    print(f"Processing PMEmo dataset from {input_dir}...")

    # TODO: Implement actual PMEmo processing
    # This is a placeholder for when you download the real dataset

    print("⚠️  PMEmo processing not yet implemented")
    print("   Please download the dataset first and implement processing")


def process_deam(input_dir: str, output_dir: str):
    """
    Process DEAM dataset into standard format.

    Args:
        input_dir: Path to raw DEAM data
        output_dir: Path to save processed data
    """
    print(f"Processing DEAM dataset from {input_dir}...")

    # TODO: Implement actual DEAM processing
    # This is a placeholder for when you download the real dataset

    print("⚠️  DEAM processing not yet implemented")
    print("   Please download the dataset first and implement processing")


def map_valence_arousal_to_emotion(valence: float, arousal: float) -> str:
    """
    Map valence/arousal to discrete emotion category.

    Uses Russell's circumplex model:
    - High valence, High arousal → Happy/Energetic
    - High valence, Low arousal → Calm/Peaceful
    - Low valence, High arousal → Angry/Tense
    - Low valence, Low arousal → Sad/Melancholic

    Args:
        valence: Valence value (0-1)
        arousal: Arousal value (0-1)

    Returns:
        Emotion category string
    """
    if valence >= 0.5 and arousal >= 0.5:
        return "happy"
    elif valence >= 0.5 and arousal < 0.5:
        return "calm"
    elif valence < 0.5 and arousal >= 0.5:
        return "tense"
    else:
        return "sad"


def main():
    """Main dataset preparation routine."""
    parser = argparse.ArgumentParser(
        description="Prepare emotion-labeled music datasets"
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample dataset for testing'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset download information'
    )
    parser.add_argument(
        '--process-pmemo',
        type=str,
        metavar='DIR',
        help='Process PMEmo dataset from directory'
    )
    parser.add_argument(
        '--process-deam',
        type=str,
        metavar='DIR',
        help='Process DEAM dataset from directory'
    )

    args = parser.parse_args()

    if args.info:
        download_pmemo_info()
        download_deam_info()
        return

    if args.create_sample:
        create_sample_dataset()

    if args.process_pmemo:
        process_pmemo(args.process_pmemo, 'data/processed/')

    if args.process_deam:
        process_deam(args.process_deam, 'data/processed/')

    if not any([args.create_sample, args.info, args.process_pmemo, args.process_deam]):
        # Default: create sample dataset
        print("No options specified. Creating sample dataset...")
        print("Use --help to see all options")
        print()
        create_sample_dataset()
        print()
        print("To get information about downloading real datasets:")
        print("  python scripts/prepare_datasets.py --info")


if __name__ == "__main__":
    main()
