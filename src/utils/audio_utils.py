"""
Utilities for audio processing and feature extraction.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch


def extract_audio_features(
    audio_path: str,
    sr: int = 32000,
    duration: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract acoustic features from an audio file.

    Features extracted:
    - Tempo (BPM)
    - Key (estimated tonic)
    - Spectral centroid (brightness)
    - RMS energy (loudness)
    - Zero-crossing rate
    - Chroma features (harmony)

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        duration: Optional duration to load (seconds)

    Returns:
        Dictionary of features
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)

    features = {}

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)

    # Spectral centroid (brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))

    # RMS energy (loudness)
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # Chroma features (harmony)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std'] = float(np.std(chroma))

    # Estimate key (0-11 for C, C#, D, ..., B)
    chroma_mean = np.mean(chroma, axis=1)
    estimated_key = int(np.argmax(chroma_mean))
    features['estimated_key'] = estimated_key

    # Estimate mode (major/minor) - simplified
    # Major: chroma pattern has stronger I, III, V
    # Minor: chroma pattern has stronger i, iii, v
    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
    minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]

    features['mode'] = 'major' if major_corr > minor_corr else 'minor'
    features['mode_confidence'] = float(abs(major_corr - minor_corr))

    # MFCCs (timbre)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(rolloff))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))

    return features


def save_audio(
    wav: torch.Tensor,
    filepath: str,
    sample_rate: int = 32000,
    strategy: str = "loudness",
    use_ffmpeg: bool = True
):
    """
    Save audio tensor to file.

    Args:
        wav: Audio tensor [channels, samples] or [samples]
        filepath: Output path (without extension)
        sample_rate: Sample rate
        strategy: Normalization strategy ("loudness", "peak", "clip")
        use_ffmpeg: If True, use FFmpeg (better quality). If False, use soundfile (no FFmpeg needed)
    """
    # Ensure on CPU
    if torch.is_tensor(wav):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav

    # Ensure 2D array [channels, samples]
    if wav_np.ndim == 1:
        wav_np = wav_np.reshape(1, -1)

    # Check if FFmpeg is available
    import shutil
    ffmpeg_available = shutil.which('ffmpeg') is not None

    if use_ffmpeg and ffmpeg_available:
        # Use audiocraft's audio_write with FFmpeg
        from audiocraft.data.audio import audio_write

        # Ensure tensor
        if not torch.is_tensor(wav):
            wav = torch.from_numpy(wav_np)

        audio_write(
            filepath,
            wav.cpu(),
            sample_rate,
            strategy=strategy,
            loudness_compressor=True
        )
    else:
        # Fallback: Use soundfile (no FFmpeg required)
        if not ffmpeg_available and use_ffmpeg:
            print("⚠️  FFmpeg not found. Using soundfile fallback.")
            print("   Install FFmpeg for better quality: brew install ffmpeg")

        # Apply simple normalization
        if strategy == "peak":
            max_val = np.abs(wav_np).max()
            if max_val > 0:
                wav_np = wav_np / max_val * 0.95
        elif strategy == "loudness" or strategy == "clip":
            # Simple RMS normalization
            rms = np.sqrt(np.mean(wav_np ** 2))
            if rms > 0:
                target_rms = 0.1
                wav_np = wav_np * (target_rms / rms)
                # Clip to prevent overflow
                wav_np = np.clip(wav_np, -1.0, 1.0)

        # Transpose to [samples, channels] for soundfile
        wav_np = wav_np.T

        # Save as WAV file
        output_file = f"{filepath}.wav"
        sf.write(output_file, wav_np, sample_rate)
        print(f"   Saved: {output_file}")


def load_audio(
    filepath: str,
    sr: int = 32000,
    mono: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        filepath: Path to audio file
        sr: Target sample rate (None to keep original)
        mono: Convert to mono

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    y, sr_orig = librosa.load(filepath, sr=sr, mono=mono)
    return y, sr


def compute_similarity(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int = 32000,
    method: str = 'chroma'
) -> float:
    """
    Compute similarity between two audio signals.

    Args:
        audio1, audio2: Audio arrays
        sr: Sample rate
        method: Similarity method
            - 'chroma': Harmonic similarity
            - 'mfcc': Timbral similarity
            - 'spectral': Spectral similarity

    Returns:
        Similarity score (0-1)
    """
    if method == 'chroma':
        # Chroma-based harmonic similarity
        chroma1 = librosa.feature.chroma_cqt(y=audio1, sr=sr)
        chroma2 = librosa.feature.chroma_cqt(y=audio2, sr=sr)

        # Align lengths
        min_len = min(chroma1.shape[1], chroma2.shape[1])
        chroma1 = chroma1[:, :min_len]
        chroma2 = chroma2[:, :min_len]

        # Cosine similarity
        similarity = np.mean([
            np.dot(chroma1[:, i], chroma2[:, i]) /
            (np.linalg.norm(chroma1[:, i]) * np.linalg.norm(chroma2[:, i]))
            for i in range(min_len)
        ])

    elif method == 'mfcc':
        # MFCC-based timbral similarity
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr)

        # Align lengths
        min_len = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_len]
        mfcc2 = mfcc2[:, :min_len]

        # Cosine similarity
        similarity = np.mean([
            np.dot(mfcc1[:, i], mfcc2[:, i]) /
            (np.linalg.norm(mfcc1[:, i]) * np.linalg.norm(mfcc2[:, i]))
            for i in range(min_len)
        ])

    elif method == 'spectral':
        # Spectral similarity
        spec1 = np.abs(librosa.stft(audio1))
        spec2 = np.abs(librosa.stft(audio2))

        # Align lengths
        min_len = min(spec1.shape[1], spec2.shape[1])
        spec1 = spec1[:, :min_len]
        spec2 = spec2[:, :min_len]

        # Cosine similarity
        similarity = np.mean([
            np.dot(spec1[:, i], spec2[:, i]) /
            (np.linalg.norm(spec1[:, i]) * np.linalg.norm(spec2[:, i]))
            for i in range(min_len)
        ])

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(np.clip(similarity, 0, 1))


def visualize_waveform(
    audio: np.ndarray,
    sr: int = 32000,
    title: str = "Waveform"
):
    """
    Create waveform visualization.

    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return plt.gcf()


def visualize_spectrogram(
    audio: np.ndarray,
    sr: int = 32000,
    title: str = "Spectrogram"
):
    """
    Create spectrogram visualization.

    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
    """
    import matplotlib.pyplot as plt

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        D,
        sr=sr,
        x_axis='time',
        y_axis='hz'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def estimate_emotion_from_features(
    features: Dict[str, float]
) -> Tuple[str, Dict[str, float]]:
    """
    Rough emotion estimation from acoustic features.

    Uses simple heuristics based on music psychology research:
    - High tempo + bright timbre → Happy/Energetic
    - Low tempo + dark timbre → Sad/Calm
    - Major mode → Positive valence
    - Minor mode → Negative valence

    Args:
        features: Dictionary from extract_audio_features()

    Returns:
        Tuple of (emotion_label, valence_arousal_dict)
    """
    tempo = features['tempo']
    spectral_centroid = features['spectral_centroid_mean']
    mode = features['mode']
    rms = features['rms_mean']

    # Estimate arousal (energy) from tempo and RMS
    # Normalize tempo (typical range: 60-180 BPM)
    arousal = (
        0.6 * (tempo - 60) / 120 +
        0.4 * (rms / 0.1)  # Rough normalization
    )
    arousal = np.clip(arousal, 0, 1)

    # Estimate valence from mode and brightness
    # Major mode and bright timbre → positive
    mode_valence = 0.7 if mode == 'major' else 0.3
    brightness_valence = spectral_centroid / 5000  # Rough normalization
    valence = np.clip(0.6 * mode_valence + 0.4 * brightness_valence, 0, 1)

    # Map to quadrants
    if valence >= 0.5 and arousal >= 0.5:
        emotion = "happy"
    elif valence >= 0.5 and arousal < 0.5:
        emotion = "calm"
    elif valence < 0.5 and arousal >= 0.5:
        emotion = "tense"
    else:
        emotion = "sad"

    return emotion, {
        'valence': float(valence),
        'arousal': float(arousal)
    }
