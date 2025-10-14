#!/usr/bin/env python3
"""
Comprehensive Emotion Search in MusicGen

Systematically searches for emotion encoding by testing:
1. All layers (0-23)
2. Multiple prompts (extreme emotions)
3. Proper statistics (within vs between similarity)
4. Alternative metrics (linear probe, etc.)

This is a SYSTEMATIC approach, not wishful thinking.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from audiocraft.models import MusicGen
from src.utils.activation_utils import ActivationExtractor, cosine_similarity
from src.utils.audio_utils import extract_audio_features
from pathlib import Path
import json
from tqdm import tqdm
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ============================================================================
# Configuration
# ============================================================================

EXTREME_PROMPTS = {
    'happy': [
        "euphoric celebration music, 150 BPM, major key, bright synthesizers, energetic drums, triumphant",
        "joyful children laughing and playing, upbeat xylophone melody, cheerful, bouncy",
        "victory fanfare, triumphant brass section, fast tempo, exciting, energetic",
    ],
    'sad': [
        "funeral dirge, 50 BPM, solo cello, crying, mournful, dark, deeply sorrowful",
        "heartbreak ballad, slow piano, minor key, melancholic, tears, depressing",
        "rainy day melancholy, somber strings, slow, quiet, sad, lonely",
    ],
    'calm': [
        "zen meditation music, 40 BPM, soft flute, peaceful ambient, gentle, quiet, soothing",
        "gentle lullaby, slow harp, peaceful, serene, relaxing, sleep",
        "tranquil forest sounds, slow tempo, calm nature ambience, peaceful",
    ],
    'energetic': [
        "high energy dance music, 140 BPM, pounding bass, intense drums, loud, powerful",
        "aggressive workout music, fast tempo, driving beat, intense, energetic, strong",
        "adrenaline-pumping action music, 160 BPM, intense orchestral, powerful, epic",
    ],
}

# ============================================================================
# Phase 1: Layer Sweep
# ============================================================================

def test_layer(model, layer_idx, prompts, n_samples=3):
    """
    Test a single layer for emotion differentiation.

    Args:
        model: MusicGen model
        layer_idx: Which layer to test
        prompts: Dict of emotion -> list of prompts
        n_samples: How many samples per prompt

    Returns:
        Signal strength (within - between similarity)
    """
    print(f"\n  Testing layer {layer_idx}...")

    extractor = ActivationExtractor(model, layers=[layer_idx])

    # Generate samples for each emotion
    activations_by_emotion = {}

    for emotion, prompt_list in prompts.items():
        activations_by_emotion[emotion] = []

        for prompt in prompt_list[:n_samples]:  # Limit to n_samples
            extractor.clear_activations()
            wav = extractor.generate([prompt])
            acts = extractor.get_activations(concatenate=True)

            # Get mean activation (average over timesteps)
            layer_key = f'layer_{layer_idx}'
            mean_act = acts[layer_key].mean(dim=0).flatten()  # [d_model]
            activations_by_emotion[emotion].append(mean_act)

    # Compute within-emotion similarity
    within_sims = []
    for emotion, acts_list in activations_by_emotion.items():
        for i in range(len(acts_list)):
            for j in range(i+1, len(acts_list)):
                sim = cosine_similarity(acts_list[i], acts_list[j])
                within_sims.append(sim)

    # Compute between-emotion similarity
    between_sims = []
    emotions = list(activations_by_emotion.keys())
    for i, e1 in enumerate(emotions):
        for e2 in emotions[i+1:]:
            acts1 = activations_by_emotion[e1]
            acts2 = activations_by_emotion[e2]
            for a1 in acts1:
                for a2 in acts2:
                    sim = cosine_similarity(a1, a2)
                    between_sims.append(sim)

    mean_within = np.mean(within_sims) if within_sims else 0
    mean_between = np.mean(between_sims) if between_sims else 0
    signal = mean_within - mean_between

    print(f"    Within: {mean_within:.4f}, Between: {mean_between:.4f}, Signal: {signal:.4f}")

    return {
        'layer': layer_idx,
        'within': mean_within,
        'between': mean_between,
        'signal': signal,
        'activations': activations_by_emotion
    }


def layer_sweep(model, prompts, layers_to_test=None):
    """
    Test all layers to find which encodes emotions best.
    """
    print("=" * 80)
    print("PHASE 1: Layer Sweep")
    print("=" * 80)
    print()
    print("Testing which layer(s) encode emotions...")
    print(f"Using {sum(len(p) for p in prompts.values())} prompts across {len(prompts)} emotions")
    print()

    if layers_to_test is None:
        # Get number of layers from model
        num_layers = len(model.lm.transformer.layers)
        layers_to_test = list(range(num_layers))

    results = []

    for layer_idx in tqdm(layers_to_test, desc="Testing layers"):
        result = test_layer(model, layer_idx, prompts, n_samples=2)
        results.append(result)

    # Find best layer
    best = max(results, key=lambda r: r['signal'])

    print()
    print("Results:")
    print("-" * 80)
    for r in results:
        indicator = " ← BEST" if r['layer'] == best['layer'] else ""
        print(f"  Layer {r['layer']:2d}: signal = {r['signal']:.4f}{indicator}")

    print()
    print(f"Best layer: {best['layer']} with signal = {best['signal']:.4f}")

    return best


# ============================================================================
# Phase 2: Prompt Validation
# ============================================================================

def validate_prompt_acoustically(prompt, expected_emotion, model):
    """
    Check if a prompt actually produces the intended emotion acoustically.
    """
    # Generate audio
    model.set_generation_params(duration=8)
    wav = model.generate([prompt])

    # Save to temp file
    temp_file = "/tmp/musicgen_test.wav"
    from src.utils.audio_utils import save_audio
    save_audio(wav[0], temp_file.replace('.wav', ''), model.sample_rate, use_ffmpeg=False)

    # Extract features
    features = extract_audio_features(temp_file + '.wav', sr=model.sample_rate, duration=8.0)

    # Check if features match expected emotion
    valid = True
    issues = []

    if expected_emotion == 'happy':
        if features['tempo'] < 100:
            valid = False
            issues.append(f"tempo too slow ({features['tempo']:.1f} BPM)")
        if features['mode'] != 'major':
            valid = False
            issues.append(f"not major key ({features['mode']})")

    elif expected_emotion == 'sad':
        if features['tempo'] > 110:
            valid = False
            issues.append(f"tempo too fast ({features['tempo']:.1f} BPM)")
        if features['mode'] != 'minor':
            valid = False
            issues.append(f"not minor key ({features['mode']})")

    elif expected_emotion == 'calm':
        if features['tempo'] > 90:
            valid = False
            issues.append(f"tempo too fast ({features['tempo']:.1f} BPM)")
        if features['rms_mean'] > 0.1:
            valid = False
            issues.append(f"too loud ({features['rms_mean']:.4f})")

    elif expected_emotion == 'energetic':
        if features['tempo'] < 120:
            valid = False
            issues.append(f"tempo too slow ({features['tempo']:.1f} BPM)")
        if features['rms_mean'] < 0.08:
            valid = False
            issues.append(f"too quiet ({features['rms_mean']:.4f})")

    return valid, features, issues


def prompt_validation(prompts, model):
    """
    Test which prompts actually produce different music.
    """
    print()
    print("=" * 80)
    print("PHASE 2: Prompt Validation")
    print("=" * 80)
    print()
    print("Testing which prompts produce acoustically different music...")
    print()

    valid_prompts = {}
    all_features = {}

    for emotion, prompt_list in prompts.items():
        print(f"Testing {emotion} prompts...")
        valid_prompts[emotion] = []
        all_features[emotion] = []

        for prompt in prompt_list:
            print(f"  '{prompt[:50]}...'")
            valid, features, issues = validate_prompt_acoustically(prompt, emotion, model)

            all_features[emotion].append(features)

            if valid:
                print(f"    ✅ VALID - Tempo: {features['tempo']:.1f} BPM, Mode: {features['mode']}")
                valid_prompts[emotion].append(prompt)
            else:
                print(f"    ❌ INVALID - {', '.join(issues)}")

        print(f"  Valid prompts: {len(valid_prompts[emotion])}/{len(prompt_list)}")
        print()

    # Summary
    print("Acoustic Feature Summary:")
    print("-" * 80)
    for emotion, features_list in all_features.items():
        tempos = [f['tempo'] for f in features_list]
        energies = [f['rms_mean'] for f in features_list]
        print(f"{emotion:12s}: Tempo = {np.mean(tempos):5.1f} ± {np.std(tempos):4.1f} BPM, "
              f"Energy = {np.mean(energies):.4f} ± {np.std(energies):.4f}")

    return valid_prompts


# ============================================================================
# Phase 3: Statistical Testing with Best Configuration
# ============================================================================

def generate_dataset(model, layer, prompts, n_samples=10):
    """
    Generate dataset with best layer and validated prompts.
    """
    print()
    print("=" * 80)
    print("PHASE 3: Dataset Generation")
    print("=" * 80)
    print()
    print(f"Generating {n_samples} samples per emotion using layer {layer}...")
    print()

    extractor = ActivationExtractor(model, layers=[layer])
    activations_by_emotion = {}

    for emotion, prompt_list in prompts.items():
        print(f"Generating {emotion} samples...")
        activations_by_emotion[emotion] = []

        # Use all valid prompts, cycling if needed
        for i in range(n_samples):
            prompt = prompt_list[i % len(prompt_list)]
            extractor.clear_activations()
            wav = extractor.generate([prompt])
            acts = extractor.get_activations(concatenate=True)

            layer_key = f'layer_{layer}'
            mean_act = acts[layer_key].mean(dim=0).flatten()
            activations_by_emotion[emotion].append(mean_act)

        print(f"  Generated {len(activations_by_emotion[emotion])} samples")

    return activations_by_emotion


def statistical_test(activations_by_emotion):
    """
    Proper within vs between similarity test.
    """
    print()
    print("=" * 80)
    print("PHASE 4: Statistical Testing")
    print("=" * 80)
    print()

    # Within-emotion similarity
    within_sims = []
    for emotion, acts_list in activations_by_emotion.items():
        for i in range(len(acts_list)):
            for j in range(i+1, len(acts_list)):
                sim = cosine_similarity(acts_list[i], acts_list[j])
                within_sims.append(sim)

    # Between-emotion similarity
    between_sims = []
    emotions = list(activations_by_emotion.keys())
    for i, e1 in enumerate(emotions):
        for e2 in emotions[i+1:]:
            acts1 = activations_by_emotion[e1]
            acts2 = activations_by_emotion[e2]
            for a1 in acts1:
                for a2 in acts2:
                    sim = cosine_similarity(a1, a2)
                    between_sims.append(sim)

    # Statistics
    mean_within = np.mean(within_sims)
    std_within = np.std(within_sims)
    mean_between = np.mean(between_sims)
    std_between = np.std(between_sims)

    # T-test
    t_stat, p_value = ttest_ind(within_sims, between_sims)

    print(f"Within-emotion similarity:  {mean_within:.4f} ± {std_within:.4f}")
    print(f"Between-emotion similarity: {mean_between:.4f} ± {std_between:.4f}")
    print(f"Difference: {mean_within - mean_between:.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print()

    # Interpretation
    if p_value < 0.001:
        print("✅✅✅ HIGHLY SIGNIFICANT (p < 0.001)")
        verdict = "STRONG"
    elif p_value < 0.01:
        print("✅✅ SIGNIFICANT (p < 0.01)")
        verdict = "MODERATE"
    elif p_value < 0.05:
        print("✅ MARGINALLY SIGNIFICANT (p < 0.05)")
        verdict = "WEAK"
    else:
        print("❌ NOT SIGNIFICANT (p >= 0.05)")
        verdict = "NONE"

    print()

    difference = mean_within - mean_between
    if difference > 0.05:
        print(f"Signal strength: STRONG ({difference:.4f} > 0.05)")
        strength = "STRONG"
    elif difference > 0.02:
        print(f"Signal strength: MODERATE ({difference:.4f} > 0.02)")
        strength = "MODERATE"
    elif difference > 0.01:
        print(f"Signal strength: WEAK ({difference:.4f} > 0.01)")
        strength = "WEAK"
    else:
        print(f"Signal strength: VERY WEAK ({difference:.4f} < 0.01)")
        strength = "VERY_WEAK"

    return {
        'within_mean': mean_within,
        'within_std': std_within,
        'between_mean': mean_between,
        'between_std': std_between,
        'difference': difference,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'verdict': verdict,
        'strength': strength,
    }


# ============================================================================
# Phase 5: Linear Probe (Definitive Test)
# ============================================================================

def linear_probe_test(activations_by_emotion):
    """
    Can a linear classifier predict emotion from activations?
    This is the DEFINITIVE test.
    """
    print()
    print("=" * 80)
    print("PHASE 5: Linear Probe Test")
    print("=" * 80)
    print()
    print("Training linear classifier to predict emotion from activations...")
    print()

    # Prepare data
    X = []
    y = []
    emotion_to_idx = {}

    for idx, (emotion, acts_list) in enumerate(activations_by_emotion.items()):
        emotion_to_idx[emotion] = idx
        for act in acts_list:
            X.append(act.cpu().numpy())
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(emotion_to_idx)} classes")
    print()

    # Train classifier with cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"Cross-validation accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print()

    # Chance level
    chance = 1.0 / len(emotion_to_idx)
    print(f"Chance level: {chance:.2%}")
    print()

    # Interpretation
    if mean_acc > 0.75:
        print("✅✅✅ EXCELLENT: Emotions are clearly encoded (>75% accuracy)")
        verdict = "CLEAR"
    elif mean_acc > 0.6:
        print("✅✅ GOOD: Emotions are encoded (>60% accuracy)")
        verdict = "MODERATE"
    elif mean_acc > chance * 1.5:
        print("✅ WEAK: Some encoding present (>1.5× chance)")
        verdict = "WEAK"
    else:
        print("❌ POOR: No clear encoding (<1.5× chance)")
        verdict = "NONE"

    return {
        'accuracy_mean': mean_acc,
        'accuracy_std': std_acc,
        'chance_level': chance,
        'verdict': verdict,
    }


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE EMOTION SEARCH IN MUSICGEN")
    print("=" * 80)
    print()
    print("This experiment will systematically search for emotion encoding.")
    print()

    # Load model
    print("Loading MusicGen Large...")
    model = MusicGen.get_pretrained('facebook/musicgen-large')
    model.set_generation_params(duration=3)  # Short for speed
    print("✅ Model loaded")
    print()

    # Phase 1: Layer sweep (test subset of layers for speed)
    layers_to_test = [0, 6, 12, 18, 24, 30, 36, 42, 47]  # Sample 9 layers
    best_layer_result = layer_sweep(model, EXTREME_PROMPTS, layers_to_test)
    best_layer = best_layer_result['layer']

    # Phase 2: Prompt validation
    valid_prompts = prompt_validation(EXTREME_PROMPTS, model)

    # Check if we have any valid prompts
    if not any(valid_prompts.values()):
        print("❌ NO VALID PROMPTS FOUND")
        print("MusicGen is not generating different emotions acoustically!")
        print("Consider pivoting research direction.")
        return

    # Phase 3: Generate larger dataset
    activations = generate_dataset(model, best_layer, valid_prompts, n_samples=10)

    # Phase 4: Statistical test
    stats = statistical_test(activations)

    # Phase 5: Linear probe
    probe = linear_probe_test(activations)

    # Save results
    results = {
        'best_layer': best_layer,
        'layer_signal': best_layer_result['signal'],
        'valid_prompts': {k: len(v) for k, v in valid_prompts.items()},
        'statistics': stats,
        'linear_probe': probe,
    }

    output_file = "../results/comprehensive_search_results.json"
    Path("../results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {output_file}")

    # Final verdict
    print()
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    print(f"Best layer: {best_layer} (signal = {best_layer_result['signal']:.4f})")
    print(f"Statistical test: {stats['verdict']} (p = {stats['p_value']:.4f})")
    print(f"Linear probe: {probe['verdict']} (acc = {probe['accuracy_mean']:.2%})")
    print()

    # Decision
    if stats['significant'] and stats['difference'] > 0.02 and probe['accuracy_mean'] > 0.65:
        print("✅✅✅ STRONG SIGNAL FOUND")
        print("Emotions ARE encoded in MusicGen!")
        print()
        print("Recommendation: Proceed to Phase 1 (SAE training)")
    elif stats['significant'] and probe['accuracy_mean'] > 0.55:
        print("✅ WEAK SIGNAL FOUND")
        print("Some emotion encoding exists, but subtle.")
        print()
        print("Recommendation: Scale up dataset (50+ samples) before Phase 1")
    else:
        print("❌ NO CLEAR SIGNAL")
        print("MusicGen does not strongly encode emotions.")
        print()
        print("Recommendation: Pivot to 'Why not?' paper or test other models")


if __name__ == '__main__':
    main()
