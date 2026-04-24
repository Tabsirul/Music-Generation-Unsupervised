"""
pitch_histogram.py — Pitch class histogram computation and similarity metrics.

Pitch Histogram Similarity:
    H(p, q) = Σ_{i=1}^{12} |p_i - q_i|
"""
import numpy as np
from typing import List


def compute_pitch_histogram(pitches: List[int], normalise: bool = True) -> np.ndarray:
    """
    Compute 12-bin pitch class histogram.

    Args:
        pitches   : List of MIDI pitch values (0-127)
        normalise : If True, normalise to a probability distribution

    Returns:
        hist : (12,) float array
    """
    hist = np.zeros(12, dtype=np.float64)
    for p in pitches:
        hist[p % 12] += 1
    if normalise and hist.sum() > 0:
        hist /= hist.sum()
    return hist


def histogram_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """
    L1-distance between two pitch histograms (lower = more similar).

    H(p, q) = Σ |p_i - q_i|
    Range: [0, 2]  (0 = identical, 2 = completely disjoint)
    """
    return float(np.sum(np.abs(p - q)))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Cosine similarity between two histograms. Range: [-1, 1]."""
    denom = (np.linalg.norm(p) * np.linalg.norm(q))
    if denom == 0:
        return 0.0
    return float(np.dot(p, q) / denom)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL divergence D_KL(p || q). p, q should be normalised distributions."""
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compare_to_reference(generated_pitches: list,
                          reference_pitches: list) -> dict:
    """Full comparison between generated and reference pitch distributions."""
    p = compute_pitch_histogram(generated_pitches)
    q = compute_pitch_histogram(reference_pitches)
    return {
        "l1_distance":       histogram_similarity(p, q),
        "cosine_similarity": cosine_similarity(p, q),
        "kl_divergence":     kl_divergence(p, q),
        "generated_hist":    p.tolist(),
        "reference_hist":    q.tolist(),
    }
