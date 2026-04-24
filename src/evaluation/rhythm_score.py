"""
rhythm_score.py — Rhythm diversity and repetition ratio metrics.

Rhythm Diversity Score:
    D_rhythm = #unique_durations / #total_notes

Repetition Ratio:
    R = #repeated_patterns / #total_patterns
"""
from typing import List
from collections import Counter


def rhythm_diversity(durations: List[float], n_bins: int = 32,
                     max_dur: float = 4.0) -> float:
    """
    Compute rhythm diversity: proportion of unique quantised duration bins.

    D_rhythm = len(unique_durations) / len(total_notes)
    Range: (0, 1]  — higher is more rhythmically diverse.
    """
    if not durations:
        return 0.0
    bin_size = max_dur / n_bins
    quantised = [min(int(d / bin_size), n_bins - 1) for d in durations]
    return len(set(quantised)) / max(len(quantised), 1)


def repetition_ratio(pitches: List[int], window: int = 4) -> float:
    """
    Compute repetition ratio: fraction of n-gram patterns that repeat.

    R = #repeated_patterns / #total_patterns
    Range: [0, 1] — lower is less repetitive (more creative).

    Args:
        pitches : List of MIDI pitch values
        window  : n-gram size
    """
    if len(pitches) < window:
        return 0.0
    ngrams = [tuple(pitches[i:i+window]) for i in range(len(pitches) - window + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / max(len(counts), 1)


def inter_onset_interval_diversity(onsets: List[float],
                                   quantise_ms: int = 50) -> float:
    """
    Diversity of inter-onset intervals (IOI), quantised to `quantise_ms` ms.
    D_IOI = unique_IOIs / total_IOIs
    """
    if len(onsets) < 2:
        return 0.0
    iois = [round((onsets[i+1] - onsets[i]) * 1000 / quantise_ms) * quantise_ms
            for i in range(len(onsets) - 1)]
    return len(set(iois)) / max(len(iois), 1)


def summarise_rhythm(durations: List[float], onsets: List[float],
                     pitches: List[int]) -> dict:
    """Return a full rhythm summary dict."""
    return {
        "rhythm_diversity":      rhythm_diversity(durations),
        "repetition_ratio":      repetition_ratio(pitches, window=4),
        "ioi_diversity":         inter_onset_interval_diversity(onsets),
        "n_notes":               len(pitches),
        "mean_duration_sec":     sum(durations) / max(len(durations), 1),
    }
