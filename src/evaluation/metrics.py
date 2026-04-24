"""
metrics.py — Aggregate evaluation runner.
Computes pitch histogram similarity, rhythm diversity, and repetition ratio
for all generated MIDI files and saves a summary CSV + plots.

Usage:
    python src/evaluation/metrics.py --generated_dir outputs/generated_midis/ --report_out outputs/plots/
"""
import os, sys, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.pitch_histogram import compute_pitch_histogram, histogram_similarity
from evaluation.rhythm_score import rhythm_diversity, repetition_ratio

try:
    import pretty_midi
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False


def evaluate_midi_file(midi_path: str) -> dict:
    """Compute all metrics for one MIDI file."""
    result = {"file": os.path.basename(midi_path)}
    if not HAS_MIDI:
        return result

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        result["error"] = str(e)
        return result

    pitches  = [n.pitch for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    durs     = [round(n.end - n.start, 3) for inst in pm.instruments
                for n in inst.notes if not inst.is_drum]

    if not pitches:
        return result

    hist     = compute_pitch_histogram(pitches)
    uniform  = np.ones(12) / 12.0          # reference: uniform distribution
    result["pitch_hist_sim"]   = float(histogram_similarity(hist, uniform))
    result["rhythm_diversity"] = float(rhythm_diversity(durs))
    result["repetition_ratio"] = float(repetition_ratio(pitches, window=4))
    result["n_notes"]          = len(pitches)
    result["duration_sec"]     = float(pm.get_end_time())
    return result


def evaluate_folder(generated_dir: str, report_out: str):
    os.makedirs(report_out, exist_ok=True)
    midi_files = (glob.glob(os.path.join(generated_dir, "**/*.mid"), recursive=True) +
                  glob.glob(os.path.join(generated_dir, "**/*.midi"), recursive=True))

    if not midi_files:
        print(f"No MIDI files found in {generated_dir}")
        return

    results = [evaluate_midi_file(f) for f in midi_files]
    df = pd.DataFrame(results)

    csv_path = os.path.join(report_out, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation CSV: {csv_path}")
    print(df.describe().to_string())

    # Bar charts
    numeric_cols = ["pitch_hist_sim", "rhythm_diversity", "repetition_ratio"]
    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(df)), df[col].fillna(0))
            plt.xlabel("Sample"); plt.ylabel(col)
            plt.title(f"Per-Sample {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(report_out, f"{col}.png"), dpi=150)
            plt.close()

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_dir", default="outputs/generated_midis")
    ap.add_argument("--report_out",    default="outputs/plots")
    a = ap.parse_args()
    evaluate_folder(a.generated_dir, a.report_out)
