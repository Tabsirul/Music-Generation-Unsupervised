"""
rlhf_comparison.py - Compare Transformer (before) vs RLHF (after) outputs
Run: python src/evaluation/rlhf_comparison.py
"""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from evaluation.pitch_histogram import compute_pitch_histogram, histogram_similarity
from evaluation.rhythm_score import rhythm_diversity, repetition_ratio

def evaluate_folder(folder):
    try:
        import pretty_midi
    except ImportError:
        return None
    midi_files = glob.glob(os.path.join(folder, "*.mid"))
    all_pitch_sim, all_rhy_div, all_rep_ratio = [], [], []
    for mf in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(mf)
            pitches = [n.pitch for inst in pm.instruments
                       for n in inst.notes if not inst.is_drum]
            durs    = [n.end - n.start for inst in pm.instruments
                       for n in inst.notes if not inst.is_drum]
            if not pitches:
                continue
            all_pitch_sim.append(histogram_similarity(
                compute_pitch_histogram(pitches), np.ones(12)/12))
            all_rhy_div.append(rhythm_diversity(durs))
            all_rep_ratio.append(repetition_ratio(pitches))
        except:
            pass
    if not all_pitch_sim:
        return None
    return {
        "pitch_hist_sim":   np.mean(all_pitch_sim),
        "rhythm_diversity": np.mean(all_rhy_div),
        "repetition_ratio": np.mean(all_rep_ratio),
    }

def compare():
    before = evaluate_folder(os.path.join(cfg.MIDI_OUT_DIR, "task3"))
    after  = evaluate_folder(os.path.join(cfg.MIDI_OUT_DIR, "task4"))

    if not before or not after:
        print("ERROR: No MIDI files found in task3/ or task4/ folders.")
        return

    metrics = ["pitch_hist_sim", "rhythm_diversity", "repetition_ratio"]
    labels  = ["Pitch Hist Sim ↓ (lower=better)", "Rhythm Diversity ↑ (higher=better)", "Repetition Ratio ↓(lower=better)"]

    print("=== Before vs After RLHF ===")
    print(f"{'Metric':<25} {'Before (Transformer)':>22} {'After (RLHF)':>14} {'Change':>10}")
    print("-" * 75)
    for m, l in zip(metrics, labels):
        b, a = before[m], after[m]
        change = ((a - b) / (b + 1e-8)) * 100
        arrow = "↑" if a > b else "↓"
        print(f"{m:<25} {b:>22.4f} {a:>14.4f} {change:>+9.1f}% {arrow}")

    # Bar chart
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, [before[m] for m in metrics],
                width, label='Transformer (Before)', color='#3498db', alpha=0.8)
    b2 = ax.bar(x + width/2, [after[m]  for m in metrics],
                width, label='RLHF Tuned (After)',  color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Before vs After RLHF Fine-Tuning", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    out = os.path.join(cfg.PLOTS_DIR, "rlhf_before_after.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    compare()
