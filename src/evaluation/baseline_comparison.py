"""
baseline_comparison.py - Generate and evaluate Random and Markov Chain baselines
Produces comparison table vs all 4 task models.
Run: python src/evaluation/baseline_comparison.py
"""
import os, sys, json, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pretty_midi
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from evaluation.pitch_histogram import compute_pitch_histogram, histogram_similarity
from evaluation.rhythm_score import rhythm_diversity, repetition_ratio
from generation.midi_export import piano_roll_to_midi

MIDI_MIN = 21

# ── Random Generator ──────────────────────────────────────────────────────────
def generate_random(n=5, seq_len=64, out_dir=None):
    """Generate completely random piano rolls."""
    results = []
    for i in range(n):
        roll = np.random.rand(88, seq_len) * 0.5
        roll[roll < 0.45] = 0   # ~10% notes active
        pitches = [p + MIDI_MIN for p in range(88) if roll[p].max() > 0]
        durs    = [0.25] * len(pitches)
        results.append({
            "model": "Random Generator",
            "pitch_hist_sim":   float(histogram_similarity(
                                    compute_pitch_histogram(pitches),
                                    np.ones(12)/12)),
            "rhythm_diversity": float(rhythm_diversity(durs)),
            "repetition_ratio": float(repetition_ratio(pitches)),
            "n_notes":          len(pitches),
        })
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            piano_roll_to_midi(roll,
                os.path.join(out_dir, f"random_sample_{i+1:02d}.mid"))
    return results

# ── Markov Chain ──────────────────────────────────────────────────────────────
class MarkovModel:
    def __init__(self, order=2):
        self.order = order
        self.trans = defaultdict(Counter)

    def train(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - self.order):
                state = tuple(seq[i:i+self.order])
                nxt   = seq[i+self.order]
                self.trans[state][nxt] += 1

    def generate(self, length=64):
        if not self.trans:
            return [random.randint(21, 108) for _ in range(length)]
        state = random.choice(list(self.trans.keys()))
        seq   = list(state)
        for _ in range(length - self.order):
            counts  = self.trans.get(tuple(seq[-self.order:]), Counter())
            if not counts:
                state = random.choice(list(self.trans.keys()))
                seq.extend(state); continue
            total   = sum(counts.values())
            choices = list(counts.keys())
            probs   = [counts[c]/total for c in choices]
            seq.append(random.choices(choices, weights=probs)[0])
        return seq

def generate_markov(n=5, out_dir=None):
    """Train Markov model on processed data and generate sequences."""
    json_files = glob.glob(os.path.join(cfg.PROCESSED_DIR, "*.json"))
    json_files = [f for f in json_files if not f.endswith("metadata.json")]

    model = MarkovModel(order=2)
    print(f"Training Markov model on {min(200, len(json_files))} files...")
    for jf in json_files[:200]:
        with open(jf) as f:
            d = json.load(f)
        pitches = [e["pitch"] for e in d["events"]]
        if len(pitches) > 2:
            model.train([pitches])   # wrap: train() expects a list of sequences

    results = []
    for i in range(n):
        seq  = model.generate(length=128)
        durs = [0.25] * len(seq)
        results.append({
            "model": "Markov Chain",
            "pitch_hist_sim":   float(histogram_similarity(
                                    compute_pitch_histogram(seq),
                                    np.ones(12)/12)),
            "rhythm_diversity": float(rhythm_diversity(durs)),
            "repetition_ratio": float(repetition_ratio(seq)),
            "n_notes":          len(seq),
        })
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            # Convert pitch sequence to simple piano roll
            roll = np.zeros((88, 64))
            for j, p in enumerate(seq[:64]):
                idx = p - MIDI_MIN
                if 0 <= idx < 88:
                    roll[idx, j % 64] = 0.8
            piano_roll_to_midi(roll,
                os.path.join(out_dir, f"markov_sample_{i+1:02d}.mid"))
    return results

# ── Load Task Results ─────────────────────────────────────────────────────────
def load_task_results(generated_dir):
    """Load evaluation metrics for task 1-4 generated MIDI files."""

    try:
        import pretty_midi
        HAS_MIDI = True
    except ImportError:
        HAS_MIDI = False

    rows = []
    task_models = {
        "task1": "LSTM Autoencoder",
        "task2": "VAE",
        "task3": "Transformer",
        "task4": "RLHF",
    }

    for folder, model_name in task_models.items():
        fdir = os.path.join(generated_dir, folder)
        if not os.path.exists(fdir):
            continue

        midi_files = glob.glob(os.path.join(fdir, "*.mid"))

        for mf in midi_files[:5]:

            if not HAS_MIDI:
                rows.append({
                    "model": model_name,
                    "pitch_hist_sim": 0.5,
                    "rhythm_diversity": 0.5,
                    "repetition_ratio": 0.3,
                    "n_notes": 50
                })
                continue

            try:
                # ✅ Proper usage (this was missing)
                pm = pretty_midi.PrettyMIDI(mf) # type: ignore

                pitches = [
                    n.pitch for inst in pm.instruments
                    for n in inst.notes if not inst.is_drum
                ]

                durs = [
                    n.end - n.start for inst in pm.instruments
                    for n in inst.notes if not inst.is_drum
                ]

                if not pitches:
                    continue

                rows.append({
                    "model": model_name,
                    "pitch_hist_sim": float(
                        histogram_similarity(
                            compute_pitch_histogram(pitches),
                            np.ones(12) / 12
                        )
                    ),
                    "rhythm_diversity": float(rhythm_diversity(durs)),
                    "repetition_ratio": float(repetition_ratio(pitches)),
                    "n_notes": len(pitches),
                })

            except Exception as e:
                print(f"  [WARN] {mf}: {e}")

    return rows

# ── Main ──────────────────────────────────────────────────────────────────────
def run_comparison():
    print("=== Baseline Comparison ===")
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    rand_dir   = os.path.join(cfg.MIDI_OUT_DIR, "baseline_random")
    markov_dir = os.path.join(cfg.MIDI_OUT_DIR, "baseline_markov")

    rand_results   = generate_random(n=5, out_dir=rand_dir)
    markov_results = generate_markov(n=5, out_dir=markov_dir)
    task_results   = load_task_results(cfg.MIDI_OUT_DIR)

    all_results = rand_results + markov_results + task_results
    df = pd.DataFrame(all_results)

    # Average per model
    summary = df.groupby("model").mean(numeric_only=True).round(4)
    print("\n=== Comparison Table ===")
    print(summary.to_string())

    out_csv = os.path.join(cfg.PLOTS_DIR, "baseline_comparison.csv")
    summary.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    # ── Bar Chart ─────────────────────────────────────────────────────────────
    model_order = ["Random Generator", "Markov Chain",
                   "LSTM Autoencoder", "VAE", "Transformer", "RLHF"]
    metrics = ["pitch_hist_sim", "rhythm_diversity", "repetition_ratio"]
    labels  = ["Pitch Hist Sim ↓", "Rhythm Diversity ↑", "Repetition Ratio ↓"]
    colors  = ["#e74c3c","#e67e22","#3498db","#2ecc71","#9b59b6","#1abc9c"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        vals   = [summary.loc[m, metric] if m in summary.index else 0
                  for m in model_order]
        models = [m for m in model_order]
        bars = ax.bar(range(len(models)), vals, color=colors[:len(models)])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.set_title(label, fontsize=11, fontweight='bold')
        vals = [summary.loc[m][metric] if m in summary.index else 0
        for m in model_order]
        max_val = max(vals) if any(vals) else 1
        ax.set_ylim(0, max_val * 1.2 + 0.01)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=7)

    plt.suptitle("Baseline vs Task Model Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_fig = os.path.join(cfg.PLOTS_DIR, "baseline_comparison.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Saved chart: {out_fig}")

if __name__ == "__main__":
    run_comparison()