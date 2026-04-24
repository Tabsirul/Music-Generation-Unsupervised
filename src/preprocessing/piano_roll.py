"""
piano_roll.py — Converts processed MIDI events into 88-key piano-roll matrices
                and saves train/val/test numpy splits.

Piano-roll shape: (batch, 88, time_steps)  — binary or velocity-weighted
"""
import os, json, glob, argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

STEPS_PER_BAR = 16
STEPS_PER_BEAT = 4
NUM_PITCHES   = 88   # MIDI 21–108  (piano range)
MIDI_MIN      = 21


def events_to_piano_roll(events: list, steps_per_bar: int = STEPS_PER_BAR,
                          n_bars: int = 16, tempo: float = 120.0) -> np.ndarray:
    """
    Convert note events to a piano-roll matrix.
    Returns array of shape (NUM_PITCHES, steps_per_bar * n_bars).
    """
    total_steps = steps_per_bar * n_bars
    seconds_per_step = 60.0 / tempo / (steps_per_bar / 4)
    roll = np.zeros((NUM_PITCHES, total_steps), dtype=np.float32)

    for e in events:
        pitch_idx = e["pitch"] - MIDI_MIN
        if not (0 <= pitch_idx < NUM_PITCHES):
            continue
        start_step = int(e["start"] / seconds_per_step)
        end_step   = int(e["end"]   / seconds_per_step)
        start_step = min(start_step, total_steps - 1)
        end_step   = min(end_step,   total_steps)
        vel_norm   = e["velocity"] / 127.0
        roll[pitch_idx, start_step:end_step] = vel_norm

    return roll


def process_to_rolls(processed_dir: str, out_dir: str,
                     n_bars: int = 8, steps_per_bar: int = STEPS_PER_BAR):
    os.makedirs(out_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(processed_dir, "*.json"))
    rolls, genres = [], []

    for jf in tqdm(json_files, desc="Piano-roll"):
        with open(jf) as f:
            d = json.load(f)
        from src.preprocessing.tokenizer import GENRE2IDX
        genre_id = GENRE2IDX.get(d["genre"], 0)
        roll = events_to_piano_roll(d["events"], steps_per_bar, n_bars)
        # Slide window over the roll
        win = steps_per_bar * n_bars
        for s in range(0, roll.shape[1] - win, win // 2):
            rolls.append(roll[:, s:s+win])
            genres.append(genre_id)

    X = np.array(rolls,  dtype=np.float32)
    y = np.array(genres, dtype=np.int32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    np.save(os.path.join(out_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_te)
    np.save(os.path.join(out_dir, "y_train.npy"), y_tr)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_te)

    print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")
    print(f"Roll shape: {X_tr[0].shape}  (pitches x time_steps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  default="data/processed")
    ap.add_argument("--out_dir", default="data/train_test_split")
    ap.add_argument("--n_bars",  type=int, default=8)
    a = ap.parse_args()
    process_to_rolls(a.in_dir, a.out_dir, a.n_bars)
