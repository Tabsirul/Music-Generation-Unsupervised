"""
tokenizer.py — Converts note-event dicts into integer token sequences and back.

Token vocabulary layout (VOCAB_SIZE = 512):
  0          : PAD
  1          : BOS (begin of sequence)
  2          : EOS (end of sequence)
  3-130      : NOTE_ON  for pitches 0-127
  131-258    : NOTE_OFF for pitches 0-127
  259-290    : TIME_SHIFT (32 quantised time bins)
  291-322    : VELOCITY  (32 quantised velocity bins)
  323-327    : GENRE tokens (classical/jazz/rock/pop/electronic)
"""
import os, json, numpy as np
from typing import List, Dict

PAD, BOS, EOS = 0, 1, 2
NOTE_ON_OFFSET  = 3
NOTE_OFF_OFFSET = 131
TIME_OFFSET     = 259
VEL_OFFSET      = 291
GENRE_OFFSET    = 323
N_TIME_BINS     = 32
N_VEL_BINS      = 32
MAX_TIME_SHIFT  = 2.0   # seconds

GENRES = ["classical", "jazz", "rock", "pop", "electronic"]
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}


def quantise_time(dt: float) -> int:
    dt = max(0.0, min(dt, MAX_TIME_SHIFT))
    return int(dt / MAX_TIME_SHIFT * (N_TIME_BINS - 1))


def quantise_velocity(v: int) -> int:
    return min(int(v / 128 * N_VEL_BINS), N_VEL_BINS - 1)


def events_to_tokens(events: List[Dict], genre: str = "classical") -> List[int]:
    tokens = [BOS, GENRE_OFFSET + GENRE2IDX.get(genre, 0)]
    prev_time = 0.0
    for e in events:
        dt = e["start"] - prev_time
        if dt > 0:
            tokens.append(TIME_OFFSET + quantise_time(dt))
        tokens.append(VEL_OFFSET + quantise_velocity(e["velocity"]))
        tokens.append(NOTE_ON_OFFSET + e["pitch"])
        dur_bin = quantise_time(e["duration"])
        tokens.append(TIME_OFFSET + dur_bin)
        tokens.append(NOTE_OFF_OFFSET + e["pitch"])
        prev_time = e["start"]
    tokens.append(EOS)
    return tokens


def tokens_to_events(tokens: List[int]) -> List[Dict]:
    events, current_time = [], 0.0
    open_notes = {}   # pitch -> (start_time, velocity)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == EOS:
            break
        elif NOTE_ON_OFFSET <= t < NOTE_OFF_OFFSET:
            pitch = t - NOTE_ON_OFFSET
            vel = 64
            if i > 0 and VEL_OFFSET <= tokens[i-1] < VEL_OFFSET + N_VEL_BINS:
                vel = int((tokens[i-1] - VEL_OFFSET) / N_VEL_BINS * 128)
            open_notes[pitch] = (current_time, vel)
        elif NOTE_OFF_OFFSET <= t < TIME_OFFSET:
            pitch = t - NOTE_OFF_OFFSET
            if pitch in open_notes:
                start, vel = open_notes.pop(pitch)
                events.append({"pitch": pitch, "velocity": vel,
                                "start": start, "end": current_time,
                                "duration": current_time - start})
        elif TIME_OFFSET <= t < VEL_OFFSET:
            bin_idx = t - TIME_OFFSET
            current_time += bin_idx / (N_TIME_BINS - 1) * MAX_TIME_SHIFT
        i += 1
    return sorted(events, key=lambda e: e["start"])


def tokenize_dataset(processed_dir: str, out_dir: str, seq_len: int = 256):
    os.makedirs(out_dir, exist_ok=True)
    import glob
    json_files = glob.glob(os.path.join(processed_dir, "*.json"))
    all_seqs, all_genres = [], []
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        tokens = events_to_tokens(d["events"], d["genre"])
        genre_id = GENRE2IDX.get(d["genre"], 0)
        for start in range(0, len(tokens) - seq_len, seq_len // 2):
            chunk = tokens[start:start + seq_len]
            if len(chunk) < seq_len:
                chunk += [PAD] * (seq_len - len(chunk))
            all_seqs.append(chunk)
            all_genres.append(genre_id)
    seqs   = np.array(all_seqs,   dtype=np.int32)
    genres = np.array(all_genres, dtype=np.int32)
    np.save(os.path.join(out_dir, "sequences.npy"), seqs)
    np.save(os.path.join(out_dir, "genres.npy"),    genres)
    print(f"Saved {len(seqs)} sequences of length {seq_len}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out_dir",       default="data/train_test_split")
    ap.add_argument("--seq_len",       type=int, default=256)
    a = ap.parse_args()
    tokenize_dataset(a.processed_dir, a.out_dir, a.seq_len)
