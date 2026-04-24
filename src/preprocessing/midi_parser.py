"""
midi_parser.py — Converts raw MIDI files into structured note-event JSON lists.
Usage: python src/preprocessing/midi_parser.py --data_dir data/raw_midi --out_dir data/processed
"""
import os, json, argparse
from pathlib import Path
from tqdm import tqdm

try:
    import pretty_midi
except ImportError:
    raise ImportError("pip install pretty_midi")


def midi_to_events(midi_path: str) -> list:
    """Parse one MIDI file -> list of note-event dicts."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"  [WARN] {midi_path}: {e}")
        return []
    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        name = pretty_midi.program_to_instrument_name(inst.program)
        for n in inst.notes:
            events.append({"pitch": n.pitch, "velocity": n.velocity,
                           "start": round(n.start, 4), "end": round(n.end, 4),
                           "duration": round(n.end - n.start, 4), "instrument": name})
    events.sort(key=lambda e: e["start"])
    return events


def infer_genre(path: str) -> str:
    p = path.lower()
    for kw, g in [("maestro","classical"),("classical","classical"),
                  ("groove","jazz"),("jazz","jazz"),("rock","rock"),
                  ("pop","pop"),("electronic","electronic"),("techno","electronic")]:
        if kw in p:
            return g
    return "unknown"


def process_dataset(data_dir: str, out_dir: str, max_files=None):
    os.makedirs(out_dir, exist_ok=True)
    files = list(Path(data_dir).rglob("*.mid")) + list(Path(data_dir).rglob("*.midi"))
    if max_files:
        files = files[:max_files]
    print(f"Found {len(files)} MIDI files")
    meta = []
    for p in tqdm(files, desc="Parsing"):
        events = midi_to_events(str(p))
        if not events:
            continue
        genre = infer_genre(str(p))
        out = os.path.join(out_dir, f"{p.stem}__{genre}.json")
        with open(out, "w") as f:
            json.dump({"genre": genre, "events": events, "source": str(p)}, f)
        meta.append({"file": out, "genre": genre, "n_events": len(events)})
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Processed {len(meta)} files -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="data/raw_midi")
    ap.add_argument("--out_dir",   default="data/processed")
    ap.add_argument("--max_files", type=int, default=None)
    a = ap.parse_args()
    process_dataset(a.data_dir, a.out_dir, a.max_files)
