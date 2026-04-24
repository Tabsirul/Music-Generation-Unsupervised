"""
midi_export.py — Convert model outputs (token sequences or piano rolls) to MIDI files.
"""
import os
import numpy as np

try:
    import pretty_midi
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False

# Token offsets (must match tokenizer.py)
NOTE_ON_OFFSET  = 3
NOTE_OFF_OFFSET = 131
TIME_OFFSET     = 259
VEL_OFFSET      = 291
N_TIME_BINS     = 32
N_VEL_BINS      = 32
MAX_TIME_SHIFT  = 2.0
MIDI_MIN        = 21


def tokens_to_midi(tokens, out_path: str, tempo: float = 120.0,
                   program: int = 0) -> str:
    """Convert integer token sequence → MIDI file."""
    if not HAS_MIDI:
        raise ImportError("pip install pretty_midi")

    pm       = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst     = pretty_midi.Instrument(program=program)
    cur_time = 0.0
    cur_vel  = 64
    open_notes = {}   # pitch -> (start_time, velocity)

    for t in tokens:
        if t == 2:   # EOS
            break
        elif VEL_OFFSET <= t < VEL_OFFSET + N_VEL_BINS:
            cur_vel = int((t - VEL_OFFSET) / N_VEL_BINS * 127)
        elif TIME_OFFSET <= t < VEL_OFFSET:
            bin_idx   = t - TIME_OFFSET
            cur_time += bin_idx / (N_TIME_BINS - 1) * MAX_TIME_SHIFT
        elif NOTE_ON_OFFSET <= t < NOTE_OFF_OFFSET:
            pitch = t - NOTE_ON_OFFSET
            open_notes[pitch] = (cur_time, cur_vel)
        elif NOTE_OFF_OFFSET <= t < TIME_OFFSET:
            pitch = t - NOTE_OFF_OFFSET
            if pitch in open_notes:
                start, vel = open_notes.pop(pitch)
                dur = max(cur_time - start, 0.05)
                note = pretty_midi.Note(velocity=vel, pitch=pitch,
                                        start=start, end=start + dur)
                inst.notes.append(note)

    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pm.write(out_path)
    return out_path


def piano_roll_to_midi(roll: np.ndarray, out_path: str,
                        tempo: float = 120.0, steps_per_bar: int = 16,
                        program: int = 0) -> str:
    """
    Convert (88, T) piano-roll numpy array → MIDI file.
    Non-zero cells indicate active notes; values represent velocity (0-1).
    """
    if not HAS_MIDI:
        raise ImportError("pip install pretty_midi")

    seconds_per_step = 60.0 / tempo / (steps_per_bar / 4)
    pm   = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program)

    for pitch_idx in range(roll.shape[0]):
        pitch   = pitch_idx + MIDI_MIN
        row     = roll[pitch_idx]
        in_note = False
        start   = 0.0
        vel     = 64

        for step, val in enumerate(row):
            t = step * seconds_per_step
            if val > 0 and not in_note:
                in_note = True
                start   = t
                vel     = min(int(val * 127), 127)
            elif (val == 0 or step == len(row) - 1) and in_note:
                in_note = False
                note = pretty_midi.Note(velocity=vel, pitch=pitch,
                                        start=start, end=t)
                inst.notes.append(note)

    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pm.write(out_path)
    return out_path
