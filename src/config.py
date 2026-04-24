# =============================================================================
# config.py — Central configuration for all tasks
# =============================================================================
import os

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
RAW_MIDI_DIR   = os.path.join(DATA_DIR, "raw_midi")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
SPLIT_DIR      = os.path.join(DATA_DIR, "train_test_split")
OUTPUTS_DIR    = "outputs"
MIDI_OUT_DIR   = os.path.join(OUTPUTS_DIR, "generated_midis")
PLOTS_DIR      = os.path.join(OUTPUTS_DIR, "plots")
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")

# ─── MIDI / Piano-Roll ───────────────────────────────────────────────────────
STEPS_PER_BAR  = 16
NUM_PITCHES    = 88
SEQUENCE_LENGTH= 256
VELOCITY_BINS  = 32

# ─── Genre ───────────────────────────────────────────────────────────────────
GENRES    = ["classical", "jazz", "rock", "pop", "electronic"]
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}
GENRE_OFFSET = 323   # must match tokenizer.py

# ─── Task 1 LSTM AE ──────────────────────────────────────────────────────────
AE_HIDDEN_DIM  = 512
AE_LATENT_DIM  = 128
AE_NUM_LAYERS  = 2
AE_DROPOUT     = 0.3
AE_LR          = 1e-3
AE_EPOCHS      = 50
AE_BATCH_SIZE  = 32

# ─── Task 2 VAE ───────────────────────────────────────────────────────────────
VAE_HIDDEN_DIM = 512
VAE_LATENT_DIM = 256
VAE_NUM_LAYERS = 2
VAE_DROPOUT    = 0.3
VAE_LR         = 1e-3
VAE_EPOCHS     = 80
VAE_BATCH_SIZE = 32
VAE_BETA       = 1.0

# ─── Task 3 Transformer ───────────────────────────────────────────────────────
VOCAB_SIZE     = 512
D_MODEL        = 256
N_HEADS        = 8
N_LAYERS       = 6
D_FF           = 1024
TF_DROPOUT     = 0.1
TF_LR          = 1e-4
TF_EPOCHS      = 100
TF_BATCH_SIZE  = 16
MAX_SEQ_LEN    = 1024

# ─── Task 4 RLHF ─────────────────────────────────────────────────────────────
RL_STEPS       = 500
RL_LR          = 5e-6
RL_SAMPLE_SIZE = 4

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_N_SAMPLES = 100
