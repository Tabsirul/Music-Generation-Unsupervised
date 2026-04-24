"""
generate_music.py — Unified entry-point for all four task models.

Usage:
    python src/generation/generate_music.py --model autoencoder --n_samples 5  --out outputs/generated_midis/task1/
    python src/generation/generate_music.py --model vae         --n_samples 8  --out outputs/generated_midis/task2/
    python src/generation/generate_music.py --model transformer --n_samples 10 --out outputs/generated_midis/task3/
    python src/generation/generate_music.py --model rlhf        --n_samples 10 --out outputs/generated_midis/task4/
"""
import os, sys, argparse, random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from generation.midi_export import tokens_to_midi, piano_roll_to_midi
from generation.sample_latent import sample_prior


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_transformer_dims(state_dict):
    """
    Read d_model, n_heads, n_layers, d_ff, vocab_size, n_genres directly from
    a checkpoint's state_dict so we never get a size-mismatch error even if
    config.py was edited after training.
    """
    # d_model / vocab_size  <- tok_emb weight: (vocab_size, d_model)
    tok_w      = state_dict["tok_emb.weight"]
    vocab_size = tok_w.shape[0]
    d_model    = tok_w.shape[1]

    # n_genres  <- genre_emb weight: (n_genres, d_model)
    n_genres = state_dict["genre_emb.weight"].shape[0]

    # n_layers  <- count unique layer indices in transformer.layers.*
    layer_ids = set()
    for k in state_dict:
        if k.startswith("transformer.layers."):
            idx = k.split(".")[2]
            if idx.isdigit():
                layer_ids.add(int(idx))
    n_layers = len(layer_ids) if layer_ids else cfg.N_LAYERS

    # d_ff  <- first linear1.weight found: (d_ff, d_model)
    d_ff = cfg.D_FF
    for k, v in state_dict.items():
        if "linear1.weight" in k:
            d_ff = v.shape[0]
            break

    # n_heads: use config default, but ensure it divides d_model
    n_heads = cfg.N_HEADS
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads //= 2

    return dict(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, n_genres=n_genres)


def _load_transformer(ckpt_path, device):
    """Load a MusicTransformer whose architecture is inferred from the checkpoint."""
    from models.transformer import MusicTransformer

    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}  — using random weights")
        return MusicTransformer(
            vocab_size=cfg.VOCAB_SIZE, d_model=cfg.D_MODEL,
            n_heads=cfg.N_HEADS, n_layers=cfg.N_LAYERS,
            d_ff=cfg.D_FF, n_genres=len(cfg.GENRES)
        ).to(device)

    sd   = torch.load(ckpt_path, map_location=device)
    dims = _infer_transformer_dims(sd)

    print(f"  Checkpoint architecture detected:")
    print(f"    vocab_size={dims['vocab_size']}  d_model={dims['d_model']}  "
          f"n_heads={dims['n_heads']}  n_layers={dims['n_layers']}  "
          f"d_ff={dims['d_ff']}  n_genres={dims['n_genres']}")

    model = MusicTransformer(
        vocab_size  = dims["vocab_size"],
        d_model     = dims["d_model"],
        n_heads     = dims["n_heads"],
        n_layers    = dims["n_layers"],
        d_ff        = dims["d_ff"],
        dropout     = 0.0,
        max_seq_len = cfg.MAX_SEQ_LEN,
        n_genres    = dims["n_genres"],
    ).to(device)

    model.load_state_dict(sd)
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


# ── Generators ────────────────────────────────────────────────────────────────

def generate_autoencoder(args, device):
    from models.autoencoder import LSTMAutoencoder
    model = LSTMAutoencoder(input_dim=88, hidden_dim=cfg.AE_HIDDEN_DIM,
                            latent_dim=cfg.AE_LATENT_DIM, seq_len=128).to(device)
    ckpt = os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "ae_best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded {ckpt}")
    model.eval()
    with torch.no_grad():
        z     = sample_prior(args.n_samples, cfg.AE_LATENT_DIM, device)
        rolls = model.decode(z).cpu().numpy()   # (N, T, 88)
    for i, roll in enumerate(rolls):
        out = os.path.join(args.out, f"ae_sample_{i+1:02d}.mid")
        piano_roll_to_midi(roll.T, out)
        print(f"  Generated: {out}")


def generate_vae(args, device):
    from models.vae import MusicVAE
    model = MusicVAE(input_dim=88, hidden_dim=cfg.VAE_HIDDEN_DIM,
                     latent_dim=cfg.VAE_LATENT_DIM, seq_len=128,
                     n_genres=len(cfg.GENRES)).to(device)
    ckpt = os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "vae_best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded {ckpt}")
    model.eval()
    genres = list(cfg.GENRE2IDX.values())
    with torch.no_grad():
        for i in range(args.n_samples):
            gid  = genres[i % len(genres)]
            roll = model.sample(1, genre_id=gid, device=device).cpu().numpy()[0]
            out  = os.path.join(args.out, f"vae_{cfg.GENRES[gid]}_sample_{i+1:02d}.mid")
            piano_roll_to_midi(roll.T, out)
            print(f"  Generated: {out}")


def generate_transformer(args, device, model_key="transformer"):
    ckpt_name = "rlhf_tuned.pt" if model_key == "rlhf" else "transformer_best.pt"
    ckpt      = os.path.join(cfg.OUTPUTS_DIR, "checkpoints", ckpt_name)

    model = _load_transformer(ckpt, device)
    model.eval()

    for i in range(args.n_samples):
        gid    = random.randint(0, len(cfg.GENRES) - 1)
        prompt = torch.tensor([[1, cfg.GENRE_OFFSET + gid]], device=device)
        tokens = model.generate(
            prompt,
            max_new_tokens = args.max_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            genre_id       = gid,
            device         = device,
        )
        out = os.path.join(args.out,
                           f"{model_key}_{cfg.GENRES[gid]}_sample_{i+1:02d}.mid")
        tokens_to_midi(tokens[0].cpu().tolist(), out)
        print(f"  Generated: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    print(f"Generating {args.n_samples} samples  model='{args.model}'  device={device}")
    print(f"Output dir: {args.out}")

    if   args.model == "autoencoder": generate_autoencoder(args, device)
    elif args.model == "vae":         generate_vae(args, device)
    elif args.model == "transformer": generate_transformer(args, device, "transformer")
    elif args.model == "rlhf":        generate_transformer(args, device, "rlhf")
    else:
        print(f"Unknown model '{args.model}'. "
              f"Choose from: autoencoder, vae, transformer, rlhf")
    print("Done.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="vae",
                    choices=["autoencoder", "vae", "transformer", "rlhf"])
    ap.add_argument("--n_samples",   type=int,   default=5)
    ap.add_argument("--out",         default="outputs/generated_midis/")
    ap.add_argument("--max_tokens",  type=int,   default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k",       type=int,   default=50)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())