"""
latent_interpolation.py - Interpolate between two points in VAE latent space
Generates a smooth transition between two genres.
Run: python src/evaluation/latent_interpolation.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from models.vae import MusicVAE
from generation.midi_export import piano_roll_to_midi

def interpolate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicVAE(input_dim=88, hidden_dim=cfg.VAE_HIDDEN_DIM,
                     latent_dim=cfg.VAE_LATENT_DIM, seq_len=64,
                     n_genres=len(cfg.GENRES)).to(device)
    ckpt = os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "vae_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    print("Performing latent space interpolation between 2 random points...")
    steps = 8
    with torch.no_grad():
        z1 = torch.randn(1, cfg.VAE_LATENT_DIM, device=device)
        z2 = torch.randn(1, cfg.VAE_LATENT_DIM, device=device)

        # Linear interpolation
        alphas = torch.linspace(0, 1, steps, device=device)
        rolls = []
        for a in alphas:
            z = (1 - a) * z1 + a * z2
            roll = model.decode(z).cpu().numpy()[0]  # (T, 88)
            rolls.append(roll.T)                      # (88, T)

    # Visualize all interpolation steps as piano rolls
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()
    for i, roll in enumerate(rolls):
        axes[i].imshow(roll, aspect='auto', origin='lower',
                       cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f"Step {i+1}/{steps} (a={i/(steps-1):.2f})", fontsize=9)
        axes[i].set_xlabel("Time"); axes[i].set_ylabel("Pitch")
    plt.suptitle("VAE Latent Space Interpolation (z1 → z2)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(cfg.PLOTS_DIR, "latent_interpolation.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved interpolation plot: {out}")

    # Save interpolated MIDI files
    interp_dir = os.path.join(cfg.MIDI_OUT_DIR, "interpolation")
    os.makedirs(interp_dir, exist_ok=True)
    for i, roll in enumerate(rolls):
        midi_path = os.path.join(interp_dir, f"interp_step_{i+1:02d}.mid")
        piano_roll_to_midi(roll, midi_path)
        print(f"  Saved: {midi_path}")
    print("Interpolation complete!")

if __name__ == "__main__":
    interpolate()
