"""
train_vae.py — Task 2 Training Loop: VAE Multi-Genre Generator
Usage: python src/training/train_vae.py --epochs 80 --beta 1.0
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vae import MusicVAE
import config as cfg


def load_data(split_dir):
    X_tr  = np.load(os.path.join(split_dir, "X_train.npy")).transpose(0, 2, 1)
    X_val = np.load(os.path.join(split_dir, "X_val.npy")).transpose(0, 2, 1)
    y_tr  = np.load(os.path.join(split_dir, "y_train.npy"))
    y_val = np.load(os.path.join(split_dir, "y_val.npy"))
    tr  = TensorDataset(torch.FloatTensor(X_tr),  torch.LongTensor(y_tr))
    val = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    return tr, val


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  β={args.beta}")

    tr_ds, val_ds = load_data(cfg.SPLIT_DIR)
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    seq_len = tr_ds[0][0].shape[0]
    model = MusicVAE(input_dim=88, hidden_dim=cfg.VAE_HIDDEN_DIM,
                     latent_dim=cfg.VAE_LATENT_DIM, seq_len=seq_len,
                     num_layers=cfg.VAE_NUM_LAYERS, dropout=cfg.VAE_DROPOUT,
                     beta=args.beta, n_genres=len(cfg.GENRES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.join(cfg.OUTPUTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    tr_total, tr_recon, tr_kl = [], [], []
    val_total = []
    best_val  = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_tot, ep_rec, ep_kl = 0, 0, 0
        for x, g in tr_dl:
            x, g = x.to(device), g.to(device)
            x_hat, mu, logvar, _ = model(x, g)
            total, recon, kl = model.loss(x, x_hat, mu, logvar)
            optimizer.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_tot += total.item(); ep_rec += recon.item(); ep_kl += kl.item()
        n = len(tr_dl)
        tr_total.append(ep_tot/n); tr_recon.append(ep_rec/n); tr_kl.append(ep_kl/n)

        model.eval(); v_tot = 0
        with torch.no_grad():
            for x, g in val_dl:
                x, g = x.to(device), g.to(device)
                x_hat, mu, logvar, _ = model(x, g)
                tot, _, _ = model.loss(x, x_hat, mu, logvar)
                v_tot += tot.item()
        v_tot /= len(val_dl); val_total.append(v_tot)

        print(f"[{epoch:3d}/{args.epochs}] Total={ep_tot/n:.4f} "
              f"Recon={ep_rec/n:.4f} KL={ep_kl/n:.4f} Val={v_tot:.4f}")

        if v_tot < best_val:
            best_val = v_tot
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "vae_best.pt"))

    # Loss plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tr_total, label="Train"); axes[0].plot(val_total, label="Val")
    axes[0].set_title("Total VAE Loss"); axes[0].legend()
    axes[1].plot(tr_recon, label="Recon"); axes[1].plot(tr_kl, label="KL")
    axes[1].set_title("Recon vs KL"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, "vae_loss_curve.png"), dpi=150)
    print(f"Loss curves saved. Best val: {best_val:.4f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=cfg.VAE_EPOCHS)
    ap.add_argument("--batch_size", type=int,   default=cfg.VAE_BATCH_SIZE)
    ap.add_argument("--lr",         type=float, default=cfg.VAE_LR)
    ap.add_argument("--beta",       type=float, default=cfg.VAE_BETA)
    return ap.parse_args()

if __name__ == "__main__":
    train(parse_args())
