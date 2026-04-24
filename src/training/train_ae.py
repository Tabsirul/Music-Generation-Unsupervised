"""
train_ae.py — Task 1 Training Loop: LSTM Autoencoder
Usage: python src/training/train_ae.py --epochs 50 --genre classical
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.autoencoder import LSTMAutoencoder
import config as cfg


def load_data(split_dir: str, genre: str = None):
    X_tr  = np.load(os.path.join(split_dir, "X_train.npy"))
    X_val = np.load(os.path.join(split_dir, "X_val.npy"))
    y_tr  = np.load(os.path.join(split_dir, "y_train.npy"))
    y_val = np.load(os.path.join(split_dir, "y_val.npy"))

    if genre is not None:
        gid = cfg.GENRE2IDX.get(genre, 0)
        X_tr  = X_tr[y_tr   == gid]
        X_val = X_val[y_val == gid]

    # reshape: (B, 88, T) -> (B, T, 88)
    X_tr  = X_tr.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    tr_ds  = TensorDataset(torch.FloatTensor(X_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_val))
    return tr_ds, val_ds


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr_ds, val_ds = load_data(cfg.SPLIT_DIR, args.genre)
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    seq_len   = tr_ds[0][0].shape[0]    # T
    model     = LSTMAutoencoder(input_dim=88, hidden_dim=cfg.AE_HIDDEN_DIM,
                                latent_dim=cfg.AE_LATENT_DIM, seq_len=seq_len,
                                num_layers=cfg.AE_NUM_LAYERS, dropout=cfg.AE_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    os.makedirs(os.path.join(cfg.OUTPUTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    tr_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for (x,) in tr_dl:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = LSTMAutoencoder.reconstruction_loss(x, x_hat)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_dl:
                x = x.to(device)
                x_hat, _ = model(x)
                val_loss += LSTMAutoencoder.reconstruction_loss(x, x_hat).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch:3d}/{args.epochs}]  Train: {tr_loss:.4f}  Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "ae_best.pt")
            torch.save(model.state_dict(), ckpt)

    # ── Plot loss curve ────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(tr_losses,  label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Task 1 — LSTM Autoencoder Reconstruction Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, "ae_loss_curve.png"), dpi=150)
    plt.close()
    print(f"Loss curve saved to {cfg.PLOTS_DIR}/ae_loss_curve.png")
    print(f"Best checkpoint: {ckpt}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=cfg.AE_EPOCHS)
    ap.add_argument("--batch_size", type=int,   default=cfg.AE_BATCH_SIZE)
    ap.add_argument("--lr",         type=float, default=cfg.AE_LR)
    ap.add_argument("--genre",      type=str,   default="classical")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
