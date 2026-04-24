"""
train_transformer.py — Task 3 Training Loop: Transformer Music Generator
Usage: python src/training/train_transformer.py --epochs 100 --n_layers 6
"""
import os, sys, argparse, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import MusicTransformer
import config as cfg


def load_token_data(split_dir):
    seqs   = np.load(os.path.join(split_dir, "sequences.npy"))
    genres = np.load(os.path.join(split_dir, "genres.npy"))
    n = len(seqs)
    split = int(0.9 * n)
    tr  = TensorDataset(torch.LongTensor(seqs[:split]),   torch.LongTensor(genres[:split]))
    val = TensorDataset(torch.LongTensor(seqs[split:]),   torch.LongTensor(genres[split:]))
    return tr, val


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr_ds, val_ds = load_token_data(cfg.SPLIT_DIR)
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = MusicTransformer(vocab_size=cfg.VOCAB_SIZE, d_model=args.d_model,
                             n_heads=args.n_heads, n_layers=args.n_layers,
                             d_ff=cfg.D_FF, dropout=cfg.TF_DROPOUT,
                             max_seq_len=cfg.MAX_SEQ_LEN,
                             n_genres=len(cfg.GENRES)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = len(tr_dl) * args.epochs
    warmup = min(4000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(os.path.join(cfg.OUTPUTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    tr_losses, val_losses, perplexities = [], [], []
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train(); ep_loss = 0
        for tokens, genres in tr_dl:
            tokens, genres = tokens.to(device), genres.to(device)
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            logits = model(inp, genres)
            loss = MusicTransformer.compute_loss(logits, tgt)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item(); global_step += 1
        ep_loss /= len(tr_dl)

        model.eval(); v_loss = 0
        with torch.no_grad():
            for tokens, genres in val_dl:
                tokens, genres = tokens.to(device), genres.to(device)
                inp, tgt = tokens[:, :-1], tokens[:, 1:]
                logits = model(inp, genres)
                v_loss += MusicTransformer.compute_loss(logits, tgt).item()
        v_loss /= len(val_dl)
        ppl = MusicTransformer.perplexity(v_loss)

        tr_losses.append(ep_loss); val_losses.append(v_loss); perplexities.append(ppl)
        print(f"[{epoch:3d}/{args.epochs}] Train NLL={ep_loss:.4f}  Val NLL={v_loss:.4f}  PPL={ppl:.2f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "transformer_best.pt"))

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(tr_losses, label="Train"); ax1.plot(val_losses, label="Val")
    ax1.set_title("Transformer NLL Loss"); ax1.legend()
    ax2.plot(perplexities, color="orange")
    ax2.set_title("Validation Perplexity")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, "transformer_loss_curve.png"), dpi=150)
    print(f"Best val PPL: {MusicTransformer.perplexity(best_val):.2f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=cfg.TF_EPOCHS)
    ap.add_argument("--batch_size", type=int,   default=cfg.TF_BATCH_SIZE)
    ap.add_argument("--lr",         type=float, default=cfg.TF_LR)
    ap.add_argument("--n_layers",   type=int,   default=cfg.N_LAYERS)
    ap.add_argument("--d_model",    type=int,   default=cfg.D_MODEL)
    ap.add_argument("--n_heads",    type=int,   default=cfg.N_HEADS)
    return ap.parse_args()

if __name__ == "__main__":
    train(parse_args())
