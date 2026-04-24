"""
train_rlhf.py — Task 4: RLHF Policy Gradient Fine-Tuning
Uses a pre-trained Transformer generator and optimises it toward human reward.

Objective:  max_θ E[r(X_gen)]
Gradient:   ∇_θ J(θ) = E[r · ∇_θ log p_θ(X)]

Usage:
    python src/training/train_rlhf.py \
        --pretrained_ckpt outputs/checkpoints/transformer_best.pt \
        --rl_steps 500
"""
import os, sys, argparse, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import MusicTransformer
from evaluation.metrics import pitch_diversity_score, rhythm_diversity_score
import config as cfg


# ─── Reward Model ─────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Lightweight reward model trained on human listening scores.
    Input: token sequence (B, T)
    Output: scalar reward in [0, 1]
    """
    def __init__(self, vocab_size=512, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, tokens):
        x = self.emb(tokens)
        out = self.encoder(x)
        return self.head(out.mean(dim=1)).squeeze(-1)   # (B,)


def heuristic_reward(tokens: torch.Tensor) -> torch.Tensor:
    """
    Rule-based proxy reward (used before human data is collected).
    Combines pitch diversity and rhythm diversity scores.
    Score in [0, 1].
    """
    rewards = []
    for seq in tokens.cpu().numpy():
        pitches = [(t - 3) for t in seq if 3 <= t < 131]   # NOTE_ON tokens
        durs    = [(t - 259) for t in seq if 259 <= t < 291]  # TIME tokens
        pd = len(set(pitches)) / max(len(pitches), 1)
        rd = len(set(durs)) / max(len(durs), 1)
        reward = 0.5 * pd + 0.5 * rd
        rewards.append(float(reward))
    return torch.tensor(rewards, dtype=torch.float32)


# ─── RLHF Training Loop ───────────────────────────────────────────────────────

def rl_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  RL steps: {args.rl_steps}")

    # Load pretrained generator — infer architecture from checkpoint to avoid
    # size-mismatch errors when config.py differs from what was used at train time
    if os.path.exists(args.pretrained_ckpt):
        sd = torch.load(args.pretrained_ckpt, map_location=device)
        tok_w      = sd["tok_emb.weight"]
        vocab_size = tok_w.shape[0]
        d_model    = tok_w.shape[1]
        n_genres   = sd["genre_emb.weight"].shape[0]
        layer_ids  = {int(k.split(".")[2]) for k in sd
                      if k.startswith("transformer.layers.") and k.split(".")[2].isdigit()}
        n_layers   = len(layer_ids) if layer_ids else cfg.N_LAYERS
        d_ff       = next((v.shape[0] for k, v in sd.items() if "linear1.weight" in k), cfg.D_FF)
        n_heads    = cfg.N_HEADS
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads //= 2
        print(f"Checkpoint: d_model={d_model} n_heads={n_heads} "
              f"n_layers={n_layers} d_ff={d_ff} vocab={vocab_size}")
        gen = MusicTransformer(vocab_size=vocab_size, d_model=d_model,
                               n_heads=n_heads, n_layers=n_layers,
                               d_ff=d_ff, n_genres=n_genres).to(device)
        gen.load_state_dict(sd)
        print(f"Loaded pretrained weights from {args.pretrained_ckpt}")
    else:
        print("[WARN] Pretrained checkpoint not found — training from scratch.")
        gen = MusicTransformer(vocab_size=cfg.VOCAB_SIZE, d_model=cfg.D_MODEL,
                               n_heads=cfg.N_HEADS, n_layers=cfg.N_LAYERS,
                               d_ff=cfg.D_FF, n_genres=len(cfg.GENRES)).to(device)

    optimizer = torch.optim.Adam(gen.parameters(), lr=args.rl_lr)

    os.makedirs(os.path.join(cfg.OUTPUTS_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    reward_history, loss_history = [], []

    for step in range(1, args.rl_steps + 1):
        gen.train()
        # Sample a batch of generated sequences
        genre_id = random.randint(0, len(cfg.GENRES) - 1)
        prompt = torch.tensor([[1, cfg.GENRE_OFFSET + genre_id]] * cfg.RL_SAMPLE_SIZE,
                               device=device)  # BOS + genre token

        # Forward pass to get log-probabilities (teacher-forced on generated tokens)
        with torch.no_grad():
            generated = gen.generate(prompt[0:1], max_new_tokens=127,
                                     genre_id=genre_id, device=device)
            # replicate for batch
            batch_tokens = generated.repeat(cfg.RL_SAMPLE_SIZE, 1)

        # Re-compute log-probs with grad
        inp, tgt = batch_tokens[:, :-1], batch_tokens[:, 1:]
        logits = gen(inp)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected  = log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T)
        seq_log_probs = selected.mean(dim=1)   # (B,)

        # Compute reward (heuristic or reward model)
        with torch.no_grad():
            rewards = heuristic_reward(batch_tokens).to(device)

        # Normalise rewards (baseline subtraction)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Policy gradient loss: -E[r * log p]
        pg_loss = -(rewards * seq_log_probs).mean()
        optimizer.zero_grad(); pg_loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        optimizer.step()

        reward_history.append(rewards.mean().item())
        loss_history.append(pg_loss.item())

        if step % 50 == 0 or step == 1:
            print(f"[Step {step:4d}/{args.rl_steps}] "
                  f"PG Loss={pg_loss.item():.4f}  Mean Reward={rewards.mean().item():.4f}")

    torch.save(gen.state_dict(),
               os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "rlhf_tuned.pt"))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(reward_history, color="green"); ax1.set_title("Mean Reward per Step")
    ax2.plot(loss_history, color="red");    ax2.set_title("Policy Gradient Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, "rlhf_training_curve.png"), dpi=150)
    print("RLHF training complete.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", default=os.path.join(cfg.OUTPUTS_DIR,
                                                               "checkpoints", "transformer_best.pt"))
    ap.add_argument("--rl_steps", type=int,   default=cfg.RL_STEPS)
    ap.add_argument("--rl_lr",    type=float, default=cfg.RL_LR)
    return ap.parse_args()

if __name__ == "__main__":
    rl_finetune(parse_args())