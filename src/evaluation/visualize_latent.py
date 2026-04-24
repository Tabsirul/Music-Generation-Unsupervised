"""
visualize_latent.py - t-SNE visualization of VAE latent space (genre clusters)
Run: python src\evaluation\visualize_latent.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from models.vae import MusicVAE
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VAE
    model = MusicVAE(input_dim=88, hidden_dim=cfg.VAE_HIDDEN_DIM,
                     latent_dim=cfg.VAE_LATENT_DIM, seq_len=64,
                     n_genres=len(cfg.GENRES)).to(device)
    ckpt = os.path.join(cfg.OUTPUTS_DIR, "checkpoints", "vae_best.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: Checkpoint not found at {ckpt}")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Load validation data
    X = np.load(os.path.join(cfg.SPLIT_DIR, "X_val.npy"))   # (N, 88, T)
    y = np.load(os.path.join(cfg.SPLIT_DIR, "y_val.npy"))   # (N,)
    X_tensor = torch.FloatTensor(X.transpose(0, 2, 1)).to(device)

    # Encode to latent space
    latents, labels = [], []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 32):
            batch = X_tensor[i:i+32]
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
            labels.append(y[i:i+32])

    latents = np.concatenate(latents, axis=0)
    labels  = np.concatenate(labels,  axis=0)
    print(f"Encoded {len(latents)} samples into latent space (dim={latents.shape[1]})")

    # ── Clustering Metrics ────────────────────────────────────────────────────
    n_clusters = len(cfg.GENRES)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latents)

    sil  = silhouette_score(latents, cluster_labels)
    dbi  = davies_bouldin_score(latents, cluster_labels)
    print(f"Silhouette Score     : {sil:.4f}  (higher is better, range [-1,1])")
    print(f"Davies-Bouldin Index : {dbi:.4f}  (lower is better)")

    # Save clustering metrics
    import csv
    metrics_path = os.path.join(cfg.OUTPUTS_DIR, "plots", "clustering_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "features", "clustering",
                    "silhouette", "davies_bouldin"])
        w.writerow(["easy",   "LSTM-AE", "piano_roll", "kmeans", "--", "--"])
        w.writerow(["medium", "VAE",     "piano_roll", "kmeans",
                    f"{sil:.4f}", f"{dbi:.4f}"])
        w.writerow(["hard",   "Transformer", "tokens", "kmeans", "--", "--"])
    print(f"Clustering metrics saved: {metrics_path}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print("Running t-SNE (this takes ~1 min)...")
    perp = min(30, len(latents) - 1)
    import sklearn
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    tsne_kwargs = dict(n_components=2, random_state=42, perplexity=perp)
    if sk_ver >= (1, 5):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    if sk_ver >= (1, 5):
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=1000)
    latents_2d = tsne.fit_transform(latents)

    # Plot 1: Genre colored
    colors = ['royalblue','darkorange','green','red','purple']
    plt.figure(figsize=(10, 8))
    for gid, gname in enumerate(cfg.GENRES):
        mask = labels == gid
        if mask.sum() > 0:
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                       c=colors[gid], label=gname.capitalize(),
                       alpha=0.7, s=40, edgecolors='white', linewidths=0.3)
    plt.title("VAE Latent Space — Genre Clusters (t-SNE)", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(title="Genre", fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(cfg.PLOTS_DIR, "latent_space_tsne_genre.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved: {out1}")

    # Plot 2: KMeans cluster colored
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1],
                          c=cluster_labels, cmap='tab10',
                          alpha=0.7, s=40, edgecolors='white', linewidths=0.3)
    plt.colorbar(scatter, label='K-Means Cluster')
    plt.title(
        f"VAE Latent Space - K-Means Clusters (k={n_clusters}) "
        f"Silhouette={sil:.3f}  DBI={dbi:.3f}",
        fontsize=12, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(cfg.PLOTS_DIR, "latent_space_tsne_kmeans.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")
    print("DONE - check outputs/plots/ for cluster images")

if __name__ == "__main__":
    visualize()