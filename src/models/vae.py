"""
vae.py — Task 2: Variational Autoencoder for multi-genre music generation.

Architecture:
    Encoder → (µ, log σ²)  →  reparameterisation  →  z
    Decoder: z → reconstructed sequence

Loss:
    L_VAE = L_recon + β * D_KL(q_φ(z|X) || p(z))
    where D_KL = -½ Σ(1 + log σ² - µ² - σ²)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm  = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_top = h[-1]               # (B, hidden_dim)
        mu     = self.fc_mu(h_top)
        logvar = self.fc_logvar(h_top)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len,
                 num_layers=2, dropout=0.3, n_genres=5):
        super().__init__()
        self.seq_len    = seq_len
        self.genre_emb  = nn.Embedding(n_genres, 16)
        self.fc_in      = nn.Linear(latent_dim + 16, hidden_dim)
        self.lstm       = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out     = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, genre_ids=None):
        B = z.size(0)
        if genre_ids is not None:
            g = self.genre_emb(genre_ids)   # (B, 16)
            z = torch.cat([z, g], dim=-1)   # (B, latent+16)
        else:
            pad = torch.zeros(B, 16, device=z.device)
            z = torch.cat([z, pad], dim=-1)
        x = self.fc_in(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc_out(out)             # (B, T, output_dim)


class MusicVAE(nn.Module):
    """Task 2: β-VAE for multi-genre music generation."""

    def __init__(self, input_dim=88, hidden_dim=512, latent_dim=256,
                 seq_len=256, num_layers=2, dropout=0.3, beta=1.0, n_genres=5):
        super().__init__()
        self.encoder  = VAEEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder  = VAEDecoder(latent_dim, hidden_dim, input_dim, seq_len,
                                   num_layers, dropout, n_genres)
        self.latent_dim = latent_dim
        self.beta       = beta

    def reparameterise(self, mu, logvar):
        """z = µ + σ ⊙ ε,   ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, genre_ids=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z, genre_ids)
        return x_hat, mu, logvar, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, genre_ids=None):
        return self.decoder(z, genre_ids)

    def sample(self, n_samples, genre_id=None, device="cpu"):
        """Sample n_samples from the prior N(0,I)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        gids = None
        if genre_id is not None:
            gids = torch.full((n_samples,), genre_id, dtype=torch.long, device=device)
        return self.decode(z, gids)

    def loss(self, x, x_hat, mu, logvar):
        """
        L_VAE = L_recon + β * D_KL
        D_KL  = -½ Σ(1 + log σ² - µ² - σ²)
        """
        L_recon = F.mse_loss(x_hat, x, reduction="mean")
        D_KL    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return L_recon + self.beta * D_KL, L_recon, D_KL

    def interpolate(self, x1, x2, steps=10, genre_id=None, device="cpu"):
        """Linearly interpolate between two encoded sequences in latent space."""
        mu1, _ = self.encoder(x1.unsqueeze(0).to(device))
        mu2, _ = self.encoder(x2.unsqueeze(0).to(device))
        alphas = torch.linspace(0, 1, steps, device=device)
        results = []
        gids = None
        if genre_id is not None:
            gids = torch.zeros(1, dtype=torch.long, device=device).fill_(genre_id)
        for a in alphas:
            z = (1 - a) * mu1 + a * mu2
            results.append(self.decode(z, gids))
        return torch.cat(results, dim=0)
