"""
autoencoder.py — Task 1: LSTM Autoencoder for single-genre music generation.

Architecture:
    Encoder: LSTM stack  →  dense projection  →  latent z
    Decoder: dense expand  →  LSTM stack  →  linear output

Loss:
    L_AE = (1/T) Σ ||x_t - x̂_t||²
"""
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, T, input_dim)
        out, (h, _) = self.lstm(x)
        # use last hidden state of top layer
        z = self.fc(h[-1])          # (B, latent_dim)
        return z, out               # also return full output for skip connections


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 seq_len: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.seq_len  = seq_len
        self.fc_in    = nn.Linear(latent_dim, hidden_dim)
        self.lstm     = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out   = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc_in(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        out, _ = self.lstm(x)
        return self.fc_out(out)   # (B, T, output_dim)


class LSTMAutoencoder(nn.Module):
    """Full LSTM Autoencoder for Task 1."""

    def __init__(self, input_dim: int = 88, hidden_dim: int = 512,
                 latent_dim: int = 128, seq_len: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        """
        x : (B, T, 88)
        Returns: x_hat (B, T, 88),  z (B, latent_dim)
        """
        z, _ = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        z, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def reconstruction_loss(x, x_hat):
        """MSE reconstruction loss: L_AE = Σ||x_t - x̂_t||²"""
        return nn.functional.mse_loss(x_hat, x, reduction="mean")
