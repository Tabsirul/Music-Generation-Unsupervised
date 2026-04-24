"""
transformer.py — Task 3: Causal Transformer decoder for long-sequence music generation.

Architecture:
    Token embedding + positional encoding + genre embedding
    → N x (Masked Multi-Head Self-Attention + FFN)
    → Linear projection → vocabulary logits

Training loss (cross-entropy / NLL):
    L_TR = -Σ log p_θ(x_t | x_{<t})

Perplexity:
    PPL = exp(L_TR / T)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MusicTransformer(nn.Module):
    """
    Causal (autoregressive) Transformer decoder for token-based music generation.

    p(X) = Π p(x_t | x_{<t})
    """

    def __init__(self, vocab_size: int = 512, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1,
                 max_seq_len: int = 1024, n_genres: int = 5):
        super().__init__()
        self.d_model     = d_model
        self.vocab_size  = vocab_size

        self.tok_emb    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.genre_emb  = nn.Embedding(n_genres, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_seq_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_out      = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len: int, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, tokens, genre_ids=None):
        """
        tokens    : (B, T)  integer token ids
        genre_ids : (B,)    optional genre condition
        Returns   : logits (B, T, vocab_size)
        """
        B, T = tokens.shape
        device = tokens.device
        # Embeddings
        x = self.tok_emb(tokens) * math.sqrt(self.d_model)   # (B, T, D)
        if genre_ids is not None:
            g = self.genre_emb(genre_ids).unsqueeze(1)        # (B, 1, D)
            x = x + g
        x = self.pos_enc(x)

        # Causal self-attention (decoder without cross-attention memory)
        causal = self._causal_mask(T, device)
        # TransformerDecoder needs memory; pass x as its own memory (decoder-only trick)
        out = self.transformer(tgt=x, memory=x, tgt_mask=causal, memory_mask=causal)
        logits = self.fc_out(out)                              # (B, T, vocab_size)
        return logits

    @staticmethod
    def compute_loss(logits, targets):
        """NLL loss: -Σ log p(x_t | x_{<t})"""
        B, T, V = logits.shape
        return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T),
                               ignore_index=0, reduction="mean")

    @staticmethod
    def perplexity(loss_value: float) -> float:
        return math.exp(loss_value)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 512,
                 temperature: float = 1.0, top_k: int = 50,
                 genre_id: int = None, device: str = "cpu"):
        """
        Autoregressive generation: x_t ~ p_θ(x_t | x_{<t})
        prompt : (1, T_prompt) starting token ids
        """
        self.eval()
        generated = prompt.to(device)
        gids = None
        if genre_id is not None:
            gids = torch.tensor([genre_id], device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, gids)   # (1, T, V)
            next_logits = logits[:, -1, :] / temperature  # (1, V)
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_tok], dim=1)
            if next_tok.item() == 2:  # EOS
                break

        return generated
