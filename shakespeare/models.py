from __future__ import annotations
import torch
from torch import nn


class CharTransformer(nn.Module):
    """Lightweight Transformer for next-character prediction.
    ~295K params with default settings. Processes all 80 positions
    in parallel — saturates GPU cores via batched self-attention."""
    def __init__(self, vocab_size: int = 80, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 256, seq_len: int = 80):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h)
        return self.fc(h[:, -1, :])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
