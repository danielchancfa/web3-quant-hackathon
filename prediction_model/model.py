"""
Model definition for next-hour close forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from modeling.transformer_trading import PositionalEncoding


@dataclass
class NextBarTransformerConfig:
    feature_dim: int
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    output_dim: int = 6  # delta close, confidence, delta open/high/low, delta volume


class NextBarTransformer(nn.Module):
    def __init__(self, config: NextBarTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.feature_dim, config.d_model)
        self.positional = PositionalEncoding(config.d_model, dropout=config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, sequence_length, feature_dim)
        """
        x = self.input_proj(x)
        x = self.positional(x)
        x = self.encoder(x)
        x = x[:, -1]
        return self.head(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

