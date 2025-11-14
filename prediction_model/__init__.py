"""
Next-bar prediction package for forecasting hourly OHLC and confidence.
"""

from .data import (
    NextBarDataset,
    build_feature_table,
    prepare_datasets,
    sequence_collate_fn,
)
from .model import NextBarTransformer, NextBarTransformerConfig

__all__ = [
    "NextBarDataset",
    "build_feature_table",
    "prepare_datasets",
    "sequence_collate_fn",
    "NextBarTransformer",
    "NextBarTransformerConfig",
]

