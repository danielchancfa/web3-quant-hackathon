"""
Data preparation utilities for next-hour OHLC forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import get_config
from modeling.data_processing import compute_daily_features, load_ohlcv

EPSILON = 1e-8


def _load_hourly_bars(db_path: Path, pair: str) -> pd.DataFrame:
    """
    Load hourly OHLCV data for a pair, resampling 1m data if needed.
    """
    hourly = load_ohlcv(db_path, pair, interval="1h")
    if hourly.empty:
        minute = load_ohlcv(db_path, pair, interval="1m")
        if minute.empty:
            raise ValueError(f"No OHLCV data available for {pair} at 1h or 1m interval.")
        minute = minute.dropna(subset=["open", "high", "low", "close"])
        minute = minute.set_index("datetime")
        hourly = (
            minute.resample("1H")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        hourly["interval"] = "1h"
        hourly["pair"] = pair
    hourly["datetime"] = pd.to_datetime(hourly["datetime"], utc=True)
    hourly = hourly.sort_values("datetime").reset_index(drop=True)
    return hourly


def _compute_hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    df = df.set_index("datetime")
    df["return"] = df["close"].pct_change().fillna(0.0)
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    df["volatility"] = (df["high"] - df["low"]) / (df["close"].abs() + EPSILON)
    df["volume_norm"] = (df["volume"] - df["volume"].rolling(48).mean()) / (
        df["volume"].rolling(48).std() + EPSILON
    )
    df["range"] = (df["high"] - df["low"]).fillna(0.0)
    df["body"] = (df["close"] - df["open"]).fillna(0.0)

    for window in (3, 6, 12, 24, 48):
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()

    df["return_6h"] = df["close"].pct_change(periods=6)
    df["return_24h"] = df["close"].pct_change(periods=24)
    df["volatility_24"] = df["return"].rolling(24).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan) + EPSILON)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["atr_14"] = df["return"].rolling(14).std().fillna(0.0)
    df["kdj_k"] = 0.0
    df["kdj_d"] = 0.0
    df["kdj_j"] = 0.0
    df["supertrend"] = 0.0
    df["supertrend_dir"] = 0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df.reset_index()


def _attach_daily_context(db_path: Path, hourly_features: pd.DataFrame, pair: str) -> pd.DataFrame:
    daily = compute_daily_features(db_path, pair=pair, label_mode="binary")
    daily.index = pd.to_datetime(daily.index)
    context_cols = [
        "return",
        "volatility",
        "volatility_30",
        "rsi_14",
        "macd",
        "macd_signal",
        "kdj_k",
        "kdj_d",
        "supertrend_dir",
        "regime_label",
        "fear_greed_value",
    ]
    missing = [c for c in context_cols if c not in daily.columns]
    for col in missing:
        daily[col] = 0.0
    daily_context = daily[context_cols].copy()
    daily_context.columns = [f"daily_{c}" for c in context_cols]

    hourly = hourly_features.copy()
    hourly["date"] = hourly["datetime"].dt.floor("D")
    joined = hourly.merge(
        daily_context.reset_index().rename(columns={"index": "date"}),
        on="date",
        how="left",
    )
    joined = joined.drop(columns=["date"])

    # Forward/back-fill first, then hard defaults for stubborn columns.
    joined = joined.fillna(method="ffill").fillna(method="bfill")

    daily_cols = [c for c in joined.columns if c.startswith("daily_")]
    for col in daily_cols:
        series = joined[col]
        if series.isna().all():
            if col.endswith("fear_greed_value"):
                joined[col] = 50.0
            elif col.endswith("regime_label"):
                joined[col] = 1.0
            else:
                joined[col] = 0.0
        else:
            joined[col] = series.fillna(method="ffill").fillna(method="bfill")
    return joined


def build_feature_table(db_path: Path, pair: str, dropna_targets: bool = True) -> pd.DataFrame:
    hourly = _load_hourly_bars(db_path, pair)
    hourly_features = _compute_hourly_features(hourly)
    hourly_features = _attach_daily_context(db_path, hourly_features, pair)

    hourly_features["next_close"] = hourly_features["close"].shift(-1)
    hourly_features["next_open"] = hourly_features["open"].shift(-1)
    hourly_features["next_high"] = hourly_features["high"].shift(-1)
    hourly_features["next_low"] = hourly_features["low"].shift(-1)
    hourly_features["next_volume"] = hourly_features["volume"].shift(-1)
    ref_close = hourly_features["close"].abs() + EPSILON
    hourly_features["target_delta_close"] = (hourly_features["next_close"] - hourly_features["close"]) / ref_close
    vol_est = hourly_features["return"].rolling(24).std().fillna(hourly_features["return"].std())
    vol_est = vol_est.replace(0.0, vol_est.mean() + EPSILON)
    hourly_features["target_confidence"] = np.clip(
        np.abs(hourly_features["target_delta_close"]) / (vol_est + EPSILON),
        0.0,
        1.0,
    )
    hourly_features["target_delta_open"] = (hourly_features["next_open"] - hourly_features["close"]) / ref_close
    hourly_features["target_delta_high"] = (hourly_features["next_high"] - hourly_features["close"]) / ref_close
    hourly_features["target_delta_low"] = (hourly_features["next_low"] - hourly_features["close"]) / ref_close
    with np.errstate(divide="ignore"):
        hourly_features["target_delta_volume"] = np.log1p(hourly_features["next_volume"]) - np.log1p(hourly_features["volume"].replace(0, np.nan))
    hourly_features["target_delta_volume"] = hourly_features["target_delta_volume"].fillna(0.0)
    if dropna_targets:
        hourly_features = hourly_features.dropna().reset_index(drop=True)
    else:
        hourly_features = hourly_features.dropna(subset=_default_feature_columns(hourly_features)).reset_index(drop=True)
    return hourly_features


def _default_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "log_return",
        "return_6h",
        "return_24h",
        "volatility",
        "volatility_24",
        "volume_norm",
        "range",
        "body",
        "sma_3",
        "sma_6",
        "sma_12",
        "sma_24",
        "sma_48",
        "ema_3",
        "ema_6",
        "ema_12",
        "ema_24",
        "ema_48",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
    ]
    cols.extend(
        [
            "daily_return",
            "daily_volatility",
            "daily_volatility_30",
            "daily_rsi_14",
            "daily_macd",
            "daily_macd_signal",
            "daily_kdj_k",
            "daily_kdj_d",
            "daily_supertrend_dir",
            "daily_regime_label",
            "daily_fear_greed_value",
        ]
    )
    return [c for c in cols if c in df.columns]


TARGET_COLUMNS = [
    "target_delta_close",
    "target_confidence",
    "target_delta_open",
    "target_delta_high",
    "target_delta_low",
    "target_delta_volume",
]

META_COLUMNS = [
    "datetime",
    "open",
    "high",
    "low",
    "volume",
    "close",
    "next_open",
    "next_high",
    "next_low",
    "next_volume",
    "next_close",
]


@dataclass
class DatasetSplit:
    dataset: "NextBarDataset"
    indices: Sequence[int]


class NextBarDataset(Dataset):
    """
    Sequence dataset that yields normalized feature windows and next-bar targets.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        sequence_length: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if len(table) <= sequence_length:
            raise ValueError("Not enough rows to build sequences.")

        self.table = table.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        self.sequence_length = sequence_length

        features = self.table[self.feature_columns].values.astype(np.float32)

        if mean is None or std is None:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0) + EPSILON
        else:
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)
        self.features = (features - self.mean) / self.std

        self.targets = self.table[self.target_columns].values.astype(np.float32)
        self.meta = self.table[META_COLUMNS].reset_index(drop=True)

        lower = max(sequence_length, 0 if start is None else start)
        upper = len(self.table) if end is None else min(len(self.table), end)
        if lower >= upper:
            raise ValueError("Invalid start/end for dataset slice.")
        self.indices = np.arange(lower, upper, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_pos = self.indices[idx]
        start = target_pos - self.sequence_length
        end = target_pos
        window = self.features[start:end]
        target = self.targets[target_pos]
        return torch.from_numpy(window), torch.from_numpy(target)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_columns)

    def get_meta(self, idx: int) -> pd.Series:
        target_pos = self.indices[idx]
        return self.meta.iloc[target_pos]


def prepare_datasets(
    db_path: Path,
    pair: str,
    sequence_length: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, List[str]]:
    config = get_config()
    db_path = Path(db_path or config.data_cache_path or "data_cache/trading_data.db")

    table = build_feature_table(db_path, pair)
    feature_columns = _default_feature_columns(table)
    if not feature_columns:
        raise ValueError("Feature column list is empty.")

    total_rows = len(table)
    train_cut = max(sequence_length + 1, int(total_rows * train_ratio))
    val_cut = max(train_cut + 1, int(total_rows * (train_ratio + val_ratio)))
    val_cut = min(val_cut, total_rows - 1)

    train_mean = table.iloc[:train_cut][feature_columns].mean(axis=0).values.astype(np.float32)
    train_std = table.iloc[:train_cut][feature_columns].std(axis=0).values.astype(np.float32) + EPSILON

    train_dataset = NextBarDataset(
        table,
        feature_columns,
        TARGET_COLUMNS,
        sequence_length=sequence_length,
        mean=train_mean,
        std=train_std,
        start=None,
        end=train_cut,
    )
    val_dataset = NextBarDataset(
        table,
        feature_columns,
        TARGET_COLUMNS,
        sequence_length=sequence_length,
        mean=train_dataset.mean,
        std=train_dataset.std,
        start=train_cut,
        end=val_cut,
    )
    test_dataset = NextBarDataset(
        table,
        feature_columns,
        TARGET_COLUMNS,
        sequence_length=sequence_length,
        mean=train_dataset.mean,
        std=train_dataset.std,
        start=val_cut,
        end=None,
    )

    return (
        DatasetSplit(train_dataset, train_dataset.indices),
        DatasetSplit(val_dataset, val_dataset.indices),
        DatasetSplit(test_dataset, test_dataset.indices),
        feature_columns,
    )


def sequence_collate_fn(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    windows, targets = zip(*batch)
    return torch.stack(windows, dim=0), torch.stack(targets, dim=0)

