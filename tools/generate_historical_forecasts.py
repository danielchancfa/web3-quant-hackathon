"""
Generate historical forecasts by replaying the model across a range of timestamps.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import get_config
from prediction_model.data import build_feature_table
from prediction_model.model import NextBarTransformer, NextBarTransformerConfig


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    data = torch.load(checkpoint_path, map_location=device)
    cfg = NextBarTransformerConfig(**data["model_config"])
    model = NextBarTransformer(cfg).to(device)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    metadata = data["metadata"]
    return {"model": model, "config": cfg, "metadata": metadata}


def prepare_pair_table(db_path: Path, pair: str) -> pd.DataFrame:
    df = build_feature_table(db_path, pair, dropna_targets=False)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def infer_windows(
    pair_table: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    window: int,
    feature_columns: List[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    windows = []
    for ts in timestamps:
        mask = pair_table["datetime"] <= ts
        sliced = pair_table.loc[mask].tail(window)
        if len(sliced) < window:
            continue
        arr = sliced[feature_columns].to_numpy(dtype=np.float32)
        arr = (arr - mean) / (std + 1e-8)
        windows.append((ts, arr))

    if not windows:
        return np.array([], dtype=np.float32), []

    times, arrays = zip(*windows)
    return np.stack(arrays), list(times)


def upsert_forecasts(
    conn: sqlite3.Connection,
    pair: str,
    times: List[pd.Timestamp],
    ref_closes: List[float],
    outputs: torch.Tensor,
    feature_table: pd.DataFrame,
) -> None:
    delta_close = outputs[:, 0].cpu().numpy()
    confidence = torch.sigmoid(outputs[:, 1]).cpu().numpy()
    delta_open = outputs[:, 2].cpu().numpy()
    delta_high = outputs[:, 3].cpu().numpy()
    delta_low = outputs[:, 4].cpu().numpy()
    delta_volume = outputs[:, 5].cpu().numpy()

    rows = []
    for idx, ts in enumerate(times):
        ref_close = float(ref_closes[idx])
        delta_close_val = float(delta_close[idx])
        delta_open_val = float(delta_open[idx])
        delta_high_val = float(delta_high[idx])
        delta_low_val = float(delta_low[idx])
        delta_volume_val = float(delta_volume[idx])
        confidence_val = float(confidence[idx])
        
        rows.append(
            (
                pair,
                ts.isoformat(),
                ref_close,
                ref_close * (1.0 + delta_open_val),
                ref_close * (1.0 + delta_high_val),
                ref_close * (1.0 + delta_low_val),
                ref_close * (1.0 + delta_close_val),
                float(np.expm1(delta_volume_val)),
                delta_close_val,
                delta_open_val,
                delta_high_val,
                delta_low_val,
                delta_volume_val,
                confidence_val,
            )
        )

    conn.executemany(
        """
        INSERT INTO next_bar_forecasts (
            pair, as_of, reference_close,
            pred_open, pred_high, pred_low, pred_close, pred_volume,
            pred_delta_close, pred_delta_open, pred_delta_high, pred_delta_low, pred_delta_volume,
            confidence,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(pair, as_of) DO UPDATE SET
            reference_close = excluded.reference_close,
            pred_open = excluded.pred_open,
            pred_high = excluded.pred_high,
            pred_low = excluded.pred_low,
            pred_close = excluded.pred_close,
            pred_volume = excluded.pred_volume,
            pred_delta_close = excluded.pred_delta_close,
            pred_delta_open = excluded.pred_delta_open,
            pred_delta_high = excluded.pred_delta_high,
            pred_delta_low = excluded.pred_delta_low,
            pred_delta_volume = excluded.pred_delta_volume,
            confidence = excluded.confidence,
            updated_at = datetime('now');
        """,
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate historical forecasts over a time range.")
    parser.add_argument("--db_path", type=Path, default=None, help="Path to SQLite database.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True, help="Directory containing per-pair checkpoints.")
    parser.add_argument("--pairs", type=str, default="", help="Comma-separated list of pairs (default: checkpoint dirs).")
    parser.add_argument("--start", type=str, required=True, help="Start timestamp (ISO 8601).")
    parser.add_argument("--end", type=str, required=True, help="End timestamp (ISO 8601).")
    parser.add_argument("--freq_hours", type=int, default=1, help="Forecast frequency in hours.")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length expected by the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = get_config()
    db_path = args.db_path or Path(config.db_path)
    device = torch.device(args.device)

    checkpoint_dir = args.checkpoint_dir
    available_pairs = [
        path.name.replace("_", "/")
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and (path / "next_bar_transformer.pt").exists()
    ]
    if not available_pairs:
        raise RuntimeError("No checkpoints found in the specified directory.")

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()] if args.pairs else available_pairs
    start_ts = pd.to_datetime(args.start).tz_localize("UTC")
    end_ts = pd.to_datetime(args.end).tz_localize("UTC")
    timestamps = pd.date_range(start_ts, end_ts, freq=f"{args.freq_hours}H")

    with sqlite3.connect(db_path) as conn:
        conn.execute("BEGIN")
        for pair in pairs:
            ckpt_path = checkpoint_dir / pair.replace("/", "_") / "next_bar_transformer.pt"
            if not ckpt_path.exists():
                print(f"[WARN] Checkpoint not found for {pair}, skipping.")
                continue

            bundle = load_checkpoint(ckpt_path, device)
            table = prepare_pair_table(db_path, pair)
            table_filtered = table[(table["datetime"] >= start_ts) & (table["datetime"] <= end_ts)]
            if table_filtered.empty:
                continue

            feature_columns = bundle["metadata"]["feature_columns"]
            mean = np.array(bundle["metadata"]["feature_mean"], dtype=np.float32)
            std = np.array(bundle["metadata"]["feature_std"], dtype=np.float32)

            windows, times = infer_windows(table, timestamps, args.sequence_length, feature_columns, mean, std)
            if windows.size == 0:
                continue

            tensor = torch.tensor(windows, dtype=torch.float32, device=device)
            with torch.no_grad():
                outputs = bundle["model"](tensor)

            ref_closes = [
                float(table[table["datetime"] <= ts]["close"].iloc[-1])
                for ts in times
            ]
            upsert_forecasts(conn, pair, times, ref_closes, outputs, table)
        conn.commit()

    print("Historical forecasts generated and stored.")


if __name__ == "__main__":
    main()

