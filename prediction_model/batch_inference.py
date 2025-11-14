"""
Batch inference utility for next-hour close forecasts.

Loads next-bar checkpoints per pair, generates forecasts, and optionally
persists them into the SQLite database alongside JSON output.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import pandas as pd

from config import get_config
from prediction_model.data import build_feature_table
from prediction_model.model import NextBarTransformer, NextBarTransformerConfig

DEFAULT_CHECKPOINT = Path("model_checkpoints/next_bar")
FORECAST_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS next_bar_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    as_of TEXT NOT NULL,
    reference_close REAL NOT NULL,
    pred_open REAL,
    pred_high REAL,
    pred_low REAL,
    pred_close REAL NOT NULL,
    pred_volume REAL,
    pred_delta_close REAL NOT NULL,
    pred_delta_open REAL,
    pred_delta_high REAL,
    pred_delta_low REAL,
    pred_delta_volume REAL,
    confidence REAL NOT NULL,
    actual_open REAL,
    actual_high REAL,
    actual_low REAL,
    actual_close REAL,
    actual_volume REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(pair, as_of)
);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-bar forecasts for multiple pairs.")
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Comma-separated list of pairs. If omitted, all checkpoint subdirectories are used.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Directory containing per-pair checkpoints.",
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=None,
        help="Override SQLite database path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write forecast results.",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Skip persisting forecasts into the SQLite database.",
    )
    return parser.parse_args()


def discover_pairs(checkpoint_dir: Path) -> List[str]:
    pairs: List[str] = []
    if not checkpoint_dir.exists():
        return pairs
    for path in sorted(checkpoint_dir.iterdir()):
        if path.is_dir() and (path / "next_bar_transformer.pt").exists():
            pair = path.name.replace("_", "/")
            pairs.append(pair)
    return pairs


def ensure_table(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(FORECAST_TABLE_SQL)
        conn.commit()


def load_model(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    data = torch.load(checkpoint_path, map_location=device)
    cfg = NextBarTransformerConfig(**data["model_config"])
    model = NextBarTransformer(cfg).to(device)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    metadata = data["metadata"]
    return {"model": model, "config": cfg, "metadata": metadata}


def generate_forecast(
    pair: str,
    table_path: Path,
    checkpoint: Dict[str, object],
    device: torch.device,
) -> Optional[Dict[str, object]]:
    table = build_feature_table(table_path, pair, dropna_targets=False)
    metadata = checkpoint["metadata"]
    sequence_length = int(metadata["sequence_length"])
    feature_columns = metadata["feature_columns"]

    if len(table) < sequence_length:
        print(f"[WARN] Not enough rows for {pair} (required {sequence_length}, found {len(table)}). Skipping.")
        return None

    window = table.tail(sequence_length).reset_index(drop=True)
    inputs = torch.tensor(window[feature_columns].values, dtype=torch.float32, device=device)

    mean = torch.tensor(metadata["feature_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(metadata["feature_std"], dtype=torch.float32, device=device)
    inputs = (inputs - mean) / std
    inputs = inputs.unsqueeze(0)

    model: NextBarTransformer = checkpoint["model"]
    with torch.no_grad():
        outputs = model(inputs)
    delta_close = float(outputs[:, 0].item())
    confidence = torch.sigmoid(outputs[:, 1]).item()
    delta_open = float(outputs[:, 2].item())
    delta_high = float(outputs[:, 3].item())
    delta_low = float(outputs[:, 4].item())
    delta_volume = float(outputs[:, 5].item())

    latest = table.iloc[-1]
    ref_close = float(latest["close"])
    pred_close = ref_close * (1.0 + delta_close)
    pred_open = ref_close * (1.0 + delta_open)
    pred_high = ref_close * (1.0 + delta_high)
    pred_low = ref_close * (1.0 + delta_low)
    curr_volume = float(latest.get("volume", 0.0))
    pred_volume = float(np.expm1(np.log1p(max(curr_volume, 0.0)) + delta_volume))
    pred_volume = max(pred_volume, 0.0)

    result = {
        "pair": pair,
        "as_of": str(latest["datetime"]),
        "reference_close": ref_close,
        "predicted": {
            "close": pred_close,
            "open": pred_open,
            "high": pred_high,
            "low": pred_low,
            "volume": pred_volume,
            "delta_close": delta_close,
            "delta_open": delta_open,
            "delta_high": delta_high,
            "delta_low": delta_low,
            "delta_volume": delta_volume,
            "confidence": confidence,
        },
    }

    actual = {}
    for key in ("next_open", "next_high", "next_low", "next_close", "next_volume"):
        val = latest.get(key)
        if not pd.isna(val):
            actual[key.replace("next_", "")] = float(val)
    if actual:
        result["actual"] = actual
    return result


def store_forecasts(db_path: Path, forecasts: List[Dict[str, object]]) -> None:
    if not forecasts:
        return
    ensure_table(db_path)

    with sqlite3.connect(db_path) as conn:
        for entry in forecasts:
            pred = entry["predicted"]
            actual = entry.get("actual", {})
            conn.execute(
                """
                INSERT INTO next_bar_forecasts (
                    pair, as_of, reference_close,
                    pred_open, pred_high, pred_low, pred_close, pred_volume,
                    pred_delta_close, pred_delta_open, pred_delta_high, pred_delta_low, pred_delta_volume,
                    confidence,
                    actual_open, actual_high, actual_low, actual_close, actual_volume,
                    created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now')
                )
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
                    actual_open = excluded.actual_open,
                    actual_high = excluded.actual_high,
                    actual_low = excluded.actual_low,
                    actual_close = excluded.actual_close,
                    actual_volume = excluded.actual_volume,
                    updated_at = datetime('now');
                """,
                (
                    entry["pair"],
                    entry["as_of"],
                    entry["reference_close"],
                    pred.get("open"),
                    pred.get("high"),
                    pred.get("low"),
                    pred["close"],
                    pred.get("volume"),
                    pred["delta_close"],
                    pred.get("delta_open"),
                    pred.get("delta_high"),
                    pred.get("delta_low"),
                    pred.get("delta_volume"),
                    pred["confidence"],
                    actual.get("open"),
                    actual.get("high"),
                    actual.get("low"),
                    actual.get("close"),
                    actual.get("volume"),
                ),
            )
        conn.commit()


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = args.db_path or getattr(config, "sqlite_path", None) or Path("data_cache/trading_data.db")
    db_path = Path(db_path)
    device = torch.device(args.device)

    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    else:
        pairs = discover_pairs(args.checkpoint_dir)

    if not pairs:
        raise RuntimeError("No trading pairs found. Provide --pairs or ensure checkpoints exist.")

    results: List[Dict[str, object]] = []
    for pair in pairs:
        ckpt_dir = args.checkpoint_dir / pair.replace("/", "_")
        checkpoint_path = ckpt_dir / "next_bar_transformer.pt"
        if not checkpoint_path.exists():
            print(f"[WARN] Checkpoint not found for {pair} at {checkpoint_path}")
            continue
        checkpoint = load_model(checkpoint_path, device)
        forecast = generate_forecast(pair, db_path, checkpoint, device)
        if forecast:
            results.append(forecast)

    if not args.no_store and results:
        store_forecasts(db_path, results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fp:
            json.dump(results, fp, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

