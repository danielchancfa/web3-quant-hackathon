"""
Run inference with a trained next-hour close forecasting model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import pandas as pd

from config import get_config
from prediction_model.data import build_feature_table
from prediction_model.model import NextBarTransformer, NextBarTransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer next-hour close forecasts.")
    parser.add_argument("--pair", type=str, default="BTC/USD", help="Trading pair.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument("--db_path", type=Path, default=None, help="Override SQLite DB path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump JSON results.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[NextBarTransformer, Dict[str, object]]:
    data = torch.load(checkpoint_path, map_location=device)
    config = NextBarTransformerConfig(**data["model_config"])
    model = NextBarTransformer(config)
    model.load_state_dict(data["model_state_dict"])
    model.to(device)
    model.eval()
    metadata = data.get("metadata", {})
    return model, metadata


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = args.db_path or getattr(config, "sqlite_path", None) or "data_cache/trading_data.db"
    db_path = Path(db_path)
    device = torch.device(args.device)

    model, metadata = load_model(args.checkpoint, device)
    if not metadata:
        raise RuntimeError("Checkpoint is missing metadata required for inference.")

    sequence_length = int(metadata["sequence_length"])
    feature_columns = metadata["feature_columns"]
    feature_mean = torch.tensor(metadata["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.tensor(metadata["feature_std"], dtype=torch.float32, device=device)

    table = build_feature_table(db_path, args.pair, dropna_targets=False)
    feature_columns = [c for c in feature_columns if c in table.columns]
    if len(feature_columns) != len(metadata["feature_columns"]):
        raise RuntimeError("Feature column mismatch between checkpoint and current table.")

    window = table.tail(sequence_length).reset_index(drop=True)
    inputs = torch.tensor(window[feature_columns].values, dtype=torch.float32, device=device)
    inputs = (inputs - feature_mean) / feature_std
    inputs = inputs.unsqueeze(0)  # batch dimension

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
        "pair": args.pair,
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

    # Include actual next values if available (training context)
    actual = {}
    if not pd.isna(latest.get("next_close")):
        actual["close"] = float(latest["next_close"])
    if not pd.isna(latest.get("next_open")):
        actual["open"] = float(latest["next_open"])
    if not pd.isna(latest.get("next_high")):
        actual["high"] = float(latest["next_high"])
    if not pd.isna(latest.get("next_low")):
        actual["low"] = float(latest["next_low"])
    if "next_volume" in latest and not pd.isna(latest["next_volume"]):
        actual["volume"] = float(latest["next_volume"])
    if actual:
        result["actual_next"] = actual

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fp:
            json.dump(result, fp, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

