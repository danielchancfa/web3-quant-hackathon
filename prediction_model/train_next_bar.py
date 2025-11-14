"""
Train a transformer to predict the next hourly close and a confidence score.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config
from prediction_model.data import (
    TARGET_COLUMNS,
    prepare_datasets,
    sequence_collate_fn,
)
from prediction_model.model import NextBarTransformer, NextBarTransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train next-hour close forecasting transformer.")
    parser.add_argument("--pair", type=str, default="BTC/USD", help="Trading pair to train on.")
    parser.add_argument("--sequence_length", type=int, default=128, help="Number of past hours per sample.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--confidence_weight", type=float, default=0.5, help="Loss weight for confidence head.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("model_checkpoints/next_bar"),
        help="Directory to store checkpoints.",
    )
    parser.add_argument("--db_path", type=Path, default=None, help="Override path to SQLite trading database.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Chronological train ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Chronological validation ratio.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional path to resume model weights.")
    return parser.parse_args()


def build_dataloaders(
    pair: str,
    sequence_length: int,
    batch_size: int,
    db_path: Path,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    train_split, val_split, test_split, feature_columns = prepare_datasets(
        db_path=db_path,
        pair=pair,
        sequence_length=sequence_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    loaders = (
        DataLoader(train_split.dataset, batch_size=batch_size, shuffle=True, collate_fn=sequence_collate_fn),
        DataLoader(val_split.dataset, batch_size=batch_size, shuffle=False, collate_fn=sequence_collate_fn),
        DataLoader(test_split.dataset, batch_size=batch_size, shuffle=False, collate_fn=sequence_collate_fn),
    )

    metadata = {
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
        "target_columns": list(TARGET_COLUMNS),
        "train_indices": list(map(int, train_split.indices)),
        "val_indices": list(map(int, val_split.indices)),
        "test_indices": list(map(int, test_split.indices)),
        "feature_mean": train_split.dataset.mean.tolist(),
        "feature_std": train_split.dataset.std.tolist(),
    }
    return (*loaders, metadata)


def evaluate(
    model: NextBarTransformer,
    loader: DataLoader,
    price_loss_fn: nn.Module,
    confidence_loss_fn: nn.Module,
    device: torch.device,
    confidence_weight: float,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_price = 0.0
    total_conf = 0.0
    total_aux = 0.0
    total_batches = 0

    with torch.no_grad():
        for windows, targets in loader:
            windows = windows.to(device)
            targets = targets.to(device)
            outputs = model(windows)
            pred_delta_close = outputs[:, 0]
            pred_conf = torch.sigmoid(outputs[:, 1])
            pred_aux = outputs[:, 2:]

            target_delta_close = targets[:, 0]
            target_conf = targets[:, 1]
            target_aux = targets[:, 2:]

            price_loss = price_loss_fn(pred_delta_close, target_delta_close)
            aux_loss = price_loss_fn(pred_aux, target_aux)
            conf_loss = confidence_loss_fn(pred_conf, target_conf)
            loss = price_loss + aux_loss + confidence_weight * conf_loss

            total_loss += loss.item()
            total_price += price_loss.item()
            total_conf += conf_loss.item()
            total_aux += aux_loss.item()
            total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / total_batches,
        total_price / total_batches,
        total_conf / total_batches,
        total_aux / total_batches,
    )


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = args.db_path or getattr(config, "sqlite_path", None) or "data_cache/trading_data.db"
    db_path = Path(db_path)
    device = torch.device(args.device)

    torch.manual_seed(42)

    train_loader, val_loader, test_loader, metadata = build_dataloaders(
        pair=args.pair,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        db_path=db_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    model_config = NextBarTransformerConfig(feature_dim=len(metadata["feature_columns"]))
    model = NextBarTransformer(model_config).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    price_loss_fn = nn.SmoothL1Loss()
    confidence_loss_fn = nn.MSELoss()

    best_val_loss = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_price = 0.0
        running_conf = 0.0
        running_aux = 0.0
        batches = 0

        for windows, targets in train_loader:
            windows = windows.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(windows)

            pred_delta_close = outputs[:, 0]
            pred_conf = torch.sigmoid(outputs[:, 1])
            pred_aux = outputs[:, 2:]
            target_delta_close = targets[:, 0]
            target_conf = targets[:, 1]
            target_aux = targets[:, 2:]

            price_loss = price_loss_fn(pred_delta_close, target_delta_close)
            aux_loss = price_loss_fn(pred_aux, target_aux)
            conf_loss = confidence_loss_fn(pred_conf, target_conf)
            loss = price_loss + aux_loss + args.confidence_weight * conf_loss
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            running_loss += loss.item()
            running_price += price_loss.item()
            running_conf += conf_loss.item()
            running_aux += aux_loss.item()
            batches += 1

        train_loss = running_loss / max(batches, 1)
        train_price = running_price / max(batches, 1)
        train_conf = running_conf / max(batches, 1)
        train_aux = running_aux / max(batches, 1)

        val_loss, val_price, val_conf, val_aux = evaluate(
            model,
            val_loader,
            price_loss_fn,
            confidence_loss_fn,
            device,
            args.confidence_weight,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_price_loss": train_price,
                "train_conf_loss": train_conf,
                "train_aux_loss": train_aux,
                "val_loss": val_loss,
                "val_price_loss": val_price,
                "val_conf_loss": val_conf,
                "val_aux_loss": val_aux,
            }
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"price_loss={val_price:.5f} "
            f"aux_loss={val_aux:.5f} "
            f"conf_loss={val_conf:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = args.checkpoint_dir / args.pair.replace("/", "_")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / "next_bar_transformer.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": asdict(model_config),
                    "metadata": metadata,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

            metrics_path = checkpoint_dir / "metrics.json"
            metrics_data = {
                "history": history,
                "best_val_loss": best_val_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            with metrics_path.open("w") as fp:
                json.dump(metrics_data, fp, indent=2)

    test_loss, test_price, test_conf, test_aux = evaluate(
        model,
        test_loader,
        price_loss_fn,
        confidence_loss_fn,
        device,
        args.confidence_weight,
    )
    print(
        f"[Test] loss={test_loss:.5f} price_loss={test_price:.5f} "
        f"aux_loss={test_aux:.5f} confidence_loss={test_conf:.5f}"
    )


if __name__ == "__main__":
    main()

