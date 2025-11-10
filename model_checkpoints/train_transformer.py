"""Training script for the multi-layer trading transformer stack."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import get_config
from modeling.data_processing import (
    prepare_daily_dataset,
    prepare_hourly_dataset,
    prepare_execution_dataset,
)
from modeling.transformer_trading import (
    TransformerConfig,
    DailyRegimeTransformer,
    HourlySignalTransformer,
    ExecutionModel,
)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def chronological_split(dataset, val_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    n = len(dataset)
    if n == 0:
        return Subset(dataset, []), Subset(dataset, [])
    val_size = max(int(n * val_ratio), 1)
    train_size = max(n - val_size, 1)
    indices = list(range(n))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def evaluate_classification(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total if total > 0 else float('nan')
    acc = correct / total if total > 0 else float('nan')
    return {"loss": avg_loss, "accuracy": acc}


def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    mse_criterion = nn.MSELoss(reduction='sum')
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mse_sum += mse_criterion(outputs, targets).item()
            mae_sum += torch.abs(outputs - targets).mean(dim=1).sum().item()
            total += targets.size(0)
    mse = mse_sum / total if total > 0 else float('nan')
    mae = mae_sum / total if total > 0 else float('nan')
    return {"mse": mse, "mae": mae}


def _ensure_non_empty(dataset, dataset_name: str, sequence_length: int) -> None:
    if len(dataset) <= 0:
        raise ValueError(
            f"{dataset_name} dataset has insufficient samples (sequence length = {sequence_length}). "
            "Make sure the historical pipeline has produced enough data, or lower the sequence length "
            "via the corresponding --seq_* argument."
        )


def train_daily_model(
    db_path: Path,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    sequence_length: int = 60,
    val_ratio: float = 0.2,
    dropout: float = 0.1,
    label_mode: str = 'ternary',
):
    dataset = prepare_daily_dataset(db_path, sequence_length=sequence_length, label_mode=label_mode)
    _ensure_non_empty(dataset, "Daily", sequence_length)
    train_ds, val_ds = chronological_split(dataset, val_ratio=val_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    config = TransformerConfig(feature_dim=dataset.features.shape[1], dropout=dropout)
    num_classes = getattr(dataset, "num_classes", 3)
    model = DailyRegimeTransformer(config, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    best_metrics = {"loss": float('inf'), "accuracy": 0.0}
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        print(f"[Daily] Epoch {epoch+1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            best_metrics = {"loss": val_loss, "accuracy": val_acc}

    if best_state is not None:
        model.load_state_dict(best_state)
    eval_metrics = evaluate_classification(model, val_loader, device)
    return model, {
        "best_val_loss": best_metrics["loss"],
        "best_val_accuracy": best_metrics["accuracy"],
        "eval_loss": eval_metrics["loss"],
        "eval_accuracy": eval_metrics["accuracy"],
        "history": history,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "label_mode": label_mode,
        "num_classes": num_classes,
    }


def train_hourly_model(
    db_path: Path,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-4,
    sequence_length: int = 24,
    val_ratio: float = 0.2,
    dropout: float = 0.1,
):
    dataset = prepare_hourly_dataset(db_path, sequence_length=sequence_length)
    _ensure_non_empty(dataset, "Hourly", sequence_length)
    train_ds, val_ds = chronological_split(dataset, val_ratio=val_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    config = TransformerConfig(feature_dim=dataset.features.shape[1], dropout=dropout)
    model = HourlySignalTransformer(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[Hourly] Epoch {epoch+1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    eval_metrics = evaluate_regression(model, val_loader, device)
    eval_metrics["best_val_loss"] = best_val
    eval_metrics["history"] = history
    eval_metrics["train_samples"] = len(train_ds)
    eval_metrics["val_samples"] = len(val_ds)
    return model, eval_metrics


def train_execution_model(
    db_path: Path,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    lookback: int = 24,
    val_ratio: float = 0.2,
    dropout: float = 0.1,
):
    dataset = prepare_execution_dataset(db_path, lookback=lookback)
    _ensure_non_empty(dataset, "Execution", lookback)
    train_ds, val_ds = chronological_split(dataset, val_ratio=val_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    input_dim = dataset[0][0].numel()
    model = ExecutionModel(input_dim=input_dim, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[Execution] Epoch {epoch+1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    eval_metrics = evaluate_regression(model, val_loader, device)
    eval_metrics["best_val_loss"] = best_val
    eval_metrics["history"] = history
    eval_metrics["train_samples"] = len(train_ds)
    eval_metrics["val_samples"] = len(val_ds)
    return model, eval_metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-layer transformer trading models.')
    parser.add_argument('--epochs_daily', type=int, default=20)
    parser.add_argument('--epochs_hourly', type=int, default=20)
    parser.add_argument('--epochs_execution', type=int, default=10)
    parser.add_argument('--batch_daily', type=int, default=32)
    parser.add_argument('--batch_hourly', type=int, default=64)
    parser.add_argument('--batch_execution', type=int, default=128)
    parser.add_argument('--lr_daily', type=float, default=1e-3)
    parser.add_argument('--lr_hourly', type=float, default=5e-4)
    parser.add_argument('--lr_execution', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='model_checkpoints')
    parser.add_argument('--seq_daily', type=int, default=60, help='Sequence length (days) for the daily regime model')
    parser.add_argument('--seq_hourly', type=int, default=24, help='Sequence length (hours) for the hourly transformer')
    parser.add_argument('--seq_execution', type=int, default=24, help='Lookback (hours) for the execution model')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio for chronological split')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for transformer and execution models')
    parser.add_argument('--daily_label_mode', type=str, choices=['ternary', 'binary'], default='ternary',
                        help='Regime label mode for daily model')
    args = parser.parse_args()

    config = get_config()
    db_path = Path(config.db_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    metrics_summary = {
        "hyperparameters": {
            "epochs_daily": args.epochs_daily,
            "epochs_hourly": args.epochs_hourly,
            "epochs_execution": args.epochs_execution,
            "batch_daily": args.batch_daily,
            "batch_hourly": args.batch_hourly,
            "batch_execution": args.batch_execution,
            "lr_daily": args.lr_daily,
            "lr_hourly": args.lr_hourly,
            "lr_execution": args.lr_execution,
            "seq_daily": args.seq_daily,
            "seq_hourly": args.seq_hourly,
            "seq_execution": args.seq_execution,
            "val_ratio": args.val_ratio,
            "dropout": args.dropout,
            "daily_label_mode": args.daily_label_mode,
        }
    }

    daily_model, daily_metrics = train_daily_model(
        db_path,
        device,
        epochs=args.epochs_daily,
        batch_size=args.batch_daily,
        lr=args.lr_daily,
        sequence_length=args.seq_daily,
        val_ratio=args.val_ratio,
        dropout=args.dropout,
        label_mode=args.daily_label_mode,
    )
    torch.save(daily_model.state_dict(), output_dir / 'daily_regime_transformer.pt')
    metrics_summary["daily"] = daily_metrics

    hourly_model, hourly_metrics = train_hourly_model(
        db_path,
        device,
        epochs=args.epochs_hourly,
        batch_size=args.batch_hourly,
        lr=args.lr_hourly,
        sequence_length=args.seq_hourly,
        val_ratio=args.val_ratio,
        dropout=args.dropout,
    )
    torch.save(hourly_model.state_dict(), output_dir / 'hourly_signal_transformer.pt')
    metrics_summary["hourly"] = hourly_metrics

    execution_model, execution_metrics = train_execution_model(
        db_path,
        device,
        epochs=args.epochs_execution,
        batch_size=args.batch_execution,
        lr=args.lr_execution,
        lookback=args.seq_execution,
        val_ratio=args.val_ratio,
        dropout=args.dropout,
    )
    torch.save(execution_model.state_dict(), output_dir / 'hourly_execution_model.pt')

    metrics_summary["execution"] = execution_metrics

    metrics_path = output_dir / 'metrics.json'
    with metrics_path.open('w') as f:
        json.dump(metrics_summary, f, indent=2)

    print('Training complete. Models saved to', output_dir)
    print('Metrics saved to', metrics_path)


if __name__ == '__main__':
    main()
