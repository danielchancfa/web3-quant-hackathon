"""
Inference service for the trading models.

Usage:
    python inference_service.py \
        --checkpoint_dir model_checkpoints/run_binary \
        --seq_daily 60 \
        --seq_hourly 36 \
        --seq_execution 36 \
        --daily_label_mode binary
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch

from config import get_config
from modeling.data_processing import (
    prepare_daily_dataset,
    prepare_hourly_dataset,
    prepare_execution_dataset,
    DEFAULT_PAIR,
)
from modeling.transformer_trading import (
    TransformerConfig,
    DailyRegimeTransformer,
    HourlySignalTransformer,
    ExecutionModel,
)


def to_float_dict(values) -> Dict[str, float]:
    return {str(i): float(v) for i, v in enumerate(values)}


def pair_to_dirname(pair: str) -> str:
    return pair.replace('/', '_')


class InferenceService:
    def __init__(
        self,
        db_path: Path,
        checkpoint_dir: Path,
        seq_daily: int,
        seq_hourly: int,
        seq_execution: int,
        daily_label_mode: str,
        dropout: float = 0.1,
        device: str = None,
        pairs: list[str] = None,
    ) -> None:
        self.db_path = db_path
        self.checkpoint_dir = checkpoint_dir
        self.seq_daily = seq_daily
        self.seq_hourly = seq_hourly
        self.seq_execution = seq_execution
        self.daily_label_mode = daily_label_mode
        self.dropout = dropout
        self.pairs = pairs or [DEFAULT_PAIR]
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_models()

    def _load_models(self) -> None:
        self.models: Dict[str, Dict[str, Any]] = {}
        for pair in self.pairs:
            pair_dir = self.checkpoint_dir / pair_to_dirname(pair)
            if not pair_dir.exists():
                raise RuntimeError(f"Checkpoint directory {pair_dir} for pair {pair} not found.")

            daily_dataset = prepare_daily_dataset(
                self.db_path,
                sequence_length=self.seq_daily,
                label_mode=self.daily_label_mode,
                pair=pair,
            )
            if len(daily_dataset) <= 0:
                raise RuntimeError(
                    f"Daily dataset has no samples for pair {pair} with seq_daily={self.seq_daily}."
                )
            daily_config = TransformerConfig(
                feature_dim=daily_dataset.features.shape[1],
                dropout=self.dropout,
            )
            num_classes = getattr(daily_dataset, "num_classes", 3)
            daily_model = DailyRegimeTransformer(daily_config, num_classes=num_classes).to(self.device)
            daily_state = torch.load(
                pair_dir / 'daily_regime_transformer.pt',
                map_location=self.device,
            )
            daily_model.load_state_dict(daily_state)
            daily_model.eval()

            hourly_dataset = prepare_hourly_dataset(
                self.db_path,
                sequence_length=self.seq_hourly,
                pair=pair,
            )
            if len(hourly_dataset) <= 0:
                raise RuntimeError(
                    f"Hourly dataset has no samples for pair {pair} with seq_hourly={self.seq_hourly}."
                )
            hourly_config = TransformerConfig(
                feature_dim=hourly_dataset.features.shape[1],
                dropout=self.dropout,
            )
            hourly_model = HourlySignalTransformer(hourly_config).to(self.device)
            hourly_state = torch.load(
                pair_dir / 'hourly_signal_transformer.pt',
                map_location=self.device,
            )
            hourly_model.load_state_dict(hourly_state)
            hourly_model.eval()

            execution_dataset = prepare_execution_dataset(
                self.db_path,
                lookback=self.seq_execution,
                pair=pair,
            )
            if len(execution_dataset) <= 0:
                raise RuntimeError(
                    f"Execution dataset has no samples for pair {pair} with seq_execution={self.seq_execution}."
                )
            exec_input_dim = execution_dataset[0][0].numel()
            execution_model = ExecutionModel(
                input_dim=exec_input_dim,
                dropout=self.dropout,
            ).to(self.device)
            exec_state = torch.load(
                pair_dir / 'hourly_execution_model.pt',
                map_location=self.device,
            )
            execution_model.load_state_dict(exec_state)
            execution_model.eval()

            self.models[pair] = {
                "daily_dataset": daily_dataset,
                "daily_model": daily_model,
                "num_classes": num_classes,
                "hourly_dataset": hourly_dataset,
                "hourly_model": hourly_model,
                "execution_dataset": execution_dataset,
                "execution_model": execution_model,
            }

    def _make_tensor(self, sample: torch.Tensor) -> torch.Tensor:
        return sample.unsqueeze(0).to(self.device)

    def predict_daily(self, pair: str) -> Dict[str, Any]:
        model_bundle = self.models[pair]
        daily_dataset = model_bundle["daily_dataset"]
        daily_model = model_bundle["daily_model"]
        num_classes = model_bundle["num_classes"]

        if len(daily_dataset) <= 0:
            raise RuntimeError("Daily dataset empty at inference time.")
        # Use the full dataset as a batch (each item already a sequence)
        features = torch.stack([daily_dataset[i][0] for i in range(len(daily_dataset))])
        logits = daily_model(features.to(self.device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        latest_probs = probs[-1]
        pred_idx = int(latest_probs.argmax())

        if self.daily_label_mode == 'binary':
            classes = ['risk_off', 'risk_on']
        else:
            classes = ['bearish', 'neutral', 'bullish'][:num_classes]

        return {
            "label_mode": self.daily_label_mode,
            "classes": classes,
            "probabilities": {cls: float(latest_probs[i]) for i, cls in enumerate(classes)},
            "prediction": classes[pred_idx],
        }

    def predict_hourly(self, pair: str) -> Dict[str, Any]:
        model_bundle = self.models[pair]
        hourly_dataset = model_bundle["hourly_dataset"]
        hourly_model = model_bundle["hourly_model"]

        features = torch.stack([hourly_dataset[i][0] for i in range(len(hourly_dataset))])
        outputs = hourly_model(features.to(self.device)).detach().cpu().numpy()
        output = outputs[-1]
        buy_amount, sell_amount, hold_confidence = output.tolist()
        net_signal = buy_amount - sell_amount

        return {
            "raw_output": {
                "buy_amount": float(buy_amount),
                "sell_amount": float(sell_amount),
                "hold_confidence": float(hold_confidence),
            },
            "net_signal": float(net_signal),
        }

    def predict_execution(self, pair: str) -> Dict[str, Any]:
        model_bundle = self.models[pair]
        execution_dataset = model_bundle["execution_dataset"]
        execution_model = model_bundle["execution_model"]

        features = torch.stack([execution_dataset[i][0] for i in range(len(execution_dataset))])
        outputs = execution_model(features.to(self.device)).detach().cpu().numpy()
        output = outputs[-1]
        buy_amount, sell_amount, hold_confidence = output.tolist()
        net_signal = buy_amount - sell_amount
        return {
            "raw_output": {
                "buy_amount": float(buy_amount),
                "sell_amount": float(sell_amount),
                "hold_confidence": float(hold_confidence),
            },
            "net_signal": float(net_signal),
        }

    def run_once(self) -> Dict[str, Any]:
        results = {}
        for pair in self.pairs:
            results[pair] = {
                "daily": self.predict_daily(pair),
                "hourly": self.predict_hourly(pair),
                "execution": self.predict_execution(pair),
            }
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using trained models.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory with model checkpoints.')
    parser.add_argument('--seq_daily', type=int, default=60)
    parser.add_argument('--seq_hourly', type=int, default=36)
    parser.add_argument('--seq_execution', type=int, default=36)
    parser.add_argument('--daily_label_mode', type=str, choices=['ternary', 'binary'], default='ternary')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on.')
    parser.add_argument('--pairs', type=str, default=DEFAULT_PAIR,
                        help='Comma-separated list of pairs to infer on (e.g., "BTC/USD,ETH/USD"). '
                             'Expect checkpoints under checkpoint_dir/<PAIR>.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    db_path = Path(config.db_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]

    service = InferenceService(
        db_path=db_path,
        checkpoint_dir=checkpoint_dir,
        seq_daily=args.seq_daily,
        seq_hourly=args.seq_hourly,
        seq_execution=args.seq_execution,
        daily_label_mode=args.daily_label_mode,
        dropout=args.dropout,
        device=args.device,
        pairs=pairs,
    )
    result = service.run_once()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

