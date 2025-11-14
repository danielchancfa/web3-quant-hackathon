# Next-Bar Prediction Module

This package trains a transformer to forecast the next hourly open, high, low, close and an internal confidence score for each trading pair.

## Components

- `data.py` – builds hourly feature tables (with daily context) and exposes `NextBarDataset` plus helpers for chronological splits.
- `model.py` – defines `NextBarTransformer`, a sequence encoder with a five-value head (`Δopen`, `Δhigh`, `Δlow`, `Δclose`, confidence).
- `train_next_bar.py` – end-to-end training script with AdamW, SmoothL1/MSE losses, gradient clipping and checkpointing.
- `inference.py` – loads a trained checkpoint, normalizes the latest sequence and emits reconstructed OHLC forecasts with confidence.

## Training

```bash
python -m prediction_model.train_next_bar \
  --pair BTC/USD \
  --sequence_length 128 \
  --epochs 40 \
  --batch_size 64
```

Checkpoints land in `model_checkpoints/next_bar/<PAIR>/` with metrics history.

## Inference

```bash
python -m prediction_model.inference \
  --pair BTC/USD \
  --checkpoint model_checkpoints/next_bar/BTC_USD/next_bar_transformer.pt
```

The script prints predicted OHLC and confidence, optionally writing JSON when `--output` is provided.

## Integration Notes

- Confidence is already sigmoid-normalized in inference output.
- Predicted prices are relative to the latest close stored in the DB; they assume Horus data is up to date.
- For live trading, convert the point forecast into order instructions (e.g., long when `close > open`, short otherwise) and close positions at the end of each hour.

