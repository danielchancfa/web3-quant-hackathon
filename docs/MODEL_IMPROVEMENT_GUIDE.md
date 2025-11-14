# Step-by-Step Guide: Improving Next-Bar Model & Policy

## Part 1: Improve the Next-Bar Model

### Step 1.1: Check Current Model Performance

First, understand what you're working with:

```bash
# Check how many forecasts pass your current thresholds
python - <<'PY'
import sqlite3, pandas as pd
with sqlite3.connect("data_cache/trading_data.db") as conn:
    for pair in ["BTC/USD", "ETH/USD", "SOL/USD"]:
        df = pd.read_sql(f"""
            SELECT confidence, pred_delta_close
            FROM next_bar_forecasts
            WHERE pair='{pair}'
              AND as_of BETWEEN '2024-01-01T00:00:00' AND '2024-12-31T23:00:00'
        """, conn)
        if not df.empty:
            passing = ((df['confidence'] >= 0.55) & (df['pred_delta_close'].abs() >= 0.0005)).sum()
            print(f"{pair}: {passing}/{len(df)} forecasts pass thresholds ({100*passing/len(df):.1f}%)")
PY
```

**What this tells you**: If <5% of forecasts pass, your model needs improvement.

---

### Step 1.2: Retrain with Better Hyperparameters

The current model uses default settings. Let's experiment:

**Option A: Longer sequence length (more context)**
```bash
python -m prediction_model.train_next_bar \
  --pair ETH/USD \
  --sequence_length 256 \
  --batch_size 32 \
  --epochs 40 \
  --lr 5e-5 \
  --confidence_weight 0.3 \
  --checkpoint_dir model_checkpoints/next_bar \
  --db_path data_cache/trading_data.db
```

**What this does**:
- `--sequence_length 256`: Model sees 256 hours (~10 days) instead of 128
- `--batch_size 32`: Smaller batches to fit longer sequences
- `--lr 5e-5`: Lower learning rate for stability
- `--confidence_weight 0.3`: Less emphasis on confidence (focus on price prediction)

**Option B: Deeper model (more capacity)**
```bash
# First, edit prediction_model/model.py to add a config option:
# In NextBarTransformerConfig, add: num_layers: int = 4
# Then retrain:
python -m prediction_model.train_next_bar \
  --pair ETH/USD \
  --sequence_length 128 \
  --epochs 50 \
  --lr 1e-4 \
  --checkpoint_dir model_checkpoints/next_bar \
  --db_path data_cache/trading_data.db
```

---

### Step 1.3: Add More Features

Edit `prediction_model/data.py` to include additional indicators:

```python
# In _compute_hourly_features, add:
def _compute_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing code ...
    
    # Add momentum features
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    # Add volatility clustering
    df['volatility_5'] = df['close'].pct_change().rolling(5).std()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    
    # Add price position in range
    df['price_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    return df
```

Then retrain:
```bash
python -m prediction_model.train_next_bar --pair ETH/USD ...
```

---

### Step 1.4: Use More Training Data

Extend the training window to include 2023:

```bash
# First, ensure you have 2023 data in the DB
python -m data_pipeline.historical_data_pipeline

# Then retrain with longer history
python -m prediction_model.train_next_bar \
  --pair ETH/USD \
  --sequence_length 128 \
  --epochs 40 \
  --checkpoint_dir model_checkpoints/next_bar \
  --db_path data_cache/trading_data.db
```

The model will automatically use all available data in the DB.

---

### Step 1.5: Regenerate Forecasts with New Model

After retraining, generate fresh forecasts:

```bash
# Generate for ETH/USD
python -m tools.generate_historical_forecasts \
  --pairs "ETH/USD" \
  --start 2024-01-01T00:00:00 \
  --end 2024-12-31T23:00:00 \
  --freq_hours 1 \
  --checkpoint_dir model_checkpoints/next_bar \
  --db_path data_cache/trading_data.db

# Backfill actuals
python -m tools.backfill_actuals \
  --pairs "ETH/USD" \
  --db_path data_cache/trading_data.db
```

---

## Part 2: Refine the Policy Backtest

### Step 2.1: Find Optimal Thresholds Per Pair

Use the optimizer to find best parameters:

```bash
python -m tools.optimize_policy \
  --db_path data_cache/trading_data.db \
  --pairs "ETH/USD" \
  --trials 100 \
  --min_trades 50 \
  --start 2024-01-01T00:00:00 \
  --end 2024-12-31T23:00:00
```

**What this does**: Optuna will try 100 different combinations of:
- `min_confidence` (0.40 to 0.70)
- `min_edge` (0.0001 to 0.002)
- `base_risk_fraction` (0.01 to 0.05)
- `max_risk_fraction` (0.03 to 0.10)

And pick the one with highest Sharpe ratio.

**Output**: You'll see something like:
```
Best Sharpe: 1.234
Best params:
  min_confidence: 0.52
  min_edge: 0.0004
  base_risk_fraction: 0.025
  max_risk_fraction: 0.06
```

---

### Step 2.2: Test the Optimized Parameters

Run a backtest with the best parameters:

```bash
python -m tools.backtest_policy \
  --pairs "ETH/USD" \
  --min_confidence 0.52 \
  --min_edge 0.0004 \
  --base_risk_fraction 0.025 \
  --max_risk_fraction 0.06 \
  --start 2024-01-01T00:00:00 \
  --end 2024-12-31T23:00:00 \
  --db_path data_cache/trading_data.db
```

**Check the output**:
- `# Trades`: Should be > 50
- `Return [%]`: Should be positive
- `Sharpe Ratio`: Should be > 1.0
- `Max. Drawdown [%]`: Should be < 20%

---

### Step 2.3: Manual Threshold Tuning

If optimizer results aren't good, manually test ranges:

```bash
# Test different confidence thresholds
for conf in 0.45 0.50 0.55 0.60; do
  echo "Testing confidence=$conf"
  python -m tools.backtest_policy \
    --pairs "ETH/USD" \
    --min_confidence $conf \
    --min_edge 0.0005 \
    --start 2024-01-01T00:00:00 \
    --end 2024-12-31T23:00:00 \
    --db_path data_cache/trading_data.db | grep -E "(# Trades|Return|Sharpe)"
done
```

Pick the confidence value that gives best Sharpe.

---

## Part 3: Automate Iterative Improvement

### Step 3.1: Create an Improvement Loop Script

Create `tools/improve_model_iterative.py`:

```python
#!/usr/bin/env python3
"""
Iterative improvement loop:
1. Train model with current hyperparameters
2. Generate forecasts
3. Backfill actuals
4. Optimize policy thresholds
5. Run backtest
6. If results unsatisfactory, adjust hyperparameters and repeat
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config

def run_cmd(cmd, check=True):
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        sys.exit(1)
    return result

def main():
    pair = "ETH/USD"
    iteration = 1
    best_sharpe = -999.0
    
    while iteration <= 3:  # Max 3 iterations
        print(f"\n=== Iteration {iteration} ===")
        
        # Step 1: Retrain model
        print(f"[1/5] Retraining {pair}...")
        run_cmd([
            "python", "-m", "prediction_model.train_next_bar",
            "--pair", pair,
            "--sequence_length", "256",
            "--epochs", "40",
            "--lr", "5e-5",
            "--confidence_weight", "0.3",
            "--checkpoint_dir", "model_checkpoints/next_bar",
            "--db_path", "data_cache/trading_data.db",
        ])
        
        # Step 2: Generate forecasts
        print(f"[2/5] Generating forecasts for {pair}...")
        run_cmd([
            "python", "-m", "tools.generate_historical_forecasts",
            "--pairs", pair,
            "--start", "2024-01-01T00:00:00",
            "--end", "2024-12-31T23:00:00",
            "--freq_hours", "1",
            "--checkpoint_dir", "model_checkpoints/next_bar",
            "--db_path", "data_cache/trading_data.db",
        ])
        
        # Step 3: Backfill actuals
        print(f"[3/5] Backfilling actuals for {pair}...")
        run_cmd([
            "python", "-m", "tools.backfill_actuals",
            "--pairs", pair,
            "--db_path", "data_cache/trading_data.db",
        ])
        
        # Step 4: Optimize policy
        print(f"[4/5] Optimizing policy for {pair}...")
        opt_result = run_cmd([
            "python", "-m", "tools.optimize_policy",
            "--db_path", "data_cache/trading_data.db",
            "--pairs", pair,
            "--trials", "50",
            "--min_trades", "30",
            "--start", "2024-01-01T00:00:00",
            "--end", "2024-12-31T23:00:00",
        ])
        
        # Extract Sharpe from output (simplified - you may need to parse JSON)
        # For now, just run a final backtest
        print(f"[5/5] Running final backtest for {pair}...")
        bt_result = run_cmd([
            "python", "-m", "tools.backtest_policy",
            "--pairs", pair,
            "--min_confidence", "0.50",
            "--min_edge", "0.0004",
            "--start", "2024-01-01T00:00:00",
            "--end", "2024-12-31T23:00:00",
            "--db_path", "data_cache/trading_data.db",
        ], check=False)
        
        # Check if Sharpe improved (you'd parse this from output)
        print(f"\nIteration {iteration} complete. Check output above for Sharpe ratio.")
        print("If Sharpe < 1.0, adjust hyperparameters and continue.")
        
        iteration += 1

if __name__ == "__main__":
    main()
```

Run it:
```bash
python tools/improve_model_iterative.py
```

---

### Step 3.2: Create a Hyperparameter Sweep Script

For systematic exploration, create `tools/hyperparameter_sweep.py`:

```python
#!/usr/bin/env python3
"""
Sweep hyperparameters and track results.
"""

import subprocess
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Define hyperparameter grid
HYPERPARAMS = [
    {"sequence_length": 128, "lr": 1e-4, "confidence_weight": 0.5},
    {"sequence_length": 256, "lr": 5e-5, "confidence_weight": 0.3},
    {"sequence_length": 128, "lr": 1e-4, "confidence_weight": 0.2},
    {"sequence_length": 192, "lr": 7.5e-5, "confidence_weight": 0.4},
]

def train_and_evaluate(pair, params, iteration):
    checkpoint_dir = Path(f"model_checkpoints/next_bar_sweep_{iteration}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    subprocess.run([
        "python", "-m", "prediction_model.train_next_bar",
        "--pair", pair,
        "--sequence_length", str(params["sequence_length"]),
        "--lr", str(params["lr"]),
        "--confidence_weight", str(params["confidence_weight"]),
        "--checkpoint_dir", str(checkpoint_dir),
        "--db_path", "data_cache/trading_data.db",
    ], check=True)
    
    # Generate forecasts
    subprocess.run([
        "python", "-m", "tools.generate_historical_forecasts",
        "--pairs", pair,
        "--start", "2024-01-01T00:00:00",
        "--end", "2024-12-31T23:00:00",
        "--checkpoint_dir", str(checkpoint_dir),
        "--db_path", "data_cache/trading_data.db",
    ], check=True)
    
    # Backfill
    subprocess.run([
        "python", "-m", "tools.backfill_actuals",
        "--pairs", pair,
        "--db_path", "data_cache/trading_data.db",
    ], check=True)
    
    # Optimize and backtest
    result = subprocess.run([
        "python", "-m", "tools.optimize_policy",
        "--db_path", "data_cache/trading_data.db",
        "--pairs", pair,
        "--trials", "30",
        "--start", "2024-01-01T00:00:00",
        "--end", "2024-12-31T23:00:00",
    ], capture_output=True, text=True)
    
    # Parse best Sharpe from output (simplified)
    return {"params": params, "output": result.stdout}

if __name__ == "__main__":
    pair = "ETH/USD"
    results = []
    
    for i, params in enumerate(HYPERPARAMS):
        print(f"\n=== Testing hyperparameters {i+1}/{len(HYPERPARAMS)} ===")
        result = train_and_evaluate(pair, params, i)
        results.append(result)
    
    # Save results
    with open("hyperparameter_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Sweep complete ===")
    print("Check hyperparameter_sweep_results.json for results")
```

---

### Step 3.3: Monitor Progress with Logs

The CSV logger (`tools/backtest_all_pairs.py`) already tracks results. After each iteration, check:

```bash
# View latest results
tail -20 logs/backtest_all_pairs.csv | cut -d',' -f1,2,7,8,11

# Compare Sharpe ratios
grep "ETH/USD" logs/backtest_all_pairs.csv | tail -5
```

---

## Summary Workflow

**Complete improvement cycle**:

1. **Retrain model** with better hyperparameters
2. **Regenerate forecasts** for the test period
3. **Backfill actuals** so you can evaluate
4. **Optimize policy thresholds** to maximize Sharpe
5. **Run backtest** with optimized params
6. **Check metrics**: If Sharpe < 1.0 or trades < 50, go back to step 1 with different hyperparameters

**Expected timeline**:
- Single iteration: ~2-4 hours (training + forecast generation)
- Full sweep (4 hyperparameter combos): ~8-16 hours
- Policy optimization: ~30 minutes per pair

**Success criteria**:
- Sharpe Ratio > 1.5
- # Trades > 50 per year
- Max Drawdown < 15%
- Win Rate > 50%

