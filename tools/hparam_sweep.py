"""
Simple hyperparameter sweep runner for the transformer training script.

Edit the `GRID` dictionary below to control which hyperparameters are tested.
Each run writes its checkpoints and metrics.json to a dedicated subdirectory
inside `model_checkpoints/`.
"""

import itertools
import json
import subprocess
import uuid
from pathlib import Path

BASE_CMD = ["python", "-m", "modeling.train_transformer"]

GRID = {
    "seq_daily": [45, 60],
    "seq_hourly": [24, 36],
    "seq_execution": [24, 36],
    "lr_hourly": [5e-4, 3e-4],
    "lr_execution": [1e-3, 5e-4],
    "dropout": [0.1, 0.2],
}

FIXED_ARGS = {
    "epochs_daily": 25,
    "epochs_hourly": 30,
    "epochs_execution": 15,
    "daily_label_mode": "ternary",
}


def format_run_name(params: dict) -> str:
    return (
        f"sd{params['seq_daily']}_sh{params['seq_hourly']}_"
        f"se{params['seq_execution']}_lrh{params['lr_hourly']}_"
        f"lre{params['lr_execution']}_do{params['dropout']}"
    )


def run_once(params: dict) -> dict:
    run_name = format_run_name(params)
    out_dir = Path("model_checkpoints") / f"{run_name}_{uuid.uuid4().hex[:6]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = BASE_CMD + [
        "--output_dir",
        str(out_dir),
    ]

    for key, value in {**FIXED_ARGS, **params}.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n=== Running {run_name} ===")
    subprocess.run(cmd, check=True)

    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            metrics = json.load(f)
    else:
        metrics = {}
    return {"run_name": run_name, "output_dir": str(out_dir), "metrics": metrics}


def main() -> None:
    keys = list(GRID.keys())
    results = []
    for values in itertools.product(*[GRID[k] for k in keys]):
        params = dict(zip(keys, values))
        result = run_once(params)
        results.append(result)

    summary_path = Path("model_checkpoints") / "sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()

