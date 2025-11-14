"""
Diagnostic script to check if model predictions are actually useful.
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config

def analyze_predictions(db_path: Path, pair: str = "ADA/USD", start: str = "2024-01-01T00:00:00", end: str = "2024-12-31T23:00:00"):
    """Analyze prediction accuracy."""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT 
                as_of,
                reference_close,
                pred_close,
                pred_delta_close,
                confidence,
                actual_close,
                (actual_close - reference_close) / reference_close as actual_delta
            FROM next_bar_forecasts
            WHERE pair = ?
              AND as_of >= ?
              AND as_of <= ?
              AND actual_close IS NOT NULL
              AND pred_delta_close IS NOT NULL
            ORDER BY as_of
        """
        df = pd.read_sql(query, conn, params=[pair, start, end])
    
    if df.empty:
        print(f"No data found for {pair}")
        return
    
    # Handle binary BLOB data
    def safe_float(x):
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, bytes):
            try:
                import struct
                if len(x) == 4:
                    return struct.unpack('<f', x)[0]
                elif len(x) == 8:
                    return struct.unpack('<d', x)[0]
            except:
                return 0.0
        try:
            return float(x)
        except:
            return 0.0
    
    df['pred_delta_close'] = df['pred_delta_close'].apply(safe_float)
    df['confidence'] = df['confidence'].apply(safe_float)
    df['reference_close'] = df['reference_close'].apply(safe_float)
    df['actual_close'] = df['actual_close'].apply(safe_float)
    df['pred_close'] = df['pred_close'].apply(safe_float)
    
    # Recalculate actual_delta
    df['actual_delta'] = (df['actual_close'] - df['reference_close']) / df['reference_close']
    
    # Filter for trades we would have taken
    df['edge'] = df['pred_delta_close']
    df['would_trade'] = (df['confidence'] >= 0.55) & (df['edge'].abs() >= 0.0005) & (df['edge'] > 0)
    
    print("="*80)
    print(f"MODEL PREDICTION ANALYSIS FOR {pair}")
    print("="*80)
    
    print(f"\nTotal forecasts: {len(df)}")
    print(f"Forecasts we would trade: {df['would_trade'].sum()}")
    
    # Overall prediction accuracy
    print("\n" + "="*80)
    print("OVERALL PREDICTION ACCURACY")
    print("="*80)
    
    # Direction accuracy
    df['pred_direction'] = np.where(df['pred_delta_close'] > 0, 1, -1)
    df['actual_direction'] = np.where(df['actual_delta'] > 0, 1, -1)
    direction_accuracy = (df['pred_direction'] == df['actual_direction']).mean()
    print(f"Direction accuracy: {direction_accuracy*100:.2f}%")
    print(f"  (Random would be 50%)")
    
    if direction_accuracy < 0.5:
        print("  ❌ WORSE THAN RANDOM!")
    elif direction_accuracy < 0.55:
        print("  ⚠️  Barely better than random")
    else:
        print(f"  ✅ Better than random by {direction_accuracy*100 - 50:.2f}%")
    
    # For trades we would take
    trades = df[df['would_trade']].copy()
    if len(trades) > 0:
        print(f"\nTrades we would take: {len(trades)}")
        trade_direction_accuracy = (trades['pred_direction'] == trades['actual_direction']).mean()
        print(f"Direction accuracy (trades): {trade_direction_accuracy*100:.2f}%")
        
        # Check if predictions are correct
        correct_predictions = (trades['actual_delta'] > 0).sum()
        print(f"Predictions that were correct (price went up): {correct_predictions}/{len(trades)} ({correct_predictions/len(trades)*100:.2f}%)")
        
        # Average predicted vs actual move
        print(f"\nAverage predicted move: {trades['pred_delta_close'].mean()*100:.4f}%")
        print(f"Average actual move: {trades['actual_delta'].mean()*100:.4f}%")
        print(f"Difference: {(trades['pred_delta_close'].mean() - trades['actual_delta'].mean())*100:.4f}%")
        
        # Correlation
        correlation = trades['pred_delta_close'].corr(trades['actual_delta'])
        print(f"\nCorrelation between predicted and actual: {correlation:.4f}")
        if correlation < 0.1:
            print("  ❌ Very weak correlation - predictions are not useful")
        elif correlation < 0.3:
            print("  ⚠️  Weak correlation - predictions have limited value")
        else:
            print("  ✅ Moderate to strong correlation - predictions are useful")
        
        # Check if stops would hit
        print("\n" + "="*80)
        print("STOP/TARGET ANALYSIS (with 0.5% stop, 1.25% target)")
        print("="*80)
        
        stop_pct = 0.005
        target_pct = 0.0125
        
        # Approximate: if actual_delta < -stop_pct, stop would hit
        # If actual_delta > target_pct, target would hit
        stop_hits = (trades['actual_delta'] < -stop_pct).sum()
        target_hits = (trades['actual_delta'] > target_pct).sum()
        neither = len(trades) - stop_hits - target_hits
        
        print(f"Stops that would hit: {stop_hits}/{len(trades)} ({stop_hits/len(trades)*100:.2f}%)")
        print(f"Targets that would hit: {target_hits}/{len(trades)} ({target_hits/len(trades)*100:.2f}%)")
        print(f"Neither hit: {neither}/{len(trades)} ({neither/len(trades)*100:.2f}%)")
        
        # Expected win rate
        expected_win_rate = target_hits / len(trades) if len(trades) > 0 else 0
        print(f"\nExpected win rate (if target hits = win): {expected_win_rate*100:.2f}%")
        print(f"Actual backtest win rate: 33.13%")
        print(f"Difference: {expected_win_rate*100 - 33.13:.2f}%")
        
        # Check magnitude of moves
        print("\n" + "="*80)
        print("MOVE MAGNITUDE ANALYSIS")
        print("="*80)
        print(f"Average predicted move magnitude: {trades['pred_delta_close'].abs().mean()*100:.4f}%")
        print(f"Average actual move magnitude: {trades['actual_delta'].abs().mean()*100:.4f}%")
        print(f"Median predicted move: {trades['pred_delta_close'].abs().median()*100:.4f}%")
        print(f"Median actual move: {trades['actual_delta'].abs().median()*100:.4f}%")
        
        # Check if predictions are too optimistic/pessimistic
        print("\n" + "="*80)
        print("PREDICTION BIAS")
        print("="*80)
        bias = trades['pred_delta_close'].mean() - trades['actual_delta'].mean()
        print(f"Average bias: {bias*100:.4f}%")
        if abs(bias) > 0.001:
            if bias > 0:
                print("  ⚠️  Model is OVERLY OPTIMISTIC (predicts larger moves than actual)")
            else:
                print("  ⚠️  Model is OVERLY PESSIMISTIC (predicts smaller moves than actual)")
        else:
            print("  ✅ Model predictions are well-calibrated")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if len(trades) > 0:
        if direction_accuracy < 0.5:
            print("❌ MODEL IS WORSE THAN RANDOM")
            print("   → Need to retrain model or use different features")
        elif direction_accuracy < 0.55:
            print("⚠️  MODEL IS BARELY BETTER THAN RANDOM")
            print("   → Model needs significant improvement")
        elif correlation < 0.2:
            print("⚠️  MODEL HAS WEAK CORRELATION WITH ACTUALS")
            print("   → Predictions are not reliable enough for trading")
        else:
            print("✅ MODEL HAS SOME PREDICTIVE POWER")
            print("   → But execution logic might need improvement")
            print("   → Or stop/target settings need adjustment")

if __name__ == "__main__":
    config = get_config()
    db_path = Path(config.db_path)
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Please run this on the server where the database exists")
        sys.exit(1)
    
    analyze_predictions(
        db_path=db_path,
        pair="ADA/USD",
        start="2024-01-01T00:00:00",
        end="2024-12-31T23:00:00"
    )

