"""
Diagnostic script to check if model predictions are accurate.
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
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
    
    # Filter for trades we would have taken
    df['edge'] = df['pred_delta_close']
    df['would_trade'] = (df['confidence'] >= 0.60) & (df['edge'].abs() >= 0.001) & (df['edge'] > 0)
    
    print("="*80)
    print(f"PREDICTION ANALYSIS FOR {pair}")
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
        
        # Correlation
        correlation = trades['pred_delta_close'].corr(trades['actual_delta'])
        print(f"Correlation: {correlation:.4f}")
        
        # Check if stops would hit
        print("\n" + "="*80)
        print("STOP/TARGET ANALYSIS")
        print("="*80)
        
        # Fixed stops: 0.5% stop, 1.25% target
        stop_pct = 0.005
        target_pct = 0.0125
        
        # Calculate if stop or target would hit first
        trades['stop_price'] = trades['reference_close'] * (1 - stop_pct)
        trades['target_price'] = trades['reference_close'] * (1 + target_pct)
        
        # Check actual high/low (we need to get this from ohlcv)
        # For now, approximate using actual_close
        # If actual_delta < -stop_pct, stop would hit
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
        print(f"Actual backtest win rate: 6.94%")
        print(f"Difference: {expected_win_rate*100 - 6.94:.2f}%")
        
        if expected_win_rate * 100 > 6.94:
            print("\n⚠️  Expected win rate is higher than actual!")
            print("   This suggests stops are hitting BEFORE targets, even when price eventually goes up")
            print("   Solution: Wider stops or better entry timing")
        else:
            print("\n⚠️  Expected win rate matches actual!")
            print("   This suggests predictions are not accurate enough")
            print("   Solution: Better model or different approach")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(trades) > 0:
        if direction_accuracy < 0.5:
            print("❌ Direction accuracy < 50% - model is worse than random!")
            print("   → Need to retrain model or use different features")
        elif direction_accuracy < 0.55:
            print("⚠️  Direction accuracy ~50% - model is barely better than random")
            print("   → Model needs improvement")
        else:
            print("✅ Direction accuracy > 55% - model has some predictive power")
            print("   → But win rate is still low - check stop/target logic")
        
        if correlation < 0.1:
            print("\n❌ Correlation < 0.1 - predictions don't match actuals")
            print("   → Model predictions are not useful")
        elif correlation < 0.3:
            print("\n⚠️  Correlation < 0.3 - weak relationship")
            print("   → Model needs improvement")
        else:
            print("\n✅ Correlation > 0.3 - moderate relationship")
            print("   → Model has some predictive power")

if __name__ == "__main__":
    config = get_config()
    db_path = Path(config.db_path)
    
    analyze_predictions(
        db_path=db_path,
        pair="ADA/USD",
        start="2024-01-01T00:00:00",
        end="2024-12-31T23:00:00"
    )

