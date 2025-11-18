#!/usr/bin/env python3
"""
Quick diagnostic script to check current price vs 16-hour MA for all pairs.
Run this on AWS to verify MA trend status.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config

def check_ma_status(db_path: Path, ma_period: int = 16):
    """Check price vs MA for all pairs in database."""
    conn = sqlite3.connect(db_path)
    
    # Get all pairs with hourly data
    pairs_query = "SELECT DISTINCT pair FROM ohlcv WHERE interval = '1h' ORDER BY pair"
    pairs_df = pd.read_sql(pairs_query, conn)
    pairs = pairs_df['pair'].tolist()
    
    print("=" * 80)
    print(f"PRICE vs {ma_period}-HOUR MA STATUS")
    print("=" * 80)
    print(f"{'Pair':<15} {'Latest Price':<15} {'16h MA':<15} {'Status':<10} {'Diff %':<10} {'Latest Time':<20}")
    print("-" * 100)
    
    results = []
    for pair in pairs:
        try:
            query = """
                SELECT close, timestamp, datetime
                FROM ohlcv 
                WHERE pair = ? AND interval = '1h'
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            min_periods = max(ma_period, 20)
            df = pd.read_sql(query, conn, params=(pair, min_periods * 2))
            
            if len(df) < min_periods:
                print(f"{pair:<15} {'INSUFFICIENT DATA':<15} {'':<15} {'':<10} {'':<10} {'':<20}")
                continue
            
            # Get latest timestamp
            latest_timestamp = df['timestamp'].iloc[0]
            latest_datetime = df['datetime'].iloc[0] if 'datetime' in df.columns else None
            
            # Calculate MA
            prices = df['close'].astype(float).iloc[::-1]  # Reverse to chronological order
            ma_values = prices.rolling(window=ma_period).mean()
            
            if len(ma_values) == 0 or pd.isna(ma_values.iloc[-1]):
                print(f"{pair:<15} {'MA CALC FAILED':<15} {'':<15} {'':<10} {'':<10} {'':<20}")
                continue
            
            current_price = float(prices.iloc[-1])
            current_ma = float(ma_values.iloc[-1])
            diff_pct = ((current_price - current_ma) / current_ma) * 100
            
            if current_price > current_ma:
                status = "UPTREND"
                status_symbol = "✅"
            else:
                status = "DOWNTREND"
                status_symbol = "❌"
            
            # Format timestamp
            if latest_datetime:
                try:
                    from datetime import datetime
                    dt = pd.to_datetime(latest_datetime)
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_str = str(latest_timestamp)
            else:
                # Convert Unix timestamp to readable format
                try:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(latest_timestamp)
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_str = str(latest_timestamp)
            
            print(f"{pair:<15} ${current_price:<14.4f} ${current_ma:<14.4f} {status_symbol} {status:<9} {diff_pct:>9.2f}% {time_str:<20}")
            
            results.append({
                'pair': pair,
                'price': current_price,
                'ma': current_ma,
                'status': status,
                'diff_pct': diff_pct
            })
            
        except Exception as e:
            print(f"{pair:<15} ERROR: {str(e)}")
    
    conn.close()
    
    print("=" * 100)
    print(f"\nSUMMARY:")
    uptrends = [r for r in results if r['status'] == 'UPTREND']
    downtrends = [r for r in results if r['status'] == 'DOWNTREND']
    print(f"  Uptrends (price > MA): {len(uptrends)}")
    print(f"  Downtrends (price < MA): {len(downtrends)}")
    
    if downtrends:
        print(f"\n  Downtrend pairs (should NOT generate BUY signals):")
        for r in downtrends:
            print(f"    • {r['pair']}: ${r['price']:.4f} < ${r['ma']:.4f} ({r['diff_pct']:.2f}%)")
    
    # Show data freshness
    if results:
        print(f"\n  DATA FRESHNESS:")
        from datetime import datetime
        now = datetime.now()
        for r in results[:5]:  # Show first 5 pairs
            print(f"    • {r['pair']}: Latest data timestamp shown above")
    
    return results

if __name__ == '__main__':
    config = get_config()
    db_path = Path(config.db_path)
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)
    
    check_ma_status(db_path, ma_period=16)

