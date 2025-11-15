#!/usr/bin/env python3
"""
Simple diagnostic to check which pairs each model can trade.
Doesn't require loading models - just checks filesystem and database.
"""

import sys
from pathlib import Path
import sqlite3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import get_config
    config = get_config()
    db_path = Path(config.db_path)
except:
    print("⚠️  Could not load config. Using default database path.")
    db_path = Path("horus.db")


def check_checkpoint_pairs(checkpoint_dir: Path):
    """Check which pairs have checkpoints."""
    if not checkpoint_dir.exists():
        return []
    
    pairs = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            # Convert directory name to pair format (e.g., ADA_USD -> ADA/USD)
            pair_name = item.name.replace('_', '/')
            pairs.append(pair_name)
    
    return sorted(pairs)


def check_database_pairs(db_path: Path):
    """Check which pairs have price data in database."""
    if not db_path.exists():
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pair FROM horus_prices_1h ORDER BY pair")
        pairs = [row[0] for row in cursor.fetchall()]
        conn.close()
        return pairs
    except Exception as e:
        print(f"⚠️  Error querying database: {e}")
        return []


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check which pairs each model can trade")
    parser.add_argument('--checkpoint_dir', type=str, default='model_checkpoints',
                        help='Directory with model checkpoints')
    parser.add_argument('--pairs', type=str, 
                        default='ADA/USD,BTC/USD,ETH/USD,BNB/USD,LINK/USD,SOL/USD',
                        help='Comma-separated pairs to check')
    args = parser.parse_args()
    
    requested_pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    checkpoint_dir = Path(args.checkpoint_dir)
    
    print("=" * 80)
    print("TRADING PAIRS DIAGNOSTIC")
    print("=" * 80)
    
    print(f"\n1️⃣  REQUESTED PAIRS (from --pairs):")
    for p in requested_pairs:
        print(f"   • {p}")
    
    print(f"\n2️⃣  PAIRS WITH CHECKPOINTS (Prediction model can trade):")
    checkpoint_pairs = check_checkpoint_pairs(checkpoint_dir)
    if checkpoint_pairs:
        for p in checkpoint_pairs:
            print(f"   • {p}")
    else:
        print(f"   ❌ No checkpoints found in {checkpoint_dir}")
    
    print(f"\n3️⃣  PAIRS WITH PRICE DATA (MA strategy can trade):")
    db_pairs = check_database_pairs(db_path)
    if db_pairs:
        for p in db_pairs:
            print(f"   • {p}")
    else:
        print(f"   ❌ No price data found in {db_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Find mismatches
    prediction_only = [p for p in checkpoint_pairs if p not in db_pairs and p in requested_pairs]
    ma_only = [p for p in db_pairs if p not in checkpoint_pairs and p in requested_pairs]
    missing_checkpoints = [p for p in requested_pairs if p not in checkpoint_pairs]
    missing_db = [p for p in requested_pairs if p not in db_pairs]
    aligned = [p for p in requested_pairs if p in checkpoint_pairs and p in db_pairs]
    
    if missing_checkpoints:
        print(f"\n⚠️  PREDICTION MODEL CANNOT TRADE ({len(missing_checkpoints)} pairs):")
        for p in missing_checkpoints:
            print(f"   • {p} (no checkpoint)")
    
    if missing_db:
        print(f"\n⚠️  MA STRATEGY CANNOT TRADE ({len(missing_db)} pairs):")
        for p in missing_db:
            print(f"   • {p} (no price data)")
    
    if prediction_only:
        print(f"\n⚠️  PREDICTION ONLY ({len(prediction_only)} pairs):")
        print(f"   Has checkpoint but no price data (MA can't trade):")
        for p in prediction_only:
            print(f"   • {p}")
    
    if ma_only:
        print(f"\n⚠️  MA STRATEGY ONLY ({len(ma_only)} pairs):")
        print(f"   Has price data but no checkpoint (Prediction can't trade):")
        for p in ma_only:
            print(f"   • {p}")
    
    if aligned:
        print(f"\n✅ ALIGNED PAIRS - BOTH MODELS CAN TRADE ({len(aligned)} pairs):")
        for p in aligned:
            print(f"   • {p}")
    
    print("\n" + "=" * 80)
    print("CURRENT BEHAVIOR IN SCHEDULER")
    print("=" * 80)
    
    print(f"\n📊 What will happen when scheduler runs:")
    print(f"   • Loops through: {requested_pairs}")
    print(f"   • Prediction model returns: {checkpoint_pairs}")
    print(f"   • MA strategy tries: {requested_pairs}")
    
    if len(aligned) < len(requested_pairs):
        print(f"\n⚠️  MISMATCH DETECTED!")
        print(f"   • {len(requested_pairs)} pairs requested")
        print(f"   • {len(aligned)} pairs aligned (both models can trade)")
        print(f"   • Prediction model will skip: {missing_checkpoints}")
        print(f"   • MA strategy will try to trade: {requested_pairs}")
        print(f"\n   💡 ISSUE: Models trading different pairs!")
        print(f"   💡 SOLUTION: Only trade aligned pairs: {aligned}")
    else:
        print(f"\n✅ ALL PAIRS ALIGNED!")
        print(f"   Both models can trade all {len(aligned)} requested pairs.")


if __name__ == "__main__":
    main()

