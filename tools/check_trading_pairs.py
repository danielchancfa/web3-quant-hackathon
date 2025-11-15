#!/usr/bin/env python3
"""
Diagnostic script to check which pairs each model is trading.
This helps identify if prediction model and MA strategy are trading different pairs.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from inference_service import InferenceService


def check_available_pairs(checkpoint_dir: Path, db_path: Path):
    """Check which pairs have checkpoints (prediction model can trade)."""
    checkpoint_dir = Path(checkpoint_dir)
    available = []
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return available
    
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            # Convert directory name back to pair format (e.g., ADA_USD -> ADA/USD)
            pair_name = item.name.replace('_', '/')
            available.append(pair_name)
    
    return sorted(available)


def check_database_pairs(db_path: Path):
    """Check which pairs have price data in database (MA strategy can trade)."""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT pair FROM horus_prices_1h ORDER BY pair"
    df = pd.read_sql(query, conn)
    conn.close()
    
    return sorted(df['pair'].tolist()) if len(df) > 0 else []


def check_inference_service_pairs(checkpoint_dir: Path, db_path: Path, requested_pairs: list):
    """Check which pairs InferenceService will actually return."""
    try:
        service = InferenceService(
            db_path=db_path,
            checkpoint_dir=checkpoint_dir,
            pairs=requested_pairs,
        )
        results = service.run_once()
        return list(results.keys())
    except Exception as e:
        print(f"❌ Error initializing InferenceService: {e}")
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
    
    config = get_config()
    db_path = Path(config.db_path)
    
    # Parse pairs from argument
    default_pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    
    # Check checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")
        checkpoint_dir = None
    
    print("=" * 80)
    print("TRADING PAIRS DIAGNOSTIC")
    print("=" * 80)
    
    print("\n1️⃣  REQUESTED PAIRS (from --pairs argument):")
    print(f"   {default_pairs}")
    
    if checkpoint_dir:
        print("\n2️⃣  PAIRS WITH CHECKPOINTS (Prediction model can trade):")
        checkpoint_pairs = check_available_pairs(checkpoint_dir, db_path)
        if checkpoint_pairs:
            print(f"   {checkpoint_pairs}")
        else:
            print("   ❌ No checkpoints found!")
        
        print("\n3️⃣  PAIRS INFERENCESERVICE WILL RETURN:")
        inference_pairs = check_inference_service_pairs(checkpoint_dir, db_path, default_pairs)
        if inference_pairs:
            print(f"   {inference_pairs}")
        else:
            print("   ❌ InferenceService failed or returned no pairs")
    else:
        print("\n2️⃣  PAIRS WITH CHECKPOINTS: (skipped - checkpoint_dir not found)")
        print("\n3️⃣  PAIRS INFERENCESERVICE WILL RETURN: (skipped - checkpoint_dir not found)")
        inference_pairs = []
    
    print("\n4️⃣  PAIRS WITH PRICE DATA (MA strategy can trade):")
    db_pairs = check_database_pairs(db_path)
    if db_pairs:
        print(f"   {db_pairs}")
    else:
        print("   ❌ No price data found in database!")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if checkpoint_dir:
        # Compare requested vs available
        missing_checkpoints = [p for p in default_pairs if p not in inference_pairs]
        if missing_checkpoints:
            print(f"\n⚠️  PREDICTION MODEL MISSING:")
            print(f"   These pairs are requested but have no checkpoints:")
            for p in missing_checkpoints:
                print(f"     • {p}")
        
        # Check if MA can trade all requested pairs
        missing_db = [p for p in default_pairs if p not in db_pairs]
        if missing_db:
            print(f"\n⚠️  MA STRATEGY MISSING:")
            print(f"   These pairs are requested but have no price data:")
            for p in missing_db:
                print(f"     • {p}")
        
        # Check alignment
        prediction_only = [p for p in inference_pairs if p not in db_pairs]
        ma_only = [p for p in db_pairs if p not in inference_pairs and p in default_pairs]
        
        if prediction_only:
            print(f"\n⚠️  PREDICTION MODEL ONLY:")
            print(f"   These pairs have checkpoints but no price data (MA can't trade):")
            for p in prediction_only:
                print(f"     • {p}")
        
        if ma_only:
            print(f"\n⚠️  MA STRATEGY ONLY:")
            print(f"   These pairs have price data but no checkpoints (Prediction can't trade):")
            for p in ma_only:
                print(f"     • {p}")
        
        # Aligned pairs
        aligned = [p for p in default_pairs if p in inference_pairs and p in db_pairs]
        if aligned:
            print(f"\n✅ ALIGNED PAIRS (both models can trade):")
            for p in aligned:
                print(f"     • {p}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        
        if len(aligned) < len(default_pairs):
            print(f"\n⚠️  MISMATCH DETECTED!")
            print(f"   • Requested: {len(default_pairs)} pairs")
            print(f"   • Aligned: {len(aligned)} pairs")
            print(f"   • Prediction model will only trade: {inference_pairs}")
            print(f"   • MA strategy will try to trade: {default_pairs}")
            print(f"\n   💡 SOLUTION: Only trade pairs that are in BOTH lists:")
            print(f"      {aligned}")
        else:
            print(f"\n✅ ALL PAIRS ALIGNED!")
            print(f"   Both models can trade all {len(aligned)} requested pairs.")
    else:
        print("\n⚠️  Cannot complete full analysis without checkpoint directory.")
        print("   Please run with --checkpoint_dir argument or ensure model_checkpoints/ exists.")


if __name__ == "__main__":
    main()

