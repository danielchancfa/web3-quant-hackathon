# Commands to Check Trading Pairs on AWS

Use these commands to check which pairs each model is actually trading on AWS.

## 1. Check Which Pairs Have Checkpoints (Prediction Model)

```bash
# List all checkpoint directories
ls -la model_checkpoints/

# Count how many pairs have checkpoints
ls -d model_checkpoints/*/ 2>/dev/null | wc -l

# List only pair directories (filter out non-pair dirs)
ls -d model_checkpoints/*/ 2>/dev/null | grep -v "final\|next\|run" | sed 's|model_checkpoints/||' | sed 's|/$||' | sed 's|_|/|g'
```

## 2. Check Which Pairs Have Price Data (MA Strategy)

```bash
# First, find the database path from config
python3 -c "from config import get_config; import json; print(json.dumps(get_config(), indent=2))" | grep db_path

# Or check config.json directly
cat config.json | grep db_path

# Then check which pairs have price data
# Replace <db_path> with actual path from config
sqlite3 <db_path> "SELECT DISTINCT pair FROM horus_prices_1h ORDER BY pair;"

# Count pairs with data
sqlite3 <db_path> "SELECT COUNT(DISTINCT pair) FROM horus_prices_1h;"
```

## 3. Check Scheduler Logs (What's Actually Being Traded)

```bash
# If running in tmux, check the logs
# First attach to tmux session
tmux attach

# Then check recent logs (if logging to file)
tail -n 100 scheduler.log

# Or check stdout if running in foreground
# Look for these diagnostic messages:
# - "🔍 DIAGNOSTIC: Prediction model pairs:"
# - "🔍 DIAGNOSTIC: Requested pairs:"
# - "🔍 DIAGNOSTIC: MA strategy will try to trade:"
# - "⚠️  Pairs in --pairs but NOT in prediction results:"
```

## 4. Check Recent Trades/Positions

```bash
# Check current positions (if position manager logs them)
# Look in scheduler logs for:
# - "Pair notionals:"
# - "Asset quantities:"

# Or check the database for recent trades (if stored)
sqlite3 <db_path> "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20;" 2>/dev/null || echo "No trades table found"
```

## 5. Check What Pairs InferenceService Actually Loads

The InferenceService loads models for pairs passed to it. But if you want to see what it's actually using:

```bash
# Check the scheduler command that's running
ps aux | grep scheduler.py

# Look for the --pairs argument to see what was requested
# Also check if InferenceService is loading all checkpoints or just requested ones
```

## 6. Quick Diagnostic Script (Run on AWS)

Create a file `check_pairs_aws.sh`:

```bash
#!/bin/bash

echo "=========================================="
echo "TRADING PAIRS DIAGNOSTIC ON AWS"
echo "=========================================="

echo ""
echo "1. CHECKPOINTS (Prediction Model):"
ls -d model_checkpoints/*/ 2>/dev/null | grep -v "final\|next\|run" | sed 's|model_checkpoints/||' | sed 's|/$||' | sed 's|_|/|g' | sort

echo ""
echo "2. PRICE DATA (MA Strategy):"
DB_PATH=$(python3 -c "from config import get_config; print(get_config().db_path)" 2>/dev/null || echo "config not found")
if [ -f "$DB_PATH" ]; then
    sqlite3 "$DB_PATH" "SELECT DISTINCT pair FROM horus_prices_1h ORDER BY pair;" 2>/dev/null || echo "Error querying database"
else
    echo "Database not found at: $DB_PATH"
fi

echo ""
echo "3. SCHEDULER PROCESS:"
ps aux | grep "scheduler.py" | grep -v grep

echo ""
echo "4. RECENT LOGS (if available):"
if [ -f "scheduler.log" ]; then
    echo "Last 20 lines with DIAGNOSTIC:"
    grep -i "diagnostic\|pairs\|trading" scheduler.log | tail -20
else
    echo "No scheduler.log file found"
fi
```

Then run:
```bash
chmod +x check_pairs_aws.sh
./check_pairs_aws.sh
```

## 7. Check if ZEC/USD is Being Traded

```bash
# Check if ZEC/USD has checkpoint
ls -d model_checkpoints/ZEC_USD/ 2>/dev/null && echo "✅ ZEC/USD has checkpoint" || echo "❌ ZEC/USD no checkpoint"

# Check if ZEC/USD has price data
sqlite3 <db_path> "SELECT COUNT(*) FROM horus_prices_1h WHERE pair='ZEC/USD';" 2>/dev/null

# Check scheduler logs for ZEC/USD
grep -i "ZEC/USD" scheduler.log | tail -10
```

## Understanding the Results

- **If prediction model has more pairs than requested**: The InferenceService might be loading all checkpoints it finds, not just requested ones. Check the `InferenceService` initialization in `scheduler.py`.

- **If MA strategy has different pairs**: The database might have price data for pairs that don't have checkpoints, or vice versa.

- **If both models trade different pairs**: This is the mismatch we need to fix by aligning them.

