#!/bin/bash

echo "=========================================="
echo "TRADING PAIRS DIAGNOSTIC ON AWS"
echo "=========================================="

echo ""
echo "1. CHECKPOINTS (Prediction Model Can Trade):"
CHECKPOINTS=$(ls -d model_checkpoints/*/ 2>/dev/null | grep -v "final\|next\|run" | sed 's|model_checkpoints/||' | sed 's|/$||' | sed 's|_|/|g' | sort)
if [ -z "$CHECKPOINTS" ]; then
    echo "   ❌ No checkpoints found"
else
    echo "$CHECKPOINTS" | while read pair; do
        echo "   • $pair"
    done
    echo ""
    echo "   Total: $(echo "$CHECKPOINTS" | wc -l) pairs"
fi

echo ""
echo "2. PRICE DATA (MA Strategy Can Trade):"
DB_PATH=$(python3 -c "from config import get_config; print(get_config().db_path)" 2>/dev/null || echo "")
if [ -z "$DB_PATH" ] || [ ! -f "$DB_PATH" ]; then
    echo "   ⚠️  Database path not found or file doesn't exist"
    echo "   Trying to find config.json..."
    if [ -f "config.json" ]; then
        DB_PATH=$(python3 -c "import json; print(json.load(open('config.json'))['db_path'])" 2>/dev/null || echo "")
    fi
fi

if [ -n "$DB_PATH" ] && [ -f "$DB_PATH" ]; then
    PAIRS=$(sqlite3 "$DB_PATH" "SELECT DISTINCT pair FROM horus_prices_1h ORDER BY pair;" 2>/dev/null)
    if [ -z "$PAIRS" ]; then
        echo "   ❌ No price data found or table doesn't exist"
    else
        echo "$PAIRS" | while read pair; do
            echo "   • $pair"
        done
        echo ""
        echo "   Total: $(echo "$PAIRS" | wc -l) pairs"
    fi
else
    echo "   ❌ Database not found at: $DB_PATH"
fi

echo ""
echo "3. SCHEDULER PROCESS:"
SCHEDULER=$(ps aux | grep "[p]ython.*scheduler.py" | head -1)
if [ -z "$SCHEDULER" ]; then
    echo "   ❌ Scheduler not running"
else
    echo "   ✅ Scheduler is running:"
    echo "$SCHEDULER" | awk '{print "   " $0}'
    echo ""
    # Extract --pairs argument
    PAIRS_ARG=$(echo "$SCHEDULER" | grep -oP '--pairs \S+' | cut -d' ' -f2)
    if [ -n "$PAIRS_ARG" ]; then
        echo "   Requested pairs: $PAIRS_ARG"
    fi
fi

echo ""
echo "4. RECENT TRADES/ACTIVITY:"
if [ -f "scheduler.log" ]; then
    echo "   Last 10 lines mentioning pairs:"
    grep -iE "\[.*/USD\]|trading pairs|diagnostic.*pairs" scheduler.log | tail -10 | sed 's/^/   /'
elif [ -f "logs/scheduler.log" ]; then
    echo "   Last 10 lines mentioning pairs:"
    grep -iE "\[.*/USD\]|trading pairs|diagnostic.*pairs" logs/scheduler.log | tail -10 | sed 's/^/   /'
else
    echo "   ⚠️  No log file found (check if logging to file)"
fi

echo ""
echo "5. CHECKING ZEC/USD SPECIFICALLY:"
if [ -d "model_checkpoints/ZEC_USD" ]; then
    echo "   ✅ ZEC/USD has checkpoint"
else
    echo "   ❌ ZEC/USD no checkpoint"
fi

if [ -n "$DB_PATH" ] && [ -f "$DB_PATH" ]; then
    ZEC_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM horus_prices_1h WHERE pair='ZEC/USD';" 2>/dev/null || echo "0")
    if [ "$ZEC_COUNT" -gt 0 ]; then
        echo "   ✅ ZEC/USD has $ZEC_COUNT price records"
    else
        echo "   ❌ ZEC/USD no price data"
    fi
fi

echo ""
echo "=========================================="
echo "ANALYSIS"
echo "=========================================="

# Compare checkpoints and database pairs
if [ -n "$CHECKPOINTS" ] && [ -n "$PAIRS" ] && [ -f "$DB_PATH" ]; then
    echo ""
    echo "Pairs with BOTH checkpoint AND price data (aligned):"
    comm -12 <(echo "$CHECKPOINTS" | sort) <(echo "$PAIRS" | sort) | while read p; do
        echo "   ✅ $p"
    done
    
    echo ""
    echo "Pairs with checkpoint ONLY (prediction can trade, MA cannot):"
    comm -23 <(echo "$CHECKPOINTS" | sort) <(echo "$PAIRS" | sort) | while read p; do
        echo "   ⚠️  $p"
    done
    
    echo ""
    echo "Pairs with price data ONLY (MA can trade, prediction cannot):"
    comm -13 <(echo "$CHECKPOINTS" | sort) <(echo "$PAIRS" | sort) | while read p; do
        echo "   ⚠️  $p"
    done
fi

