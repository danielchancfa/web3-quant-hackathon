#!/bin/bash

echo "=========================================="
echo "CHECKING ACTUAL TRADING PAIRS ON AWS"
echo "=========================================="

echo ""
echo "1. SCHEDULER PROCESS (what --pairs argument is used):"
SCHEDULER_CMD=$(ps aux | grep "[p]ython.*scheduler.py" | head -1)
if [ -z "$SCHEDULER_CMD" ]; then
    echo "   ❌ Scheduler not running"
else
    echo "$SCHEDULER_CMD" | awk '{for(i=1;i<=NF;i++) if($i=="--pairs") {print "   --pairs:", $(i+1); break}}'
    echo ""
    echo "   Full command:"
    echo "$SCHEDULER_CMD" | sed 's/^/   /'
fi

echo ""
echo "2. PREDICTION MODEL PAIRS (from logs - what InferenceService loaded):"
if [ -f "scheduler.log" ]; then
    PRED_PAIRS=$(grep "DIAGNOSTIC.*Prediction model pairs" scheduler.log | tail -1 | sed 's/.*\[\(.*\)\].*/\1/' | tr -d "'" | tr ',' '\n' | sed 's/^/   • /')
    if [ -n "$PRED_PAIRS" ]; then
        echo "$PRED_PAIRS"
    else
        echo "   ⚠️  No diagnostic message found in logs"
    fi
elif [ -f "logs/scheduler.log" ]; then
    PRED_PAIRS=$(grep "DIAGNOSTIC.*Prediction model pairs" logs/scheduler.log | tail -1 | sed 's/.*\[\(.*\)\].*/\1/' | tr -d "'" | tr ',' '\n' | sed 's/^/   • /')
    if [ -n "$PRED_PAIRS" ]; then
        echo "$PRED_PAIRS"
    else
        echo "   ⚠️  No diagnostic message found in logs"
    fi
else
    echo "   ⚠️  No log file found"
fi

echo ""
echo "3. REQUESTED PAIRS (from logs - what --pairs argument was):"
if [ -f "scheduler.log" ]; then
    REQ_PAIRS=$(grep "DIAGNOSTIC.*Requested pairs" scheduler.log | tail -1 | sed 's/.*\[\(.*\)\].*/\1/' | tr -d "'" | tr ',' '\n' | sed 's/^/   • /')
    if [ -n "$REQ_PAIRS" ]; then
        echo "$REQ_PAIRS"
    else
        echo "   ⚠️  No diagnostic message found in logs"
    fi
elif [ -f "logs/scheduler.log" ]; then
    REQ_PAIRS=$(grep "DIAGNOSTIC.*Requested pairs" logs/scheduler.log | tail -1 | sed 's/.*\[\(.*\)\].*/\1/' | tr -d "'" | tr ',' '\n' | sed 's/^/   • /')
    if [ -n "$REQ_PAIRS" ]; then
        echo "$REQ_PAIRS"
    else
        echo "   ⚠️  No diagnostic message found in logs"
    fi
else
    echo "   ⚠️  No log file found"
fi

echo ""
echo "4. RECENTLY TRADED PAIRS (from logs - actual trades):"
if [ -f "scheduler.log" ]; then
    TRADED=$(grep -E '\[.*/USD\].*(buy|sell|BUY|SELL)' scheduler.log | tail -20 | grep -oE '\[[^]]+\]' | sort -u | sed 's/\[/   • /' | sed 's/\]//')
    if [ -n "$TRADED" ]; then
        echo "$TRADED"
    else
        echo "   ⚠️  No trades found in recent logs"
    fi
elif [ -f "logs/scheduler.log" ]; then
    TRADED=$(grep -E '\[.*/USD\].*(buy|sell|BUY|SELL)' logs/scheduler.log | tail -20 | grep -oE '\[[^]]+\]' | sort -u | sed 's/\[/   • /' | sed 's/\]//')
    if [ -n "$TRADED" ]; then
        echo "$TRADED"
    else
        echo "   ⚠️  No trades found in recent logs"
    fi
else
    echo "   ⚠️  No log file found"
fi

echo ""
echo "5. CHECK FOR ZEC/USD SPECIFICALLY:"
if [ -f "scheduler.log" ]; then
    ZEC_COUNT=$(grep -i "ZEC/USD" scheduler.log | wc -l)
    if [ "$ZEC_COUNT" -gt 0 ]; then
        echo "   ✅ Found $ZEC_COUNT mentions of ZEC/USD in logs"
        echo "   Recent activity:"
        grep -i "ZEC/USD" scheduler.log | tail -5 | sed 's/^/      /'
    else
        echo "   ❌ No ZEC/USD found in logs"
    fi
elif [ -f "logs/scheduler.log" ]; then
    ZEC_COUNT=$(grep -i "ZEC/USD" logs/scheduler.log | wc -l)
    if [ "$ZEC_COUNT" -gt 0 ]; then
        echo "   ✅ Found $ZEC_COUNT mentions of ZEC/USD in logs"
        echo "   Recent activity:"
        grep -i "ZEC/USD" logs/scheduler.log | tail -5 | sed 's/^/      /'
    else
        echo "   ❌ No ZEC/USD found in logs"
    fi
else
    echo "   ⚠️  No log file found"
fi

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "• Checkpoints available: 13 pairs"
echo "• Prediction model CAN trade: All 13 pairs (if requested)"
echo "• Prediction model IS trading: Only pairs in --pairs argument"
echo ""
echo "To see what's actually being traded, check section 4 above."

