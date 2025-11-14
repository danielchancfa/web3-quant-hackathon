#!/bin/bash
# Start trading scheduler for all pairs
# Usage: ./scripts/start_trading.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Starting Trading Scheduler"
echo "=========================================="

# Navigate to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

echo -e "${GREEN}✓${NC} Project directory: $PROJECT_DIR"

# Check if already running
if pgrep -f "scheduler.py" > /dev/null; then
    echo -e "${YELLOW}⚠ Warning: Scheduler appears to be already running${NC}"
    echo "Processes:"
    ps aux | grep scheduler.py | grep -v grep
    read -p "Do you want to stop and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing processes..."
        pkill -f scheduler.py || true
        sleep 2
    else
        echo "Exiting. Please stop the existing process first."
        exit 1
    fi
fi

# Verify configuration
echo -e "${GREEN}✓${NC} Verifying configuration..."
python3 << 'PYTHON_SCRIPT'
from prediction_execution.policy import NextBarPolicyConfig

cfg = NextBarPolicyConfig()
print(f"Strategy Mode: {cfg.strategy_mode}")
print(f"Simple MA Only: {cfg.use_simple_ma_only}")
print(f"Stop Loss: {cfg.fixed_stop_pct * 100}%")
print(f"Take Profit: {cfg.fixed_target_pct * 100}%")

if cfg.use_simple_ma_only:
    print("\n✅ Configuration is correct for simple MA strategy!")
else:
    print("\n⚠️  Warning: Configuration may not match expected values")
PYTHON_SCRIPT

# Default values
CHECKPOINT_DIR="${CHECKPOINT_DIR:-model_checkpoints/ADA_USD}"
PAIRS="${PAIRS:-ADA/USD,BTC/USD,ETH/USD,BNB/USD,LINK/USD,SOL/USD}"
LOOP_INTERVAL="${LOOP_INTERVAL:-3600}"
POSITION_LIMIT="${POSITION_LIMIT:-0.3}"

# Allow override via environment variables or command line
if [ "$1" != "" ]; then
    CHECKPOINT_DIR="$1"
fi
if [ "$2" != "" ]; then
    PAIRS="$2"
fi

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Pairs: $PAIRS"
echo "  Loop Interval: $LOOP_INTERVAL seconds (1 hour)"
echo "  Position Limit: $POSITION_LIMIT"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${YELLOW}⚠ Warning: Checkpoint directory not found: $CHECKPOINT_DIR${NC}"
    echo "Available checkpoints:"
    ls -d model_checkpoints/*/ 2>/dev/null | head -5 || echo "  (none found)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start scheduler
echo ""
echo -e "${GREEN}Starting scheduler...${NC}"
echo "Logs will be written to: trading.log"
echo ""
echo "Command:"
echo "python3 scheduler.py \\"
echo "  --checkpoint_dir $CHECKPOINT_DIR \\"
echo "  --pairs '$PAIRS' \\"
echo "  --loop_interval $LOOP_INTERVAL \\"
echo "  --position_limit $POSITION_LIMIT"
echo ""

# Run scheduler (no --paper flag = real trades!)
python3 scheduler.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --pairs "$PAIRS" \
  --loop_interval "$LOOP_INTERVAL" \
  --position_limit "$POSITION_LIMIT"

