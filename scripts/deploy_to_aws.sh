#!/bin/bash
# Quick deployment script for AWS
# Usage: ./scripts/deploy_to_aws.sh

set -e  # Exit on error

echo "=========================================="
echo "AWS Deployment Script - Simple MA Strategy"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check if we're in the right directory
if [ ! -f "prediction_execution/policy.py" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo -e "${GREEN}✓${NC} Step 1: Checking git status..."
git status --short

# Step 2: Pull latest code
echo ""
echo -e "${GREEN}✓${NC} Step 2: Pulling latest code from GitHub..."
git fetch origin
git pull origin main

# Step 3: Verify configuration
echo ""
echo -e "${GREEN}✓${NC} Step 3: Verifying configuration..."
python3 << 'PYTHON_SCRIPT'
from prediction_execution.policy import NextBarPolicyConfig

cfg = NextBarPolicyConfig()
print(f"Strategy Mode: {cfg.strategy_mode}")
print(f"Simple MA Only: {cfg.use_simple_ma_only}")
print(f"Stop Loss: {cfg.fixed_stop_pct * 100}%")
print(f"Take Profit: {cfg.fixed_target_pct * 100}%")

if cfg.use_simple_ma_only and cfg.fixed_stop_pct == 0.01 and cfg.fixed_target_pct == 0.01:
    print("\n✅ Configuration is correct!")
else:
    print("\n⚠️  Warning: Configuration may not match expected values")
PYTHON_SCRIPT

# Step 4: Check dependencies
echo ""
echo -e "${GREEN}✓${NC} Step 4: Checking dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Requirements file found. Run 'pip install -r requirements.txt' if needed."
else
    echo -e "${YELLOW}⚠ Warning: requirements.txt not found${NC}"
fi

# Step 5: Find running trading processes
echo ""
echo -e "${GREEN}✓${NC} Step 5: Checking for running trading processes..."
TRADING_PIDS=$(ps aux | grep -E "scheduler|run_execution|python.*trading" | grep -v grep | awk '{print $2}' || true)

if [ -z "$TRADING_PIDS" ]; then
    echo "No running trading processes found."
else
    echo -e "${YELLOW}Found running processes:${NC}"
    ps aux | grep -E "scheduler|run_execution|python.*trading" | grep -v grep
    echo ""
    read -p "Do you want to stop these processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping processes..."
        echo "$TRADING_PIDS" | xargs kill || true
        sleep 2
        echo "Processes stopped."
    fi
fi

# Step 6: Deployment complete
echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Restart your trading system:"
echo "   nohup python3 scheduler.py --checkpoint_dir <path> --pairs <pairs> > trading.log 2>&1 &"
echo ""
echo "2. Monitor logs:"
echo "   tail -f trading.log"
echo ""
echo "3. Check process:"
echo "   ps aux | grep scheduler"
echo ""

