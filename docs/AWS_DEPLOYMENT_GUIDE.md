# AWS Deployment Guide - Simple MA Crossover Strategy

## Overview
This guide walks you through deploying the new simple MA crossover strategy to your AWS instance.

## Strategy Configuration
- **Strategy**: Simple MA Crossover (10/20 periods)
- **Stop Loss**: 1%
- **Take Profit**: 1%
- **Risk per Trade**: 5% of portfolio
- **Max Risk per Pair**: 20%

## Pre-Deployment Checklist
- [ ] AWS instance is running
- [ ] SSH access to AWS instance
- [ ] Git repository is accessible
- [ ] Trading system is currently running (to restart after update)

## Step-by-Step Deployment

### Step 1: Connect to AWS Instance via Session Manager

**Using AWS Console:**
1. Go to AWS Console → EC2 → Instances
2. Select your instance
3. Click "Connect" button
4. Choose "Session Manager" tab
5. Click "Connect"

**Using AWS CLI (if configured):**
```bash
aws ssm start-session --target i-xxxxxxxxxxxxx
```

**Note:** Once connected via Session Manager, you'll have a terminal session. All commands below work the same as SSH.

### Step 2: Navigate to Project Directory

```bash
# Navigate to your project directory
cd ~/projects/Web3\ Quant\ Hackathon
# OR wherever your project is located
cd /path/to/your/project
```

### Step 3: Check Current Status

```bash
# Check current git status
git status

# Check current branch
git branch

# View recent commits
git log --oneline -5
```

### Step 4: Pull Latest Code from GitHub

```bash
# Fetch latest changes
git fetch origin

# Pull latest code from main branch
git pull origin main

# Verify the latest commit
git log --oneline -1
# Should show: "Implement simple MA crossover strategy for competition"
```

### Step 5: Verify Configuration

```bash
# Check the policy configuration file
cat prediction_execution/policy.py | grep -A 5 "use_simple_ma_only"
# Should show: use_simple_ma_only: bool = True

# Verify stop/target settings
cat prediction_execution/policy.py | grep -A 2 "fixed_stop_pct"
# Should show: fixed_stop_pct: float = 0.01
# Should show: fixed_target_pct: float = 0.01
```

### Step 6: Check Dependencies

```bash
# Verify Python dependencies are installed
pip list | grep -E "pandas|numpy|backtesting"

# If needed, install/update dependencies
pip install -r requirements.txt
```

### Step 7: Stop Current Trading System (if running)

```bash
# Find running Python processes related to trading
ps aux | grep -E "scheduler|run_execution|python.*trading"

# Stop the process (replace PID with actual process ID)
kill <PID>

# OR if using systemd/service
sudo systemctl stop trading-bot
# OR
pm2 stop trading-bot
```

### Step 8: Test Configuration (Optional but Recommended)

```bash
# Test that the policy can be imported
python3 -c "from prediction_execution.policy import NextBarPolicyConfig; cfg = NextBarPolicyConfig(); print(f'MA Strategy: {cfg.use_simple_ma_only}, Stop: {cfg.fixed_stop_pct}, Target: {cfg.fixed_target_pct}')"

# Expected output:
# MA Strategy: True, Stop: 0.01, Target: 0.01
```

### Step 9: Restart Trading System

**Option A: If using a scheduler script directly:**
```bash
# Navigate to project root
cd ~/projects/Web3\ Quant\ Hackathon

# Run scheduler (adjust paths as needed)
python3 scheduler.py \
  --checkpoint_dir model_checkpoints/ADA_USD \
  --pairs "ADA/USD,BTC/USD,ETH/USD" \
  --loop_interval 3600 \
  --position_limit 0.3

# Run in background with nohup
nohup python3 scheduler.py \
  --checkpoint_dir model_checkpoints/ADA_USD \
  --pairs "ADA/USD,BTC/USD,ETH/USD" \
  --loop_interval 3600 \
  --position_limit 0.3 \
  > trading.log 2>&1 &
```

**Option B: If using systemd service:**
```bash
# Edit service file if needed
sudo nano /etc/systemd/system/trading-bot.service

# Reload systemd
sudo systemctl daemon-reload

# Start service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# Enable auto-start on boot
sudo systemctl enable trading-bot
```

**Option C: If using PM2:**
```bash
# Start with PM2
pm2 start scheduler.py --name trading-bot --interpreter python3 -- \
  --checkpoint_dir model_checkpoints/ADA_USD \
  --pairs "ADA/USD,BTC/USD,ETH/USD" \
  --loop_interval 3600 \
  --position_limit 0.3

# Save PM2 configuration
pm2 save

# Check status
pm2 status
pm2 logs trading-bot
```

**Option D: If using screen/tmux:**
```bash
# Start new screen session
screen -S trading

# Run scheduler
python3 scheduler.py \
  --checkpoint_dir model_checkpoints/ADA_USD \
  --pairs "ADA/USD,BTC/USD,ETH/USD" \
  --loop_interval 3600 \
  --position_limit 0.3

# Detach: Ctrl+A, then D
# Reattach: screen -r trading
```

### Step 10: Monitor Deployment

```bash
# Monitor logs in real-time
tail -f trading.log
# OR
pm2 logs trading-bot
# OR
journalctl -u trading-bot -f

# Check for errors
grep -i error trading.log
grep -i "MA Strategy\|simple_ma\|crossover" trading.log

# Verify process is running
ps aux | grep scheduler
```

### Step 11: Verify Strategy is Active

```bash
# Check logs for strategy confirmation
grep -i "technical\|MA\|crossover" trading.log | tail -20

# Look for policy decisions in logs
grep -i "policy decision\|action\|notional" trading.log | tail -20
```

## Troubleshooting

### Issue: Git pull fails
```bash
# Check git remote
git remote -v

# If needed, set remote
git remote set-url origin https://github.com/danielchancfa/web3-quant-hackathon.git

# Try pull again
git pull origin main
```

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python path
python3 -c "import sys; print(sys.path)"
```

### Issue: Strategy not working
```bash
# Verify configuration is loaded
python3 -c "
from prediction_execution.policy import NextBarPolicyConfig
cfg = NextBarPolicyConfig()
print('Config loaded:')
print(f'  Strategy Mode: {cfg.strategy_mode}')
print(f'  Simple MA Only: {cfg.use_simple_ma_only}')
print(f'  Stop: {cfg.fixed_stop_pct}')
print(f'  Target: {cfg.fixed_target_pct}')
"
```

### Issue: Price history not available
```bash
# Check if database has price data
python3 -c "
import sqlite3
from pathlib import Path
from config import get_config

config = get_config()
db_path = Path(config.db_path)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check for price data
cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"')
tables = cursor.fetchall()
print('Available tables:', tables)

conn.close()
"
```

## Rollback Plan

If something goes wrong, you can rollback:

```bash
# View commit history
git log --oneline -10

# Rollback to previous commit (replace with actual commit hash)
git reset --hard <previous-commit-hash>

# OR revert the last commit
git revert HEAD

# Restart trading system
# (Follow Step 9 above)
```

## Post-Deployment Verification

After deployment, verify:

1. **Configuration is correct:**
   - `use_simple_ma_only = True`
   - `fixed_stop_pct = 0.01`
   - `fixed_target_pct = 0.01`

2. **System is running:**
   - Process is active
   - No errors in logs
   - Trades are being executed (if applicable)

3. **Strategy is active:**
   - Logs show technical indicator calculations
   - Policy decisions reflect MA crossover logic

## Monitoring Commands

```bash
# Watch logs continuously
tail -f trading.log

# Check system resources
htop
# OR
top

# Check disk space
df -h

# Check network connectivity
ping -c 3 mock-api.roostoo.com
```

## Support

If you encounter issues:
1. Check logs for error messages
2. Verify configuration matches expected values
3. Ensure all dependencies are installed
4. Check that price history is available (if using technical indicators)

## Quick Reference

```bash
# Quick deployment script (save as deploy.sh)
#!/bin/bash
cd ~/projects/Web3\ Quant\ Hackathon
git pull origin main
python3 -c "from prediction_execution.policy import NextBarPolicyConfig; print('Config OK')"
# Restart your trading system here
echo "Deployment complete!"
```

