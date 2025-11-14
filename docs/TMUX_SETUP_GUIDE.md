# Complete tmux Setup Guide - Trading Scheduler

## Quick Start

After successfully pulling code from GitHub, follow these steps:

## Step 1: Create/Attach to tmux Session

```bash
# Check if tmux session already exists
tmux ls

# If 'trading' session exists, attach to it:
tmux attach -t trading

# If it doesn't exist, create new session:
tmux new -s trading
```

You're now in tmux (you'll see a green bar at the bottom).

## Step 2: Navigate to Project Directory

```bash
cd ~/projects/Web3\ Quant\ Hackathon

# Verify you're in the right place
pwd

# Verify you have the latest code
git log --oneline -1
# Should show: a009a96 Add start_trading.sh script for easy deployment
```

## Step 3: Verify Configuration

```bash
# Check that simple MA strategy is configured
python3 -c "from prediction_execution.policy import NextBarPolicyConfig; cfg = NextBarPolicyConfig(); print(f'Strategy: {cfg.strategy_mode}, MA Only: {cfg.use_simple_ma_only}, Stop: {cfg.fixed_stop_pct}, Target: {cfg.fixed_target_pct}')"
```

**Expected output:**
```
Strategy: technical, MA Only: True, Stop: 0.01, Target: 0.01
```

## Step 4: Start Trading Scheduler

### Option A: Use Helper Script (Recommended)

```bash
chmod +x scripts/start_trading.sh
./scripts/start_trading.sh
```

### Option B: Run Manually

```bash
python3 scheduler.py \
  --checkpoint_dir model_checkpoints/ADA_USD \
  --pairs 'ADA/USD,BTC/USD,ETH/USD,BNB/USD,LINK/USD,SOL/USD' \
  --loop_interval 3600 \
  --position_limit 0.3
```

**⚠️ IMPORTANT: NO `--paper` flag = REAL TRADES!**

## Step 5: Detach from tmux (Keeps Running)

Press these keys in sequence:
1. Press: `Ctrl+B`
2. Release `Ctrl+B`
3. Press: `D`

You'll see: `[detached (from session trading)]`

The scheduler keeps running in the background!

## Step 6: Verify It's Running

```bash
# Check if process is running
ps aux | grep scheduler

# You should see a python3 scheduler.py process
```

## Monitoring

### View Live Output

```bash
# Reattach to tmux to see live output
tmux attach -t trading
```

You'll see:
- Portfolio value updates
- Policy decisions for each pair
- Trade execution messages
- Any errors

### Check Logs

```bash
# If logging to file
tail -f trading.log

# OR if using nohup
tail -f nohup.out
```

## Management

### Stop Trading

```bash
# Attach to tmux
tmux attach -t trading

# Stop the process
Ctrl+C

# OR kill from outside tmux
pkill -f scheduler.py
```

### Restart Trading

Follow steps 1-4 again.

## What Happens Next

✅ The scheduler will:
- Run continuously in the background
- Execute trades every hour (3600 seconds)
- Trade all 6 pairs: ADA, BTC, ETH, BNB, LINK, SOL
- Use simple MA crossover strategy
- Apply 1% stop loss and 1% take profit
- Make REAL TRADES (not paper trading)

⏰ **Timing:**
- First execution: Immediately when started
- Subsequent executions: Every hour after that
- Example: Start at 2:00 PM → runs at 2:00, 3:00, 4:00, etc.

📈 **What to Expect:**
- Logs showing portfolio value
- Policy decisions for each pair
- Trade signals based on MA crossovers
- Actual trades executed on the exchange
- Position updates

## Troubleshooting

### If scheduler stops:
```bash
# Check for errors
tmux attach -t trading
# Look for error messages

# Check if checkpoint directory exists
ls -la model_checkpoints/ADA_USD
```

### If no trades are happening:
- Check policy decisions in logs
- Look for 'Policy decision:' messages
- Verify MA crossover signals are being generated
- Check if price history is available

### If you see errors:
```bash
# Check Python dependencies
pip install -r requirements.txt

# Check database connection
# Verify API credentials are set
```

## Quick Reference

### Complete Setup (Copy-Paste)

```bash
tmux new -s trading
cd ~/projects/Web3\ Quant\ Hackathon
python3 -c "from prediction_execution.policy import NextBarPolicyConfig; cfg = NextBarPolicyConfig(); print(f'MA Only: {cfg.use_simple_ma_only}, Stop: {cfg.fixed_stop_pct}, Target: {cfg.fixed_target_pct}')"
python3 scheduler.py \
  --checkpoint_dir model_checkpoints/ADA_USD \
  --pairs 'ADA/USD,BTC/USD,ETH/USD,BNB/USD,LINK/USD,SOL/USD' \
  --loop_interval 3600 \
  --position_limit 0.3
# Then: Ctrl+B, D to detach
```

### Useful tmux Commands

```bash
# List all sessions
tmux ls

# Attach to session
tmux attach -t trading

# Create new session
tmux new -s trading

# Kill session
tmux kill-session -t trading

# Detach from session (while inside)
Ctrl+B, then D

# Scroll in tmux
Ctrl+B, then [ (enter copy mode)
# Use arrow keys to scroll
# Press Q to exit
```

## Important Reminders

- ✅ NO `--paper` flag = REAL TRADES
- ✅ `loop_interval 3600` = runs every hour
- ✅ All 6 pairs will be traded
- ✅ Simple MA strategy is active
- ✅ 1% stop, 1% target

