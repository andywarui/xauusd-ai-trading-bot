"""
Enhanced Backtesting with Time Filters and News Avoidance
Implements recommendations from PROJECT_ANALYSIS.md:
- Avoid first/last 30 minutes of trading session
- News event filtering
- Improved trade frequency management
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from datetime import datetime, time
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(__file__))
from news_filter import NewsFilter

print("=" * 70)
print("ENHANCED BACKTEST - XAUUSD AI BOT")
print("With Time Filters & News Avoidance")
print("=" * 70)
print()

# Enhanced Configuration
CONFIDENCE_THRESHOLD = 0.60  # Increased from 0.55 to reduce overtrading
INITIAL_CAPITAL = 50.0
RISK_PERCENT = 0.05
SPREAD_PIPS = 2.5
MAX_DRAWDOWN = 0.25  # 25% kill switch

# Conversion constants for XAUUSD
XAUUSD_PIP_VALUE = 20000  # Approximate conversion: return * 20000 ≈ USD P&L for XAUUSD

# Time filters
SESSION_START = time(13, 0)  # London-NY overlap starts at 13:00 UTC
SESSION_END = time(16, 59)   # Ends at 16:59 UTC
AVOID_FIRST_MINS = 30  # Avoid first 30 minutes of session
AVOID_LAST_MINS = 30   # Avoid last 30 minutes of session

print("⚙️ Enhanced Backtest Configuration:")
print(f"   Initial Capital: ${INITIAL_CAPITAL}")
print(f"   Risk per Trade: {RISK_PERCENT*100:.1f}%")
print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}% (optimized)")
print(f"   Spread: {SPREAD_PIPS} pips")
print(f"   Max Drawdown Kill Switch: {MAX_DRAWDOWN*100:.0f}%")
print()
print("⏰ Time Filters:")
print(f"   Trading Session: {SESSION_START} - {SESSION_END} UTC")
print(f"   Avoid First: {AVOID_FIRST_MINS} minutes")
print(f"   Avoid Last: {AVOID_LAST_MINS} minutes")
print(f"   Safe Window: 13:30 - 16:29 UTC")
print()

# Load data
print("📥 Loading test data...")
df = pd.read_csv('data/processed/xauusd_labeled.csv')
df['time'] = pd.to_datetime(df['time'])

# Load model and features
with open('python_training/models/feature_list.json', 'r') as f:
    feature_cols = json.load(f)

model = lgb.Booster(model_file='python_training/models/lightgbm_xauusd_v1.txt')

# Use test set (last 20%)
split_idx = int(0.8 * len(df))
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

print(f"   Test period: {df_test['time'].min().date()} → {df_test['time'].max().date()}")
print(f"   Test bars: {len(df_test):,}")
print()

# Get predictions
print("🔮 Running model predictions...")
X_test = df_test[feature_cols].values
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
max_proba = np.max(y_pred_proba, axis=1)

df_test['pred_class'] = y_pred
df_test['pred_confidence'] = max_proba

print(f"   ✓ {len(df_test):,} predictions generated")
print()

# Apply time filters
print("⏰ Applying time filters...")

def is_safe_trading_time(dt):
    """Check if time is within safe trading window"""
    t = dt.time()
    
    # Must be within session
    if not (SESSION_START <= t <= SESSION_END):
        return False
    
    # Avoid first 30 minutes (13:00-13:29)
    if t < time(13, 30):
        return False
    
    # Avoid last 30 minutes (16:30-16:59)
    if t >= time(16, 30):
        return False
    
    return True

df_test['safe_time'] = df_test['time'].apply(is_safe_trading_time)
time_filtered = (~df_test['safe_time']).sum()
print(f"   Time filter removed {time_filtered:,} bars ({time_filtered/len(df_test)*100:.1f}%)")

# Apply news filter (if calendar exists)
print()
print("📰 Applying news filter...")
news_filter = NewsFilter(buffer_minutes=30, impact_levels=['high'])

if news_filter.load_news_calendar('data/news/example_calendar.json'):
    df_test = news_filter.filter_trade_signals(df_test, time_column='time')
else:
    print("   ⚠️ No news calendar loaded - skipping news filter")
    print("   (This is OK for historical backtest, but enable for live trading)")

print()

# Filter by confidence
print(f"📊 Filtering signals (confidence ≥ {CONFIDENCE_THRESHOLD*100:.0f}%)...")
df_test = df_test[df_test['safe_time']].copy()  # Apply time filter
df_trades = df_test[
    (df_test['pred_confidence'] >= CONFIDENCE_THRESHOLD) &
    (df_test['pred_class'] != 1)  # Exclude HOLD
].copy()

print(f"   Total high-confidence signals: {len(df_trades):,}")
print(f"   ({len(df_trades)/len(df_test)*100:.1f}% of filtered test set)")
print()

# Simulate trades with enhanced risk management
print("💰 Simulating trades with enhanced risk management...")
print()

trades = []
equity = INITIAL_CAPITAL
equity_curve = [INITIAL_CAPITAL]
peak_equity = INITIAL_CAPITAL
max_drawdown = 0.0
kill_switch_triggered = False

for idx, row in df_trades.iterrows():
    # Check kill switch
    current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
    if current_drawdown >= MAX_DRAWDOWN:
        kill_switch_triggered = True
        print(f"   ⚠️ KILL SWITCH TRIGGERED at {row['time']}")
        print(f"      Drawdown: {current_drawdown*100:.1f}% (max: {MAX_DRAWDOWN*100:.0f}%)")
        print(f"      Equity: ${equity:.2f}")
        break
    
    signal = row['pred_class']  # 0=SHORT, 1=HOLD, 2=LONG
    confidence = row['pred_confidence']
    forward_return = row['forward_return_15m']
    
    # Calculate trade outcome
    if signal == 2:  # LONG
        net_return = forward_return - (SPREAD_PIPS / XAUUSD_PIP_VALUE)
        direction = 'LONG'
    else:  # SHORT (signal == 0)
        net_return = -forward_return - (SPREAD_PIPS / XAUUSD_PIP_VALUE)
        direction = 'SHORT'
    
    # Position sizing: 5% risk
    position_size = equity * RISK_PERCENT
    trade_pnl = net_return * XAUUSD_PIP_VALUE  # Convert return to USD P&L
    
    # Update equity
    equity += trade_pnl
    equity_curve.append(equity)
    
    # Update peak and drawdown
    if equity > peak_equity:
        peak_equity = equity
    
    current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
    max_drawdown = max(max_drawdown, current_drawdown)
    
    # Record trade
    trades.append({
        'time': row['time'],
        'direction': direction,
        'confidence': confidence,
        'pnl': trade_pnl,
        'equity': equity,
        'drawdown': current_drawdown
    })

# Calculate performance metrics
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    wins = (trades_df['pnl'] > 0).sum()
    losses = (trades_df['pnl'] <= 0).sum()
    total_trades = len(trades_df)
    
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    win_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    loss_profit = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = win_profit / loss_profit if loss_profit > 0 else 0
    
    net_profit = equity - INITIAL_CAPITAL
    return_pct = (net_profit / INITIAL_CAPITAL) * 100
    
    # Calculate trades per day
    test_days = (df_test['time'].max() - df_test['time'].min()).days
    trades_per_day = total_trades / test_days if test_days > 0 else 0
    
    # Performance by direction
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    
    long_wr = (long_trades['pnl'] > 0).sum() / len(long_trades) if len(long_trades) > 0 else 0
    short_wr = (short_trades['pnl'] > 0).sum() / len(short_trades) if len(short_trades) > 0 else 0
    
    # Print results
    print("=" * 70)
    print("📊 ENHANCED BACKTEST RESULTS")
    print("=" * 70)
    print()
    print(f"📈 Overall Performance:")
    print(f"   Total Trades: {total_trades:,}")
    print(f"   Trades/Day: {trades_per_day:.1f} {'✅' if 4 <= trades_per_day <= 6 else '⚠️'}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Net Profit: ${net_profit:.2f}")
    print(f"   Return: {return_pct:.1f}%")
    print(f"   Max Drawdown: {max_drawdown*100:.1f}%")
    print(f"   Final Equity: ${equity:.2f}")
    print()
    
    print(f"📊 By Direction:")
    print(f"   LONG:  {len(long_trades):,} trades | {long_wr*100:.1f}% WR")
    print(f"   SHORT: {len(short_trades):,} trades | {short_wr*100:.1f}% WR")
    print()
    
    if kill_switch_triggered:
        print(f"⚠️ WARNING: Kill switch was triggered during test")
        print(f"   This indicates the strategy experienced severe drawdown")
        print(f"   Consider further optimization or risk reduction")
        print()
    
    # Compare to original backtest
    print("=" * 70)
    print("📊 COMPARISON TO ORIGINAL BACKTEST")
    print("=" * 70)
    print()
    
    # Load original results
    try:
        with open('python_training/models/backtest_results.json', 'r') as f:
            original = json.load(f)['performance']
        
        print("Metric                 | Original  | Enhanced  | Change")
        print("-" * 70)
        print(f"Win Rate               | {original['win_rate']*100:6.1f}%   | {win_rate*100:6.1f}%   | {(win_rate - original['win_rate'])*100:+.1f}%")
        print(f"Profit Factor          | {original['profit_factor']:6.2f}    | {profit_factor:6.2f}    | {profit_factor - original['profit_factor']:+.2f}")
        print(f"Trades/Day             | {original['total_trades']/149:6.1f}    | {trades_per_day:6.1f}    | {trades_per_day - original['total_trades']/149:+.1f}")
        print(f"Max Drawdown           | {abs(original['max_drawdown']):6.1f}%   | {max_drawdown*100:6.1f}%   | {(max_drawdown*100 - abs(original['max_drawdown'])):+.1f}%")
        print()
        
        print("✅ Improvements:")
        if trades_per_day < 10:
            print("   • Significantly reduced overtrading")
        if max_drawdown * 100 < 20:
            print("   • Improved risk management")
        if win_rate > original['win_rate']:
            print("   • Higher win rate")
        print()
        
    except Exception as e:
        print(f"   Could not load original results for comparison")
    
    # Save enhanced results
    output = {
        'config': {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'initial_capital': INITIAL_CAPITAL,
            'risk_percent': RISK_PERCENT,
            'time_filters': {
                'avoid_first_mins': AVOID_FIRST_MINS,
                'avoid_last_mins': AVOID_LAST_MINS,
                'safe_window': '13:30-16:29 UTC'
            },
            'news_filter_enabled': news_filter.news_calendar is not None
        },
        'performance': {
            'total_trades': int(total_trades),
            'trades_per_day': float(trades_per_day),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'net_profit': float(net_profit),
            'return_percent': float(return_pct),
            'max_drawdown': float(max_drawdown * 100),
            'kill_switch_triggered': kill_switch_triggered
        },
        'breakdown': {
            'long': {
                'trades': int(len(long_trades)),
                'win_rate': float(long_wr)
            },
            'short': {
                'trades': int(len(short_trades)),
                'win_rate': float(short_wr)
            }
        }
    }
    
    with open('python_training/models/backtest_enhanced.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Results saved to: python_training/models/backtest_enhanced.json")
    print()
    
else:
    print("❌ No trades executed - check filters and thresholds")

print("=" * 70)
print("✅ ENHANCED BACKTEST COMPLETE")
print("=" * 70)
