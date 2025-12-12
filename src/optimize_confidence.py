"""
Enhanced Confidence Threshold Optimization
Analyzes trade frequency and performance across different thresholds
Recommendation: Use 60-65% confidence to reduce overtrading
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 70)
print("ENHANCED CONFIDENCE THRESHOLD OPTIMIZATION")
print("=" * 70)
print()

# Load data and model
print("📥 Loading data and model...")
df = pd.read_csv('data/processed/xauusd_labeled.csv')
df['time'] = pd.to_datetime(df['time'])

with open('python_training/models/feature_list.json', 'r') as f:
    feature_cols = json.load(f)

model = lgb.Booster(model_file='python_training/models/lightgbm_xauusd_v1.txt')

# Split data
split_idx = int(0.8 * len(df))
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
X_test = df_test[feature_cols].values

print(f"   Test samples: {len(X_test):,}")
print(f"   Test period: {df_test['time'].min().date()} → {df_test['time'].max().date()}")
print()

# Get predictions and confidence
print("🔮 Computing predictions and confidence...")
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
max_proba = np.max(y_pred_proba, axis=1)

df_test['pred_class'] = y_pred
df_test['pred_confidence'] = max_proba

print(f"   ✓ Predictions computed")
print()

# Analyze by confidence threshold
print("=" * 70)
print("📊 PERFORMANCE BY CONFIDENCE THRESHOLD (WITH TRADE FREQUENCY)")
print("=" * 70)
print()

# Configuration
INITIAL_CAPITAL = 50.0
RISK_PERCENT = 0.05
SPREAD_PIPS = 2.5

thresholds = [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75]
results = []

test_days = (df_test['time'].max() - df_test['time'].min()).days

for threshold in thresholds:
    # Filter by confidence
    df_trades = df_test[df_test['pred_confidence'] >= threshold].copy()
    
    # Skip HOLD signals
    df_trades = df_trades[df_trades['pred_class'] != 1].copy()
    
    if len(df_trades) == 0:
        continue
    
    # Simulate trades
    wins = 0
    losses = 0
    total_profit = 0.0
    win_profit = 0.0
    loss_profit = 0.0
    
    for idx, row in df_trades.iterrows():
        signal = row['pred_class']
        forward_return = row['forward_return_15m']
        
        if signal == 2:  # LONG
            net_return = forward_return - (SPREAD_PIPS / 20000)
        else:  # SHORT (signal == 0)
            net_return = -forward_return - (SPREAD_PIPS / 20000)
        
        # Simplified P&L calculation (proportional to return)
        # In production, use proper position sizing based on equity and risk %
        trade_pnl = net_return * 50000  # Approximate P&L for comparison
        
        if trade_pnl > 0:
            wins += 1
            win_profit += trade_pnl
        else:
            losses += 1
            loss_profit += abs(trade_pnl)
        
        total_profit += trade_pnl
    
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = win_profit / loss_profit if loss_profit > 0 else 0
    trades_per_day = total_trades / test_days if test_days > 0 else 0
    
    results.append({
        'threshold': threshold,
        'total_trades': total_trades,
        'trades_per_day': trades_per_day,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'net_profit': total_profit,
        'return_pct': (total_profit / INITIAL_CAPITAL) * 100
    })
    
    print(f"Confidence ≥ {threshold*100:.0f}%:")
    print(f"   Total Trades: {total_trades:,}")
    print(f"   Trades/Day: {trades_per_day:.1f}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Net Profit: ${total_profit:.2f}")
    print(f"   Return: {(total_profit/INITIAL_CAPITAL)*100:.1f}%")
    
    # Recommendation based on trade frequency
    if 4 <= trades_per_day <= 6:
        print(f"   ✅ OPTIMAL RANGE (4-6 trades/day)")
    elif trades_per_day < 4:
        print(f"   ⚠️ Too conservative")
    else:
        print(f"   ⚠️ Overtrading (reduce frequency)")
    print()

# Find optimal threshold (balance of win rate, profit factor, and trade frequency)
print("=" * 70)
print("🎯 OPTIMAL THRESHOLD RECOMMENDATION")
print("=" * 70)
print()

# Score each threshold based on multiple criteria
for result in results:
    # Ideal: 4-6 trades/day, >60% win rate, >1.5 profit factor
    tpd_score = 1.0 if 4 <= result['trades_per_day'] <= 6 else max(0, 1 - abs(result['trades_per_day'] - 5) / 10)
    wr_score = result['win_rate'] / 0.7  # Normalize to 70% ideal
    pf_score = result['profit_factor'] / 2.0  # Normalize to 2.0 ideal
    
    result['composite_score'] = (tpd_score * 0.4 + wr_score * 0.3 + pf_score * 0.3)

# Sort by composite score
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('composite_score', ascending=False)

print("Top 3 Recommended Thresholds:")
print()
for idx, row in results_df.head(3).iterrows():
    print(f"#{idx+1}. Confidence ≥ {row['threshold']*100:.0f}%")
    print(f"     Trades/Day: {row['trades_per_day']:.1f}")
    print(f"     Win Rate: {row['win_rate']*100:.1f}%")
    print(f"     Profit Factor: {row['profit_factor']:.2f}")
    print(f"     Return: {row['return_pct']:.0f}%")
    print(f"     Score: {row['composite_score']:.3f}")
    print()

# Save results
output = {
    'analysis_date': datetime.now().isoformat(),
    'test_period_days': test_days,
    'recommended_threshold': float(results_df.iloc[0]['threshold']),
    'all_results': results_df.to_dict('records')
}

with open('python_training/models/confidence_optimization.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"💾 Results saved to: python_training/models/confidence_optimization.json")
print()

# Create visualization
print("📊 Generating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Win Rate vs Threshold
axes[0, 0].plot(results_df['threshold']*100, results_df['win_rate']*100, 'o-', linewidth=2, markersize=8)
axes[0, 0].axhline(y=60, color='g', linestyle='--', alpha=0.5, label='60% target')
axes[0, 0].set_xlabel('Confidence Threshold (%)', fontsize=11)
axes[0, 0].set_ylabel('Win Rate (%)', fontsize=11)
axes[0, 0].set_title('Win Rate vs Confidence Threshold', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

# Plot 2: Trades Per Day vs Threshold
axes[0, 1].plot(results_df['threshold']*100, results_df['trades_per_day'], 'o-', linewidth=2, markersize=8, color='orange')
axes[0, 1].axhspan(4, 6, alpha=0.2, color='green', label='Optimal range (4-6)')
axes[0, 1].set_xlabel('Confidence Threshold (%)', fontsize=11)
axes[0, 1].set_ylabel('Trades Per Day', fontsize=11)
axes[0, 1].set_title('Trade Frequency vs Confidence Threshold', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()

# Plot 3: Profit Factor vs Threshold
axes[1, 0].plot(results_df['threshold']*100, results_df['profit_factor'], 'o-', linewidth=2, markersize=8, color='green')
axes[1, 0].axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='1.5 minimum')
axes[1, 0].set_xlabel('Confidence Threshold (%)', fontsize=11)
axes[1, 0].set_ylabel('Profit Factor', fontsize=11)
axes[1, 0].set_title('Profit Factor vs Confidence Threshold', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend()

# Plot 4: Composite Score
axes[1, 1].bar(results_df['threshold']*100, results_df['composite_score'], color='steelblue', alpha=0.7)
best_idx = results_df['composite_score'].idxmax()
axes[1, 1].bar(results_df.loc[best_idx, 'threshold']*100, results_df.loc[best_idx, 'composite_score'], 
               color='gold', alpha=0.9, label='Best')
axes[1, 1].set_xlabel('Confidence Threshold (%)', fontsize=11)
axes[1, 1].set_ylabel('Composite Score', fontsize=11)
axes[1, 1].set_title('Overall Performance Score', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('python_training/models/confidence_optimization.png', dpi=150, bbox_inches='tight')
print(f"💾 Chart saved to: python_training/models/confidence_optimization.png")
print()

print("=" * 70)
print("✅ ANALYSIS COMPLETE")
print("=" * 70)
print()
print(f"🎯 RECOMMENDED THRESHOLD: {results_df.iloc[0]['threshold']*100:.0f}%")
print(f"   This reduces overtrading while maintaining strong performance")
print()
