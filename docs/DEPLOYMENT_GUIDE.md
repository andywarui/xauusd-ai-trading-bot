# 🚀 Deployment Guide - XAUUSD AI Trading Bot

## Overview
This guide covers deploying the XAUUSD AI Trading Bot from development to production, including setup, testing, monitoring, and maintenance.

---

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] Python 3.11+ installed
- [ ] MetaTrader 5 installed and configured
- [ ] 16GB RAM available
- [ ] Stable internet connection
- [ ] VPS (recommended for 24/7 operation)

### Data Requirements
- [ ] Historical data downloaded (minimum 6 months)
- [ ] Data pipeline validated
- [ ] Features calculated and verified
- [ ] Model trained and tested

### Performance Validation
- [ ] Backtest win rate ≥ 55%
- [ ] Profit factor ≥ 1.5
- [ ] Max drawdown < 25%
- [ ] Feature parity verified (Python vs MT5)

---

## 🔧 Phase 1: Environment Setup

### 1.1 Install Dependencies

```bash
# Clone repository
git clone https://github.com/andywarui/xauusd-ai-trading-bot.git
cd xauusd-ai-trading-bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 1.2 Configure Settings

Create `config/production_config.json`:
```json
{
  "trading": {
    "confidence_threshold": 0.60,
    "initial_capital": 50.0,
    "risk_percent": 0.05,
    "max_drawdown": 0.25,
    "max_trades_per_day": 6
  },
  "session": {
    "start_time": "13:30",
    "end_time": "16:29",
    "timezone": "UTC"
  },
  "monitoring": {
    "enable_alerts": true,
    "alert_channels": ["console", "email"],
    "email": "your-email@example.com"
  }
}
```

### 1.3 Setup News Calendar

```bash
# Download economic calendar data
python src/news_filter.py

# Update config/news_sources.json with API keys
{
  "forexfactory_api_key": "your-key-here",
  "update_frequency": "daily"
}
```

---

## 🧪 Phase 2: Validation Testing

### 2.1 Run Enhanced Backtest

```bash
# Run optimized backtest with filters
python src/backtest_enhanced.py

# Verify results:
# - Win rate ≥ 55%
# - Trades/day: 4-6
# - Max drawdown < 25%
```

### 2.2 Optimize Confidence Threshold

```bash
# Find optimal threshold
python src/optimize_confidence.py

# Review results in:
# python_training/models/confidence_optimization.json
```

### 2.3 Validate Feature Calculations

```bash
# Compare Python features vs MT5 features
# Ensure ±2% tolerance
python src/validate_features.py --compare-mt5
```

---

## 📊 Phase 3: Shadow Testing (30 Days)

Shadow testing means running the bot in "observation mode" without executing real trades.

### 3.1 Setup Paper Trading

```bash
# Configure MT5 demo account
# Update config/mt5_credentials.json:
{
  "server": "Demo Server",
  "login": 12345678,
  "password": "your-demo-password",
  "mode": "demo"
}
```

### 3.2 Start Shadow Test

```python
# In MT5 EA or Python script
from monitoring import TradingMonitor

monitor = TradingMonitor('config/monitoring_config.json')

# Run bot in paper trading mode
# Log all signals but don't execute
while shadow_testing:
    signal = get_trading_signal()
    if signal:
        log_signal(signal)  # Don't execute
        monitor.add_trade(simulated_trade)
```

### 3.3 Daily Monitoring

```bash
# Generate daily reports
python src/generate_daily_report.py

# Check monitoring dashboard
python src/view_dashboard.py

# Review alerts
cat logs/alerts_$(date +%Y%m%d).log
```

### 3.4 Shadow Test Success Criteria

After 30 days, verify:
- [ ] Win rate ≥ 55%
- [ ] Profit factor ≥ 1.5
- [ ] Max drawdown < 25%
- [ ] No critical system errors
- [ ] Average 4-6 trades/day
- [ ] News filter working correctly

---

## 🚀 Phase 4: Live Deployment

### 4.1 Pre-Live Checklist

- [ ] Shadow test passed (30 days)
- [ ] All metrics meet targets
- [ ] Monitoring system operational
- [ ] Alert system configured
- [ ] Backup systems in place
- [ ] Emergency stop procedure documented

### 4.2 Start with Micro Capital

```json
// config/production_config.json
{
  "trading": {
    "initial_capital": 50.0,  // Start small
    "risk_percent": 0.03,      // Reduce to 3% for first week
    "confidence_threshold": 0.65  // Higher threshold initially
  }
}
```

### 4.3 Gradual Scale-Up

| Week | Capital | Risk % | Confidence | Max Trades/Day |
|------|---------|--------|------------|----------------|
| 1    | $50     | 3%     | 65%        | 4              |
| 2-3  | $50     | 4%     | 62%        | 5              |
| 4+   | $50+    | 5%     | 60%        | 6              |

### 4.4 Enable Live Trading

```bash
# Switch to live mode
python src/deploy_live.py --mode=live --confirm

# Monitor first trades closely
tail -f logs/trading_$(date +%Y%m%d).log
```

---

## 📈 Phase 5: Ongoing Monitoring

### 5.1 Daily Tasks

```bash
# Morning routine (before session)
python src/check_system_health.py
python src/update_news_calendar.py
python src/view_dashboard.py

# Evening routine (after session)
python src/generate_daily_report.py
python src/backup_trade_log.py
```

### 5.2 Weekly Tasks

```bash
# Review performance
python src/weekly_performance_review.py

# Check for model drift
python src/check_model_drift.py

# Update monitoring thresholds if needed
```

### 5.3 Monthly Tasks

```bash
# Run auto-retraining pipeline
python src/auto_retrain.py

# Comprehensive performance review
python src/monthly_report.py

# Backup all data and models
python src/backup_system.py
```

---

## ⚠️ Emergency Procedures

### Kill Switch Activation

**Automatic:**
- Triggered at 25% drawdown
- Halts all trading immediately
- Closes open positions
- Sends critical alert

**Manual:**
```bash
# Emergency stop
python src/emergency_stop.py --reason="manual intervention"

# Or in MT5 EA
# Press Emergency Stop button
# Or set input parameter: EmergencyStop = true
```

### Recovery Procedure

1. **Stop Trading**
   ```bash
   python src/emergency_stop.py
   ```

2. **Analyze Issue**
   ```bash
   python src/diagnose_issue.py
   ```

3. **Review Logs**
   ```bash
   cat logs/error_log.txt
   cat logs/trade_history.csv
   ```

4. **Fix Root Cause**
   - Model degradation → Retrain
   - Market regime change → Adjust parameters
   - Technical issue → Debug and fix

5. **Run Validation Tests**
   ```bash
   python src/validate_system.py
   ```

6. **Resume Trading (if safe)**
   ```bash
   python src/resume_trading.py --confirmed
   ```

---

## 🔄 Model Retraining Schedule

### Automatic Retraining

```bash
# Setup cron job for monthly retraining
# Edit crontab: crontab -e

# Run on 1st of each month at 2 AM
0 2 1 * * cd /path/to/xauusd-ai-trading-bot && /path/to/.venv/bin/python src/auto_retrain.py >> logs/retrain.log 2>&1
```

### Manual Retraining

```bash
# When performance degrades or after market regime change
python src/auto_retrain.py --force

# Review new model performance
python src/compare_models.py --old=v1 --new=v2

# Deploy if improved
python src/deploy_model.py --version=v2
```

---

## 📊 Performance Tracking

### Key Metrics to Monitor

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Win Rate | ≥60% | <55% | <50% |
| Profit Factor | ≥1.5 | <1.3 | <1.2 |
| Max Drawdown | <20% | >20% | >25% |
| Trades/Day | 4-6 | <3 or >8 | <2 or >10 |
| Consecutive Losses | <4 | 5 | ≥6 |

### Dashboard Access

```bash
# Launch web dashboard (if implemented)
python src/launch_dashboard.py --port=8080

# Or generate HTML report
python src/generate_html_report.py
# Open: reports/dashboard.html
```

---

## 🛡️ Risk Management

### Position Sizing
- **Default:** 5% risk per trade
- **Conservative:** 3% (during high volatility)
- **Aggressive:** Max 7% (only if consistently profitable)

### Stop Loss
- ATR-based dynamic stops
- Minimum: 20 pips
- Maximum: 100 pips

### Daily Loss Limit
```json
{
  "risk_limits": {
    "max_daily_loss_percent": 10,
    "max_daily_loss_absolute": 5.0,
    "stop_trading_on_limit": true
  }
}
```

---

## 📝 Logging and Auditing

### Log Files Structure

```
logs/
├── trading_YYYYMMDD.log      # Daily trade log
├── errors_YYYYMMDD.log        # Error log
├── alerts_YYYYMMDD.log        # Alert history
├── performance_YYYYMMDD.log   # Performance metrics
└── system_YYYYMMDD.log        # System events
```

### Trade Logging Format

```python
{
  "timestamp": "2025-12-12T14:30:45Z",
  "trade_id": "T20251212_001",
  "direction": "LONG",
  "entry_price": 2650.50,
  "exit_price": 2652.30,
  "pnl": 1.80,
  "confidence": 0.65,
  "reason": "ML signal with SMC confirmation"
}
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue: Low Win Rate (<50%)**
- Check model accuracy on recent data
- Verify feature calculations
- Review market regime (trending vs ranging)
- Consider retraining

**Issue: Overtrading (>8 trades/day)**
- Increase confidence threshold
- Tighten time filters
- Review signal generation logic

**Issue: High Drawdown (>20%)**
- Reduce position size
- Increase confidence threshold
- Enable news filter
- Check for model overfitting

**Issue: Model Not Predicting**
- Verify ONNX model loaded correctly
- Check feature calculation errors
- Review input data quality
- Validate feature list matches training

---

## 📚 Additional Resources

- **Project Analysis:** `docs/PROJECT_ANALYSIS.md`
- **Feature Engineering:** `docs/DATA_FORMAT.md`
- **Future Enhancements:** `docs/FUTURE_ENHANCEMENTS.md`
- **MT5 EA Development:** `TODO_MT5.md`

---

## ✅ Deployment Checklist Summary

- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Configuration files created
- [ ] News calendar configured
- [ ] Backtest validated (≥55% WR)
- [ ] Confidence threshold optimized
- [ ] 30-day shadow test passed
- [ ] Monitoring system operational
- [ ] Alert system configured
- [ ] Emergency procedures documented
- [ ] Started with micro capital
- [ ] Daily monitoring routine established
- [ ] Auto-retraining scheduled

---

**⚠️ Important Reminder:**
- Always start with demo trading
- Never risk more than you can afford to lose
- Monitor daily for first 30 days
- Be prepared to stop trading if metrics degrade
- Review and adapt to changing market conditions

**Good luck with your deployment! 🚀**
