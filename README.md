# XAUUSD AI Trading Bot

Industrial-grade MT5 trading bot using hybrid ML/RL architecture.

## Project Status
- [x] Project structure created
- [ ] Data downloaded (2020-2025)
- [ ] Data filtered (London-NY overlap)
- [ ] Features engineered (60 features)
- [ ] Model trained (LightGBM + PyTorch)
- [ ] ONNX exported
- [ ] MT5 EA compiled
- [ ] Backtested (3 years)
- [ ] Shadow tested (30 days)
- [ ] Live deployed

## Quick Start
1. Download data: `python download_dukascopy_data.py`
2. Filter overlap: `python filter_overlap.py`
3. Engineer features: `python feature_engineering.py`
4. Train models: `python python_training/training/train_ensemble.py`
5. Export ONNX: `python python_training/export/to_onnx.py`

## Structure
- `data/` - Raw and processed datasets
- `python_training/` - ML model training pipeline
- `mt5_expert_advisor/` - MT5 EA code (MQL5)
- `deployment/` - Production deployment scripts
- `tests/` - Unit tests
