# Development Roadmap

## Phase 1: Data Acquisition (Week 1)
- [x] Setup project structure
- [ ] Download 5 years XAUUSD M1 data
- [ ] Validate data quality
- [ ] Filter London-NY overlap

## Phase 2: Feature Engineering (Week 1-2)
- [ ] Implement 60 feature pipeline
- [ ] Verify feature calculations
- [ ] Create training labels

## Phase 3: Model Training (Week 2-3)
- [ ] Train LightGBM classifier
- [ ] Train PyTorch hybrid model
- [ ] Train PPO RL agent
- [ ] Create ensemble meta-learner

## Phase 4: ONNX Export (Week 3)
- [ ] Export to ONNX format
- [ ] Validate ONNX inference
- [ ] Test feature parity (Python vs ONNX)

## Phase 5: Backtesting (Week 4)
- [ ] Python tick-level backtest (3 years)
- [ ] MT5 Strategy Tester validation
- [ ] Walk-forward analysis
- [ ] Monte Carlo stress tests

## Phase 6: MT5 EA Development (Week 5)
- [ ] Write MQL5 EA code
- [ ] Implement dual-mode (Validation/ML)
- [ ] Add risk management
- [ ] Compile and test on demo

## Phase 7: Shadow Testing (Week 6-9)
- [ ] 30-day paper trading
- [ ] Performance monitoring
- [ ] Model validation
- [ ] Emergency procedures tested

## Phase 8: Live Deployment (Week 10+)
- [ ] Deploy with $50 capital
- [ ] Conservative config (5% risk)
- [ ] Daily monitoring
- [ ] Performance review
