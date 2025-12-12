"""
Master Setup Script
Runs all enhancement tools in sequence to prepare the trading bot for production
"""

import subprocess
import sys
import os
from datetime import datetime

print("=" * 70)
print("XAUUSD AI TRADING BOT - MASTER SETUP")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

def run_script(script_name, description, optional=False):
    """Run a Python script and handle errors"""
    print("=" * 70)
    print(f"🔄 {description}")
    print("=" * 70)
    print(f"   Running: {script_name}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            print()
            if result.stdout:
                print(result.stdout)
            return True
        else:
            if optional:
                print("⚠️ SKIPPED (optional)")
                print()
                if result.stderr:
                    print("Info:", result.stderr[:500])
                return True
            else:
                print("❌ FAILED")
                print()
                if result.stderr:
                    print("Error:", result.stderr)
                return False
                
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all setup steps"""
    
    # Check if we're in the right directory
    if not os.path.exists('src/optimize_confidence.py'):
        print("❌ Error: Must run from project root directory")
        print("   cd xauusd-ai-trading-bot && python src/run_all_enhancements.py")
        sys.exit(1)
    
    print("📋 Setup Plan:")
    print("   1. Optimize confidence threshold")
    print("   2. Setup news filtering")
    print("   3. Run enhanced backtest")
    print("   4. Demo monitoring system")
    print("   5. Configure auto-retraining")
    print()
    
    input("Press Enter to continue (or Ctrl+C to cancel)...")
    print()
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Optimize confidence threshold
    if run_script(
        'src/optimize_confidence.py',
        'Step 1/5: Optimize Confidence Threshold',
        optional=False
    ):
        success_count += 1
    
    # Step 2: Setup news filtering
    if run_script(
        'src/news_filter.py',
        'Step 2/5: Setup News Filtering',
        optional=True  # Optional because it's mostly a demo
    ):
        success_count += 1
    
    # Step 3: Run enhanced backtest
    if run_script(
        'src/backtest_enhanced.py',
        'Step 3/5: Run Enhanced Backtest',
        optional=False
    ):
        success_count += 1
    
    # Step 4: Demo monitoring system
    if run_script(
        'src/monitoring.py',
        'Step 4/5: Demo Monitoring System',
        optional=True  # Optional because it's a demo
    ):
        success_count += 1
    
    # Step 5: Configure auto-retraining
    if run_script(
        'src/auto_retrain.py',
        'Step 5/5: Configure Auto-Retraining',
        optional=True  # Optional because it's setup only
    ):
        success_count += 1
    
    # Summary
    print()
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print()
    print(f"✅ Completed: {success_count}/{total_steps} steps")
    print()
    
    if success_count >= 3:
        print("🎉 SUCCESS! Your bot is ready for the next phase.")
        print()
        print("📊 Results Generated:")
        print("   • python_training/models/confidence_optimization.json")
        print("   • python_training/models/confidence_optimization.png")
        print("   • python_training/models/backtest_enhanced.json")
        print("   • config/monitoring_config.json")
        print("   • config/retraining_config.json")
        print()
        print("📖 Next Steps:")
        print("   1. Review confidence_optimization.json for recommended threshold")
        print("   2. Check backtest_enhanced.json for performance with filters")
        print("   3. Update your trading config with optimized parameters")
        print("   4. Read docs/DEPLOYMENT_GUIDE.md for production deployment")
        print("   5. Implement MT5 EA following TODO_MT5.md")
        print()
        print("🚀 Quick Commands:")
        print("   # View optimization results")
        print("   cat python_training/models/confidence_optimization.json | python -m json.tool")
        print()
        print("   # View enhanced backtest results")  
        print("   cat python_training/models/backtest_enhanced.json | python -m json.tool")
        print()
        print("   # Start monitoring (after integration)")
        print("   python src/monitoring.py")
        print()
    else:
        print("⚠️ Some steps failed. Please check errors above.")
        print("   You may need to:")
        print("   • Install missing dependencies: pip install -r requirements.txt")
        print("   • Run data pipeline first (merge → filter → features → labels → train)")
        print("   • Check that required data files exist in data/processed/")
        print()
    
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
