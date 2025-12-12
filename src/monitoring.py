"""
Trading Bot Monitoring and Alerting System
Tracks performance, detects anomalies, and sends alerts
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class TradingMonitor:
    """
    Real-time monitoring system for the trading bot
    Tracks key metrics and sends alerts when thresholds are breached
    """
    
    def __init__(self, config_file='config/monitoring_config.json'):
        """Initialize monitoring system with configuration"""
        self.config = self.load_config(config_file)
        self.trades = []
        self.alerts = []
        
    def load_config(self, config_file):
        """Load monitoring configuration"""
        default_config = {
            'thresholds': {
                'max_drawdown': 0.25,  # 25%
                'min_win_rate': 0.50,   # 50%
                'min_profit_factor': 1.2,
                'max_consecutive_losses': 5,
                'max_trades_per_day': 10,
                'min_equity': 25.0  # Half of initial capital
            },
            'lookback_periods': {
                'short_term': 20,   # Last 20 trades
                'medium_term': 50,  # Last 50 trades
                'long_term': 100    # Last 100 trades
            },
            'alerts': {
                'email_enabled': False,
                'email_address': 'your-email@example.com',
                'slack_enabled': False,
                'slack_webhook': 'your-webhook-url',
                'console_enabled': True
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                print(f"⚠️ Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def add_trade(self, trade_data):
        """
        Add a new trade to monitoring
        
        Args:
            trade_data: dict with keys: time, direction, pnl, equity, confidence
        """
        self.trades.append({
            'timestamp': trade_data.get('time', datetime.now()),
            'direction': trade_data['direction'],
            'pnl': trade_data['pnl'],
            'equity': trade_data['equity'],
            'confidence': trade_data.get('confidence', 0),
            'win': trade_data['pnl'] > 0
        })
        
        # Run checks after each trade
        self.check_thresholds()
    
    def check_thresholds(self):
        """Check if any alert thresholds have been breached"""
        if len(self.trades) < 2:
            return
        
        thresholds = self.config['thresholds']
        periods = self.config['lookback_periods']
        
        # Check drawdown
        current_equity = self.trades[-1]['equity']
        peak_equity = max(t['equity'] for t in self.trades)
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        
        if drawdown >= thresholds['max_drawdown']:
            self.trigger_alert(
                'CRITICAL',
                'Max Drawdown Exceeded',
                f"Current drawdown: {drawdown*100:.1f}% (threshold: {thresholds['max_drawdown']*100:.0f}%)"
            )
        
        # Check minimum equity
        if current_equity <= thresholds['min_equity']:
            self.trigger_alert(
                'CRITICAL',
                'Minimum Equity Breached',
                f"Current equity: ${current_equity:.2f} (minimum: ${thresholds['min_equity']:.2f})"
            )
        
        # Check consecutive losses
        consecutive_losses = 0
        for trade in reversed(self.trades):
            if trade['win']:
                break
            consecutive_losses += 1
        
        if consecutive_losses >= thresholds['max_consecutive_losses']:
            self.trigger_alert(
                'WARNING',
                'Consecutive Losses',
                f"{consecutive_losses} consecutive losses (threshold: {thresholds['max_consecutive_losses']})"
            )
        
        # Check short-term performance
        if len(self.trades) >= periods['short_term']:
            recent_trades = self.trades[-periods['short_term']:]
            recent_wins = sum(1 for t in recent_trades if t['win'])
            recent_wr = recent_wins / len(recent_trades)
            
            if recent_wr < thresholds['min_win_rate']:
                self.trigger_alert(
                    'WARNING',
                    'Low Win Rate',
                    f"Win rate last {periods['short_term']} trades: {recent_wr*100:.1f}% (threshold: {thresholds['min_win_rate']*100:.0f}%)"
                )
        
        # Check trades per day
        if len(self.trades) >= 10:
            recent_times = [t['timestamp'] for t in self.trades[-10:]]
            time_span = (recent_times[-1] - recent_times[0]).total_seconds() / 3600  # hours
            
            if time_span > 0:
                trades_per_day = 10 / (time_span / 24)
                
                if trades_per_day > thresholds['max_trades_per_day']:
                    self.trigger_alert(
                        'WARNING',
                        'Overtrading Detected',
                        f"Current rate: {trades_per_day:.1f} trades/day (threshold: {thresholds['max_trades_per_day']})"
                    )
    
    def trigger_alert(self, severity, title, message):
        """
        Trigger an alert
        
        Args:
            severity: 'INFO', 'WARNING', or 'CRITICAL'
            title: Alert title
            message: Detailed message
        """
        alert = {
            'timestamp': datetime.now(),
            'severity': severity,
            'title': title,
            'message': message
        }
        
        self.alerts.append(alert)
        
        # Send alert via configured channels
        if self.config['alerts']['console_enabled']:
            self._send_console_alert(alert)
        
        # Email and Slack alerts require implementation
        if self.config['alerts'].get('email_enabled', False):
            print(f"⚠️ Email alerts not yet implemented. Alert would be sent to: {self.config['alerts'].get('email_address')}")
        
        if self.config['alerts'].get('slack_enabled', False):
            print(f"⚠️ Slack alerts not yet implemented. Alert would be sent to webhook.")
    
    def _send_console_alert(self, alert):
        """Print alert to console"""
        severity_icon = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }
        
        icon = severity_icon.get(alert['severity'], '•')
        
        print()
        print("=" * 70)
        print(f"{icon} ALERT: {alert['title']} [{alert['severity']}]")
        print("=" * 70)
        print(f"Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Message: {alert['message']}")
        print("=" * 70)
        print()
    
    def get_summary(self):
        """Get current performance summary"""
        if not self.trades:
            return "No trades recorded yet"
        
        total = len(self.trades)
        wins = sum(1 for t in self.trades if t['win'])
        losses = total - wins
        
        win_rate = wins / total if total > 0 else 0
        
        total_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        current_equity = self.trades[-1]['equity']
        peak_equity = max(t['equity'] for t in self.trades)
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'current_equity': current_equity,
            'current_drawdown': drawdown,
            'alerts_triggered': len(self.alerts)
        }
    
    def export_report(self, filepath='monitoring_report.json'):
        """Export monitoring report to file"""
        report = {
            'report_date': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'recent_trades': self.trades[-20:] if len(self.trades) >= 20 else self.trades,
            'alerts': self.alerts[-10:] if len(self.alerts) >= 10 else self.alerts
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Monitoring report exported to: {filepath}")


def create_monitoring_config():
    """Create example monitoring configuration file"""
    os.makedirs('config', exist_ok=True)
    
    config = {
        'thresholds': {
            'max_drawdown': 0.25,
            'min_win_rate': 0.50,
            'min_profit_factor': 1.2,
            'max_consecutive_losses': 5,
            'max_trades_per_day': 10,
            'min_equity': 25.0
        },
        'lookback_periods': {
            'short_term': 20,
            'medium_term': 50,
            'long_term': 100
        },
        'alerts': {
            'email_enabled': False,
            'email_address': 'your-email@example.com',
            'slack_enabled': False,
            'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
            'console_enabled': True
        }
    }
    
    with open('config/monitoring_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created monitoring config at: config/monitoring_config.json")


if __name__ == '__main__':
    print("=" * 70)
    print("TRADING MONITOR - DEMO")
    print("=" * 70)
    print()
    
    # Create config
    create_monitoring_config()
    print()
    
    # Initialize monitor
    monitor = TradingMonitor('config/monitoring_config.json')
    
    print("Testing monitoring system with simulated trades...")
    print()
    
    # Simulate trades
    equity = 50.0
    
    # Good trades
    for i in range(3):
        equity += 2.0
        monitor.add_trade({
            'time': datetime.now() - timedelta(hours=10-i),
            'direction': 'LONG',
            'pnl': 2.0,
            'equity': equity,
            'confidence': 0.65
        })
    
    # Simulate consecutive losses (should trigger alert)
    for i in range(6):
        equity -= 1.5
        monitor.add_trade({
            'time': datetime.now() - timedelta(hours=5-i),
            'direction': 'SHORT',
            'pnl': -1.5,
            'equity': equity,
            'confidence': 0.58
        })
    
    print()
    print("=" * 70)
    print("MONITORING SUMMARY")
    print("=" * 70)
    print()
    
    summary = monitor.get_summary()
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Win Rate: {summary['win_rate']*100:.1f}%")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print(f"Current Equity: ${summary['current_equity']:.2f}")
    print(f"Current Drawdown: {summary['current_drawdown']*100:.1f}%")
    print(f"Alerts Triggered: {summary['alerts_triggered']}")
    print()
    
    # Export report
    monitor.export_report('python_training/models/monitoring_demo_report.json')
    print()
    
    print("=" * 70)
    print("INTEGRATION INSTRUCTIONS")
    print("=" * 70)
    print()
    print("To integrate monitoring into your trading bot:")
    print()
    print("1. Import and initialize:")
    print("   from monitoring import TradingMonitor")
    print("   monitor = TradingMonitor('config/monitoring_config.json')")
    print()
    print("2. After each trade execution:")
    print("   monitor.add_trade({")
    print("       'time': trade_time,")
    print("       'direction': 'LONG' or 'SHORT',")
    print("       'pnl': trade_profit_loss,")
    print("       'equity': current_equity,")
    print("       'confidence': prediction_confidence")
    print("   })")
    print()
    print("3. Periodically export reports:")
    print("   monitor.export_report('daily_report.json')")
    print()
    print("4. Customize thresholds in config/monitoring_config.json")
    print()
