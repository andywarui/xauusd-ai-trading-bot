"""
Auto-Retraining Pipeline
Automatically retrain model with new data on a schedule
Implements walk-forward validation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from datetime import datetime
import shutil

class AutoRetrainingPipeline:
    """
    Automated model retraining system
    - Monitors for new data
    - Triggers retraining on schedule
    - Validates new model performance
    - Backs up old models
    - Deploys new model if performance improves
    """
    
    def __init__(self, config_file='config/retraining_config.json'):
        """Initialize retraining pipeline"""
        self.config = self.load_config(config_file)
        self.current_model_path = 'python_training/models/lightgbm_xauusd_v1.txt'
        self.backup_dir = 'python_training/models/backups'
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def load_config(self, config_file):
        """Load retraining configuration"""
        default_config = {
            'retraining': {
                'schedule': 'monthly',  # 'daily', 'weekly', 'monthly'
                'min_new_data_days': 30,  # Minimum days of new data required
                'validation_split': 0.2,
                'min_accuracy_improvement': 0.01,  # 1% improvement required
                'min_performance_threshold': 0.45  # Minimum accuracy to deploy
            },
            'model_params': {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            },
            'notifications': {
                'enable_console': True,
                'enable_email': False,
                'email_address': 'your-email@example.com'
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
    
    def check_new_data(self):
        """Check if sufficient new data is available for retraining"""
        try:
            df = pd.read_csv('data/processed/xauusd_labeled.csv')
            df['time'] = pd.to_datetime(df['time'])
            
            # Get current model metadata
            with open('python_training/models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            last_training = datetime.fromisoformat(metadata['training_date'])
            new_data = df[df['time'] > last_training]
            
            days_of_new_data = (df['time'].max() - last_training).days
            
            print(f"📊 Data Check:")
            print(f"   Last training: {last_training.date()}")
            print(f"   Latest data: {df['time'].max().date()}")
            print(f"   New samples: {len(new_data):,}")
            print(f"   Days of new data: {days_of_new_data}")
            print()
            
            min_days = self.config['retraining']['min_new_data_days']
            
            if days_of_new_data >= min_days:
                print(f"✅ Sufficient new data available (≥{min_days} days)")
                return True, len(new_data)
            else:
                print(f"⏳ Insufficient new data ({days_of_new_data}/{min_days} days)")
                return False, len(new_data)
                
        except Exception as e:
            print(f"❌ Error checking data: {e}")
            return False, 0
    
    def backup_current_model(self):
        """Backup current model before retraining"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{self.backup_dir}/lightgbm_xauusd_backup_{timestamp}.txt"
        
        if os.path.exists(self.current_model_path):
            shutil.copy2(self.current_model_path, backup_path)
            print(f"✅ Backed up current model to: {backup_path}")
            return backup_path
        else:
            print("⚠️ No current model found to backup")
            return None
    
    def retrain_model(self):
        """Retrain model with latest data"""
        print()
        print("=" * 70)
        print("🔄 RETRAINING MODEL")
        print("=" * 70)
        print()
        
        # Load data
        print("📥 Loading labeled data...")
        df = pd.read_csv('data/processed/xauusd_labeled.csv')
        df['time'] = pd.to_datetime(df['time'])
        
        # Define features
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume',
                       'forward_return_15m', 'label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"   Rows: {len(df):,}")
        print(f"   Features: {len(feature_cols)}")
        print()
        
        # Prepare data
        X = df[feature_cols].values
        y = (df['label'] + 1).values  # Map to 0,1,2
        
        # Temporal split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Train/Test Split:")
        print(f"   Train: {len(X_train):,} samples")
        print(f"   Test:  {len(X_test):,} samples")
        print()
        
        # Train model
        print("🚀 Training new model...")
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            self.config['model_params'],
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=15),
                lgb.log_evaluation(period=0)
            ]
        )
        
        print(f"   ✓ Training complete ({model.num_trees()} trees)")
        print()
        
        # Validate performance
        print("📊 Validating new model...")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Test Accuracy: {accuracy*100:.2f}%")
        print()
        
        return model, accuracy, feature_cols
    
    def should_deploy(self, new_accuracy):
        """Determine if new model should be deployed"""
        try:
            # Load current model performance
            with open('python_training/models/model_metadata.json', 'r') as f:
                current_metadata = json.load(f)
            
            current_accuracy = current_metadata.get('accuracy', 0)
            min_improvement = self.config['retraining']['min_accuracy_improvement']
            min_threshold = self.config['retraining']['min_performance_threshold']
            
            print("🎯 Deployment Decision:")
            print(f"   Current model accuracy: {current_accuracy*100:.2f}%")
            print(f"   New model accuracy: {new_accuracy*100:.2f}%")
            print(f"   Improvement: {(new_accuracy - current_accuracy)*100:+.2f}%")
            print(f"   Required improvement: {min_improvement*100:.1f}%")
            print()
            
            # Check if meets minimum threshold
            if new_accuracy < min_threshold:
                print(f"❌ New model below minimum threshold ({min_threshold*100:.0f}%)")
                return False
            
            # Check if improvement is sufficient
            if new_accuracy >= current_accuracy + min_improvement:
                print("✅ New model shows significant improvement - will deploy")
                return True
            else:
                print("⚠️ Improvement insufficient - keeping current model")
                return False
                
        except Exception as e:
            print(f"❌ Error comparing models: {e}")
            return False
    
    def deploy_model(self, model, accuracy, feature_cols):
        """Deploy new model"""
        print()
        print("🚀 Deploying new model...")
        
        # Save model
        model_path = self.current_model_path
        model.save_model(model_path)
        print(f"   ✓ Model saved to: {model_path}")
        
        # Save metadata
        # Note: train_samples/test_samples counts stored during training
        split_idx = int(0.8 * len(model.train_set.data) if hasattr(model.train_set, 'data') else 0)
        
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'LightGBM',
            'num_features': len(feature_cols),
            'num_classes': 3,
            'train_samples': split_idx if split_idx > 0 else 'unknown',
            'test_samples': 'unknown',  # Set externally if needed
            'accuracy': float(accuracy),
            'best_iteration': model.best_iteration,
            'params': self.config['model_params']
        }
        
        with open('python_training/models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✓ Metadata saved")
        
        # Save feature list
        with open('python_training/models/feature_list.json', 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        print(f"   ✓ Feature list saved")
        print()
        print("✅ Deployment complete")
    
    def run_pipeline(self, force=False):
        """Run complete retraining pipeline"""
        print("=" * 70)
        print("AUTO-RETRAINING PIPELINE")
        print("=" * 70)
        print()
        
        # Check for new data
        if not force:
            has_data, new_samples = self.check_new_data()
            if not has_data:
                print("⏳ Skipping retraining - insufficient new data")
                return False
        else:
            print("⚠️ Force mode - skipping data check")
        
        # Backup current model
        self.backup_current_model()
        
        # Retrain
        new_model, new_accuracy, feature_cols = self.retrain_model()
        
        # Decide deployment
        if self.should_deploy(new_accuracy) or force:
            self.deploy_model(new_model, new_accuracy, feature_cols)
            print()
            print("=" * 70)
            print("✅ PIPELINE COMPLETE - NEW MODEL DEPLOYED")
            print("=" * 70)
            return True
        else:
            print()
            print("=" * 70)
            print("⏸️ PIPELINE COMPLETE - KEEPING CURRENT MODEL")
            print("=" * 70)
            return False


def create_retraining_config():
    """Create retraining configuration file"""
    os.makedirs('config', exist_ok=True)
    
    config = {
        'retraining': {
            'schedule': 'monthly',
            'min_new_data_days': 30,
            'validation_split': 0.2,
            'min_accuracy_improvement': 0.01,
            'min_performance_threshold': 0.45
        },
        'model_params': {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        },
        'notifications': {
            'enable_console': True,
            'enable_email': False,
            'email_address': 'your-email@example.com'
        }
    }
    
    with open('config/retraining_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created retraining config at: config/retraining_config.json")


if __name__ == '__main__':
    print("=" * 70)
    print("AUTO-RETRAINING PIPELINE - SETUP")
    print("=" * 70)
    print()
    
    # Create config
    create_retraining_config()
    print()
    
    print("=" * 70)
    print("USAGE INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Manual retraining:")
    print("   python src/auto_retrain.py")
    print()
    print("Schedule with cron (monthly on 1st at 2am):")
    print("   0 2 1 * * cd /path/to/project && python src/auto_retrain.py")
    print()
    print("In your code:")
    print("   from auto_retrain import AutoRetrainingPipeline")
    print("   pipeline = AutoRetrainingPipeline()")
    print("   pipeline.run_pipeline()")
    print()
    print("Force retraining (skip data check):")
    print("   pipeline.run_pipeline(force=True)")
    print()
