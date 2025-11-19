"""
Download XAUUSD Historical Data from Dukascopy using duka package
Downloads M1 (1-minute) bars from 2020-01-01 to 2025-11-19
"""

import os
import subprocess
import sys
import pandas as pd
from datetime import datetime

class DukaDownloader:
    """
    Wrapper for duka CLI tool to download Dukascopy data.
    """
    
    def __init__(self):
        self.duka_installed = self.check_duka_installation()
    
    def check_duka_installation(self):
        """Check if duka is installed, install if not."""
        try:
            result = subprocess.run(['duka', '-h'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            print("✅ duka package is installed")
            return True
        except FileNotFoundError:
            print("⚠️ duka not found. Installing now...")
            return self.install_duka()
        except Exception as e:
            print(f"⚠️ Error checking duka: {e}")
            return False
    
    def install_duka(self):
        """Install duka package using pip."""
        try:
            print("📦 Installing duka package...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'duka'])
            print("✅ duka installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install duka: {e}")
            print("\nManual installation:")
            print("   pip install duka")
            return False
    
    def download_data(self, symbol='XAUUSD', start_date='2024-01-01', 
                     end_date='2025-11-19', candle='M1', output_dir='data/raw'):
        """
        Download historical data using duka Python API.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("DUKASCOPY DATA DOWNLOAD")
        print("=" * 80)
        print(f"   Symbol: {symbol}")
        print(f"   Timeframe: {candle}")
        print(f"   Start: {start_date}")
        print(f"   End: {end_date}")
        print(f"   Output: {output_dir}")
        print(f"\n⏳ Downloading... (may take 10-30 minutes for 5 years of M1 data)\n")
        
        try:
            from duka.app import app
            from duka.core.utils import TimeFrame
            
            # Map candle to TimeFrame
            tf_map = {'M1': TimeFrame.M1, 'M5': TimeFrame.M5, 'M15': TimeFrame.M15, 
                     'M30': TimeFrame.M30, 'H1': TimeFrame.H1, 'H4': TimeFrame.H4}
            
            app(
                [symbol],
                start_date,
                end_date,
                threads=10,
                folder=output_dir,
                timeframe=tf_map.get(candle, TimeFrame.M1),
                header=True
            )
            
            print("\n✅ Download completed!")
            return True
                
        except Exception as e:
            print(f"\n❌ Download error: {e}")
            print("\nNote: Dukascopy may be blocking requests. Try:")
            print("   1. Shorter date range (e.g., 1 year at a time)")
            print("   2. Use a VPN")
            print("   3. Download manually from another source")
            return False
    
    def validate_downloaded_data(self, output_dir='data/raw', symbol='XAUUSD'):
        """
        Validate the downloaded CSV files.
        """
        print("\n" + "=" * 80)
        print("DATA VALIDATION")
        print("=" * 80)
        
        # Find CSV files
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ No CSV files found in output directory")
            return False
        
        print(f"📁 Found {len(csv_files)} CSV file(s):")
        for f in csv_files:
            print(f"   - {f}")
        
        # Load and validate first file
        first_file = os.path.join(output_dir, csv_files[0])
        print(f"\n🔍 Validating: {first_file}")
        
        try:
            df = pd.read_csv(first_file)
            
            print(f"\n✅ File loaded successfully")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")
            
            # Show sample
            print(f"\n📋 Sample data (first 5 rows):")
            print(df.head())
            
            # Basic validation
            required_cols = ['time', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"\n⚠️ Missing columns: {missing_cols}")
            else:
                print(f"\n✅ All required columns present")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
    
    def merge_files(self, output_dir='data/raw', output_file='xauusd_m1_full.csv'):
        """
        Merge multiple CSV files into one master file.
        """
        print("\n" + "=" * 80)
        print("MERGING CSV FILES")
        print("=" * 80)
        
        csv_files = [f for f in os.listdir(output_dir) 
                     if f.endswith('.csv') and f != output_file]
        
        if not csv_files:
            print("❌ No CSV files to merge")
            return False
        
        print(f"📁 Merging {len(csv_files)} file(s)...")
        
        try:
            # Read all files
            dfs = []
            for file in csv_files:
                filepath = os.path.join(output_dir, file)
                df = pd.read_csv(filepath)
                dfs.append(df)
                print(f"   ✓ {file}: {len(df):,} rows")
            
            # Concatenate all dataframes
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates (if any)
            merged_df = merged_df.drop_duplicates(subset=['time'])
            
            # Sort by time
            merged_df = merged_df.sort_values('time')
            
            # Save merged file
            output_path = os.path.join(output_dir, output_file)
            merged_df.to_csv(output_path, index=False)
            
            print(f"\n✅ Merged file saved:")
            print(f"   Path: {output_path}")
            print(f"   Total rows: {len(merged_df):,}")
            print(f"   Date range: {merged_df['time'].min()} → {merged_df['time'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Merge error: {e}")
            return False


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Configuration
    SYMBOL = 'XAUUSD'
    START_DATE = '2020-01-01'
    END_DATE = '2025-11-19'
    TIMEFRAME = 'M1'  # 1-minute bars
    OUTPUT_DIR = 'data/raw'
    
    # Initialize downloader
    downloader = DukaDownloader()
    
    # Download data
    success = downloader.download_data(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        candle=TIMEFRAME,
        output_dir=OUTPUT_DIR
    )
    
    if success:
        # Validate downloaded data
        downloader.validate_downloaded_data(OUTPUT_DIR, SYMBOL)
        
        # Merge files if multiple files downloaded
        downloader.merge_files(OUTPUT_DIR, f'{SYMBOL.lower()}_m1_2020_2025.csv')
        
        print("\n" + "=" * 80)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("   1. Run filter_overlap.py to filter for London-NY hours")
        print("   2. Run feature_engineering.py to compute 60 features")
        print("   3. Proceed to model training (Phase 3)")
    else:
        print("\n" + "=" * 80)
        print("❌ DOWNLOAD FAILED")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("   1. Check internet connection")
        print("   2. Verify symbol name (XAUUSD for gold)")
        print("   3. Try shorter date range (e.g., 1 year)")
        print("   4. Manual install: pip install duka")
