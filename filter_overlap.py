"""
Filter XAUUSD data for London-NY overlap window.
Run this after downloading full 3-year dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def filter_london_ny_overlap(input_file, output_file, 
                             start_hour=13, end_hour=17):
    """
    Filter XAUUSD M1 data for London-NY overlap.
    
    Args:
        input_file: Path to full dataset CSV
        output_file: Path to save filtered CSV
        start_hour: Overlap start (default 13 = 13:00 UTC)
        end_hour: Overlap end (default 17 = 17:00 UTC)
    
    Returns:
        DataFrame with filtered data
    """
    print(f"📥 Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Parse timestamps
    df['time'] = pd.to_datetime(df['UTC'], format='%d.%m.%Y %H:%M:%S.%f %Z')
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    
    print(f"   Original: {len(df):,} bars")
    print(f"   Date range: {df['time'].min().date()} → {df['time'].max().date()}")
    
    # Filter for overlap window
    overlap_df = df[
        (df['hour'] >= start_hour) & (df['hour'] < end_hour)
    ].copy()
    
    # Remove weekends (optional, markets usually closed)
    overlap_df = overlap_df[overlap_df['dayofweek'] < 5]  # Monday-Friday
    
    print(f"   Filtered: {len(overlap_df):,} bars")
    print(f"   Reduction: {(1 - len(overlap_df)/len(df))*100:.1f}%")
    print(f"   Avg bars/day: {len(overlap_df) / (len(df)/1440):.0f}")
    
    # Verify data quality
    print(f"\n✅ Quality checks:")
    print(f"   Missing values: {overlap_df.isnull().sum().sum()}")
    print(f"   Duplicates: {overlap_df.duplicated(subset=['time']).sum()}")
    print(f"   Price range: ${overlap_df['Close'].min():.2f} - ${overlap_df['Close'].max():.2f}")
    
    # Save filtered data
    overlap_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to {output_file}")
    
    return overlap_df

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Configuration
    INPUT_FILE = 'data/raw/XAUUSD_1min.csv'
    OUTPUT_FILE = 'data/processed/xauusd_m1_overlap.csv'
    
    # Filter data
    overlap_data = filter_london_ny_overlap(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        start_hour=13,  # 13:00 UTC
        end_hour=17     # 17:00 UTC (exclusive)
    )
    
    # Show sample
    print("\n📋 Sample filtered data:")
    print(overlap_data[['time', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
    
    print("\n✅ Filtering complete!")
    print(f"   Next step: Feature engineering on {len(overlap_data):,} bars")
