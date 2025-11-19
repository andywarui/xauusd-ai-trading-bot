import pandas as pd

# Load filtered data
df = pd.read_csv('data/processed/xauusd_m1_overlap.csv')

# Check hours
print(df['hour'].value_counts().sort_index())
# Should only show: 13, 14, 15, 16

# Check volume distribution
print(f"Avg volume: {df['Volume'].mean():.5f}")
print(f"Max volume: {df['Volume'].max():.5f}")
