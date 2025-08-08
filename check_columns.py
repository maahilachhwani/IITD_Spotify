import pandas as pd

# Load the dataset
df = pd.read_csv('spotify_iitd1.csv')

print("Dataset columns:")
print(df.columns.tolist())
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head()) 