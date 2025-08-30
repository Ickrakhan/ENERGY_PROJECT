import pandas as pd

# Input file (your original dataset)
input_file = "household_power_consumption.txt"

# Output file (smaller version)
output_file = "household_power_consumption_small.txt"

print(" Reading dataset (this may take a while)...")
df = pd.read_csv(input_file, sep=";", low_memory=False)

print(f" Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

# Take a 10% random sample (you can change frac=0.1 to 0.05 or 0.2)
sample_df = df.sample(frac=0.1, random_state=42)

print(f" New dataset size: {sample_df.shape[0]} rows, {sample_df.shape[1]} columns")

# Save the smaller dataset
sample_df.to_csv(output_file, sep=";", index=False)

print(f" Shrunk dataset saved as {output_file}")






