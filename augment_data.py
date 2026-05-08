import pandas as pd
import numpy as np

# 1. Load your Parkinson's base data
# Make sure your text file is saved exactly as 'parkinson_base.csv'
df = pd.read_csv('parkinson_base.csv')

# We don't want to augment the timestamp, just the sensor data
sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

# Create an empty list to hold all our new datasets
augmented_datasets = [df] 

# Number of synthetic variations you want to create
num_variations = 50 

print(f"Original rows: {len(df)}. Generating massive Parkinson's dataset...")

for i in range(num_variations):
    new_df = df.copy()
    
    # Technique A: Add Jitter (Gaussian Noise)
    # This simulates natural sensor noise and slight human variance
    noise = np.random.normal(0, 150, new_df[sensor_cols].shape) 
    new_df[sensor_cols] = new_df[sensor_cols] + noise
    
    # Technique B: Magnitude Scaling
    # This simulates stepping slightly harder or softer (scaling by 90% to 110%)
    scaling_factor = np.random.uniform(0.90, 1.10)
    new_df[sensor_cols] = new_df[sensor_cols] * scaling_factor
    
    # Fix the timestamps so they continue continuously
    max_time = augmented_datasets[-1]['Timestamp'].max()
    time_diffs = df['Timestamp'].diff().fillna(22) # ~22ms diff based on your ESP32 data
    new_df['Timestamp'] = max_time + time_diffs.cumsum()
    
    # Append the new synthetic walk to our massive list
    augmented_datasets.append(new_df)

# Combine everything into one massive DataFrame
massive_dataset = pd.concat(augmented_datasets, ignore_index=True)

# Add the AI Label (CRITICAL: 1 for Parkinson's)
massive_dataset['Label'] = 1

# Save to a new file
massive_dataset.to_csv('parkinson_large_dataset.csv', index=False)

print(f"Success! Generated a massive dataset with {len(massive_dataset)} rows.")
print("Saved as 'parkinson_large_dataset.csv'")
