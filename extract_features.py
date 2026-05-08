import pandas as pd
import numpy as np

print("Loading datasets...")
# Load both of your large datasets
df_normal = pd.read_csv('normal_large_dataset.csv')
df_parkinson = pd.read_csv('parkinson_large_dataset.csv')

sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
window_size = 150 # Roughly 3 seconds of walking data at 50Hz

def extract_features_from_df(df, label):
    print(f"Extracting features for Label {label}...")
    features = []
    
    # Slide a window across the data, 150 rows at a time
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        
        # Skip the very last window if it doesn't have a full 150 rows
        if len(window) < window_size:
            break 
            
        row_features = {}
        for col in sensor_cols:
            # Calculate the math for the AI to learn from
            row_features[f'{col}_mean'] = window[col].mean()
            row_features[f'{col}_std'] = window[col].std()
            row_features[f'{col}_max'] = window[col].max()
            row_features[f'{col}_min'] = window[col].min()
            row_features[f'{col}_rms'] = np.sqrt(np.mean(window[col]**2))
            
        row_features['Label'] = label
        features.append(row_features)
        
    return pd.DataFrame(features)

# Extract features for both classes
features_normal = extract_features_from_df(df_normal, 0)
features_parkinson = extract_features_from_df(df_parkinson, 1)

# Combine them into one final, clean dataset for the Machine Learning model
final_ml_dataset = pd.concat([features_normal, features_parkinson], ignore_index=True)

# Shuffle the data randomly so the AI doesn't learn in a biased order
final_ml_dataset = final_ml_dataset.sample(frac=1).reset_index(drop=True)

final_ml_dataset.to_csv('final_training_data.csv', index=False)

print("Success! Features extracted.")
print(f"Total training samples generated: {len(final_ml_dataset)}")
print("Saved as 'final_training_data.csv'")
