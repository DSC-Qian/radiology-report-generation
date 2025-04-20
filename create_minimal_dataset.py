import os
import pandas as pd
import numpy as np
import random
import shutil
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
ORIGINAL_CSV = 'mimic-cxr-list-filtered.csv'
MINIMAL_CSV = 'mimic-cxr-list-minimal.csv'
TARGET_VALID_SAMPLES = 50  # Target number of valid samples
MAX_ATTEMPTS = 1000  # Maximum number of samples to check

# Read the original CSV
print(f"Reading original CSV from {ORIGINAL_CSV}...")
df = pd.read_csv(ORIGINAL_CSV)

# Shuffle the dataframe
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Collect valid samples
valid_samples = []
valid_count = 0

print(f"Finding {TARGET_VALID_SAMPLES} valid image-report pairs...")
for idx in tqdm(range(min(MAX_ATTEMPTS, len(df_shuffled))), desc="Checking files"):
    if valid_count >= TARGET_VALID_SAMPLES:
        break
        
    image_path = df_shuffled.iloc[idx]['image_path']
    report_path = df_shuffled.iloc[idx]['report_path']
    
    if os.path.isfile(image_path) and os.path.isfile(report_path):
        valid_samples.append(df_shuffled.iloc[idx])
        valid_count += 1

# Convert valid samples to dataframe
if valid_samples:
    df_minimal = pd.DataFrame(valid_samples)
    print(f"Found {len(df_minimal)} valid image-report pairs.")
    
    # Save the minimal CSV
    print(f"Saving minimal dataset to {MINIMAL_CSV}...")
    df_minimal.to_csv(MINIMAL_CSV, index=False)
    
    print("Done! You can use this minimal dataset for training with 8GB VRAM.")
    print(f"To train with the minimal dataset, run:")
    print(f"python main.py --mode train --stage 1 --config_file config/minimal_train_config.json --csv_file {MINIMAL_CSV}")
else:
    print("No valid image-report pairs found. Please check your data paths.") 