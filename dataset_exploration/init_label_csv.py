import numpy as np
import glob
import pandas as pd
import os
import re

# Function for natural sorting
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Get all subdirectories in the dataset directory
base_dir = "data/"
subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for subdir in subdirs:
    print(f"Processing directory: {subdir}")
    
    # Get all .npy files in the current subdirectory, sorted naturally by filename
    files = sorted(glob.glob(os.path.join(subdir, "*.npy")), key=natural_sort_key)
    if not files:
        print(f"No .npy files found in {subdir}")
        continue
    
    # Prepare data for the Excel file
    rows = []
    for i, file in enumerate(files, start=1):
        try:
            print(f"Processing file {i}/{len(files)}: {file}")
            data = np.load(file, allow_pickle=True)
            num_trajectories = len(data)
            
            if num_trajectories > 0:
                for traj_idx in range(num_trajectories):
                    rows.append({
                        "File Name": file,
                        "Index": traj_idx,
                        "Label": "",  # Placeholder for manual annotation
                        "Notes": ""   # Placeholder for manual annotation
                    })
            else:
                rows.append({
                    "File Name": file,
                    "Index": "No trajectories",
                    "Label": "",
                    "Notes": ""
                })
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Create a DataFrame and save to an Excel file for the current directory
    if rows:
        df = pd.DataFrame(rows)
        output_file = os.path.join(subdir, "contents.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved contents to {output_file}")
    else:
        print(f"No .npy files found in {subdir}")