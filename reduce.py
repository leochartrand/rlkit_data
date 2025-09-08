# take data from /rlkit_labels/ and reduce the observation sizes to 48x48x3
import re
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
import cv2
import pickle

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
def load_labels():
    df = pd.read_csv("data/labels.csv")
    return df
    
def save_labels(dataframe: pd.DataFrame):
    """Update local labels csv file"""
    dataframe.to_csv("data/labels.csv", index=False)

def get_dataset_structure():
    # Load filepath/ID mapping
    file_mapping = json.load(open('drive_file_mapping.json'))

    # Extract directory structure from mapping
    drive_structure = {}
    for file_path in file_mapping.keys():
        parts = file_path.split('/')
        subdir = parts[0] 
        filename = parts[1]
        if subdir not in drive_structure:
            drive_structure[subdir] = []
        drive_structure[subdir].append(filename)

    return drive_structure, file_mapping

def load_npy_file(file_path):
    data = np.load("rlkit_labels/"+file_path, allow_pickle=True)
    return data
        
def get_trajectory_images(trajectory_data):
    frames = []
    frames.append([
        trajectory_data[0]["hires_image_observation"],
        trajectory_data[2]["hires_image_observation"],
    ])
    return frames[0]

labels = load_labels()

drive_structure = get_dataset_structure()[0]

subdirs = sorted(drive_structure.keys(), key=natural_sort_key)

resized_trajectories = [] # Trajectories dimension (N)

for subdir in subdirs:
    if not os.path.exists("rlkit_labels/"+subdir):
        print(f"Skipping {subdir}, directory does not exist.")
        continue
    # Load .npy files from the selected directory
    files = sorted(drive_structure[subdir], key=natural_sort_key)
    for file in tqdm(files, desc=f"{subdir}"):
        file_path = f"{subdir}/{file}"
        if not os.path.exists("rlkit_labels/"+file_path):
            print(f"Skipping {file_path}, file does not exist.")
            continue
        data = load_npy_file(file_path) 
        for idx in range(len(data)):
            trajectory = data[idx]
            try:
                frames = get_trajectory_images(trajectory)
            except Exception as e:
                print(f"Error processing {file_path} index {idx}: {e}")
                continue
            trajectory = [] # (Initial, target) dimension
            for i, frame in enumerate(frames):
                # Reduce size to 48x48
                resized = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_LINEAR)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) 
                trajectory.append(resized)
            label = labels.loc[labels['File Name'] == file_path, 'Label'].values[0]
            if label == "none":
                print(f"Skipping {file_path} index {idx}, label is 'none'.")
                continue
            trajectory.append(label)
            resized_trajectories.append(trajectory)

print(f"Total resized trajectories: {len(resized_trajectories)}")
print(f"Example trajectory: {resized_trajectories[0]}")
if not os.path.exists("val"):
    os.makedirs("val")
# Save to pickle file
with open(f"val_data.pkl", "wb") as f:
    pickle.dump(resized_trajectories, f)
            

            