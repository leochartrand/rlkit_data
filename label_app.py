import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import re
from io import BytesIO
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

@st.cache_resource
def setup_drive_service():
    try:
        # Try to load from Streamlit secrets first (for deployment)
        if hasattr(st, 'secrets') and 'google_service_account' in st.secrets:
            service_account_info = dict(st.secrets["google_service_account"])
        else:
            # Fallback to local file for development
            with open('service-account-key.json', 'r') as f:
                service_account_info = json.load(f)
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        return build('drive', 'v3', credentials=credentials)
    
    except Exception as e:
        st.error(f"Failed to setup Google Drive service: {str(e)}")
        return None

# Function for natural sorting
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

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
subdirs = sorted(drive_structure.keys(), key=natural_sort_key)

@st.cache_data(ttl=600)
def load_npy_from_drive(file_id):
    service = setup_drive_service()
    
    try:
        # Download file content using API
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()
        
        # Load numpy array from bytes
        return np.load(BytesIO(file_content), allow_pickle=True)
    
    except Exception as e:
        st.error(f"Failed to download file {file_id}: {str(e)}")
        return None

if not subdirs:
    st.error("No subdirectories found in the data directory.")
else:
    # Initialize session state for the selected trajectory index and label input
    if "selected_trajectory_idx" not in st.session_state:
        st.session_state.selected_trajectory_idx = 0
    if "label_input" not in st.session_state:
        st.session_state.label_input = ""

    # Sidebar to select a directory (remove "data/" prefix for display)
    subdir_labels = [os.path.basename(d) for d in subdirs]
    selected_dir = st.sidebar.selectbox("Select a directory", subdirs, format_func=lambda x: os.path.basename(x))

    # Load .npy files from the selected directory
    files = sorted(drive_structure[selected_dir], key=natural_sort_key)
    # files = sorted(glob.glob(os.path.join(selected_dir, "*.npy")), key=natural_sort_key)
    if not files:
        st.error("No .npy files found in the selected directory.")
    else:
        # Sidebar to select a file
        selected_file = st.sidebar.selectbox("Select a file", files, format_func=lambda x: os.path.basename(x))

        if selected_file:
            # Get relative path for file mapping
            subdir_name = os.path.basename(selected_dir)
            file_name = os.path.basename(selected_file)
            file_path = f"{subdir_name}/{file_name}"
            
            # Load from Google Drive
            file_id = file_mapping.get(file_path)
            if file_id:
                data = load_npy_from_drive(file_id)
            else:
                st.error(f"File ID not found for {file_path}")
                data = None

            if data is None:
                st.stop()
            st.sidebar.write(f"Number of trajectories: {len(data)}")

            # Load the CSV file
            csv_file = os.path.join(selected_dir, "contents.csv")
            if os.path.exists(csv_file):
                st.session_state.df = pd.read_csv(csv_file)
            else:
                # Create a new DataFrame if the file doesn't exist
                st.session_state.df = pd.DataFrame(columns=["File Name", "Index", "Label", "Notes"])

            # Define callbacks outside of the main flow
            def set_trajectory_idx(idx):
                st.session_state.selected_trajectory_idx = idx
                
                # Reset label input based on the new trajectory
                row = st.session_state.df[
                    (st.session_state.df["File Name"] == os.path.basename(selected_file)) & 
                    (st.session_state.df["Index"] == idx)
                ]
                if not row.empty:
                    st.session_state.label_input = row.iloc[0]["Label"]
                else:
                    st.session_state.label_input = ""

            def save_label():
                label = st.session_state.label_input
                file_name = os.path.basename(selected_file)
                idx = st.session_state.selected_trajectory_idx
                
                row = st.session_state.df[
                    (st.session_state.df["File Name"] == file_name) & 
                    (st.session_state.df["Index"] == idx)
                ]
                
                if not row.empty:
                    st.session_state.df.loc[row.index, "Label"] = label
                else:
                    new_row = {
                        "File Name": file_name,
                        "Index": idx,
                        "Label": label,
                        "Notes": ""
                    }
                    st.session_state.df = pd.concat(
                        [st.session_state.df, pd.DataFrame([new_row])], 
                        ignore_index=True
                    )

                # Save the updated CSV
                st.session_state.df.to_csv(csv_file, index=False)
                st.success(f"Label saved to {csv_file}!")

            # Display trajectory buttons in the sidebar
            st.sidebar.write("Select a trajectory:")
            cols = st.sidebar.columns(10)  # Create 10 buttons per row in the sidebar

            for idx in range(len(data)):
                # Determine button color based on session state df
                has_label = not st.session_state.df[
                    (st.session_state.df["File Name"] == os.path.basename(selected_file)) & 
                    (st.session_state.df["Index"] == idx)
                ].empty
                
                button_color = "primary" if has_label else "secondary"
                icon = "✍️" if idx == st.session_state.selected_trajectory_idx else None

                # Render button with callback
                cols[idx % 10].button(
                    f"{idx}", 
                    key=f"trajectory_{idx}", 
                    type=button_color, 
                    icon=icon,
                    on_click=set_trajectory_idx,
                    args=(idx,)
                )

            # Use the updated trajectory index
            trajectory_idx = st.session_state.selected_trajectory_idx
            trajectory = data[trajectory_idx]

            # Display the first and last frames of the trajectory side by side
            if "observations" in trajectory:
                st.write(f"Trajectory {trajectory_idx} from {selected_file}")

                # Get the total number of frames
                total_frames = len(trajectory["observations"])

                if total_frames > 0:
                    # Extract the first and last frames
                    first_frame = trajectory["observations"][0]
                    last_frame = trajectory["observations"][-1]

                    # Create two columns for side-by-side display
                    col1, col2 = st.columns(2)

                    # Display the first frame
                    with col1:
                        if "hires_image_observation" in first_frame:
                            img = first_frame["hires_image_observation"]
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                            st.image(
                                Image.fromarray(img),
                                caption="First Frame",
                                use_container_width=True,
                            )
                        else:
                            st.write("First frame image not available.")

                    # Display the last frame
                    with col2:
                        if "hires_image_observation" in last_frame:
                            img = last_frame["hires_image_observation"]
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                            st.image(
                                Image.fromarray(img),
                                caption="Last Frame",
                                use_container_width=True,
                            )
                        else:
                            st.write("Last frame image not available.")

            # Reset the label input for the new trajectory
            row = st.session_state.df[(st.session_state.df["File Name"] == os.path.basename(selected_file)) & (st.session_state.df["Index"] == st.session_state.selected_trajectory_idx)]
            if not row.empty:
                st.session_state.label_input = row.iloc[0]["Label"]
            else:
                st.session_state.label_input = ""

            # Show current label in the textbox
            st.text_input(
                "Enter a label for this trajectory:",
                value=st.session_state.label_input,
                key="label_input",
                on_change=save_label
            )