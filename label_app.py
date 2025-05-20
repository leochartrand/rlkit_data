import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image
import glob
import pandas as pd
import time
import re

# Function for natural sorting
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Load all subdirectories
data_dir = "data/"
subdirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))], key=natural_sort_key)

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
    files = sorted(glob.glob(os.path.join(selected_dir, "*.npy")), key=natural_sort_key)
    if not files:
        st.error("No .npy files found in the selected directory.")
    else:
        # Sidebar to select a file
        selected_file = st.sidebar.selectbox("Select a file", files, format_func=lambda x: os.path.basename(x))

        if selected_file:
            data = np.load(selected_file, allow_pickle=True)
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