import streamlit as st
import pandas as pd
import os
import cv2
from PIL import Image

from utils import create_data_handler, natural_sort_key

st.set_page_config(layout="wide")
st.title("Trajectory Labeling")

# Initialize session state 
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "selected_trajectory_idx" not in st.session_state:
    st.session_state.selected_trajectory_idx = 0
if "label_input" not in st.session_state:
    st.session_state.label_input = ""

USE_REMOTE = False
data_handler = create_data_handler(USE_REMOTE)

drive_structure = data_handler.get_structure()

subdirs = sorted(drive_structure.keys(), key=natural_sort_key)
if not subdirs:
    st.error("No directories found in the dataset structure.")
    st.stop()


st.session_state.df = data_handler.load_labels()

st.sidebar.write("Traectories labeled: " +
                    str(st.session_state.df["Label"].notna().sum())
                    + " / " + str(len(st.session_state.df)))

selected_dir = st.sidebar.selectbox("Select a directory", subdirs, format_func=lambda x: os.path.basename(x))

# Util function to get the row from contents.csv for a specific file and index
def get_csv_row(file_path, idx):
    return st.session_state.df[
        (st.session_state.df["File Name"] == file_path) & 
        (st.session_state.df["Index"] == idx)
    ]

# Load .npy files from the selected directory
files = sorted(drive_structure[selected_dir], key=natural_sort_key)
if not files:
    st.error(f"No files found in the directory: {selected_dir}")
    st.stop()

def format_file_with_status(file):
    file_name = os.path.basename(file)
    file_path = f"{selected_dir}/{file_name}"

   # Get all rows for this file
    file_rows = st.session_state.df[st.session_state.df["File Name"] == file_path]
    
    if file_rows.empty:
        return file_name  # No labels yet
    
    # Labeled trajectories (not NA and not empty)
    labeled_rows = file_rows[
        file_rows["Label"].notna() & 
        (file_rows["Label"].astype(str).str.strip() != "")
    ]
    
    if labeled_rows.empty:
        return file_name  # No valid labels
    
    # Check if indices are sequential from 0 with no gaps
    indices = sorted(labeled_rows["Index"].tolist())
    max_index = max(indices)
    
    # Complete if: indices start at 0, no gaps, and all have labels
    expected_indices = list(range(max_index + 1))
    is_complete = (
        indices == expected_indices and 
        len(labeled_rows) == len(file_rows)  # All CSV entries have labels
    )
    
    return f"✅ {file_name}" if is_complete else file_name

# Sidebar to select a file
selected_file = st.sidebar.selectbox("Select a file", files, format_func=format_file_with_status)

# Reset trajectory index when file changes
if st.session_state.current_file != selected_file:
    st.session_state.current_file = selected_file
    st.session_state.selected_trajectory_idx = 0

if selected_file:
    # Get relative path for file mapping
    file_name = os.path.basename(selected_file)
    file_path = f"{selected_dir}/{file_name}"
    

    data = data_handler.load_npy_file(file_path)

    st.sidebar.write(f"Number of trajectories: {len(data)}")
    
    # Define callbacks outside of the main flow
    def set_trajectory_idx(idx):
        st.session_state.selected_trajectory_idx = idx
        
        # Reset label input based on the new trajectory
        row = get_csv_row(file_path, idx)
        if not row.empty:
            label = row.iloc[0]["Label"]
            # Convert NaN to empty string
            st.session_state.label_input = "" if pd.isna(label) else str(label)
        else:
            st.session_state.label_input = ""

    def save_label():
        label = st.session_state.label_input
        idx = st.session_state.selected_trajectory_idx
        
        row = get_csv_row(file_path, idx)
        
        if not row.empty:
            st.session_state.df.loc[row.index, "Label"] = label
        else:
            new_row = {
                "File Name": file_path,
                "Index": idx,
                "Label": label,
                "Notes": ""
            }
            st.session_state.df = pd.concat(
                [st.session_state.df, pd.DataFrame([new_row])], 
                ignore_index=True
            )
        
        if not USE_REMOTE:
            data_handler.save_labels(st.session_state.df)
    
        # Advance to next trajectory
        if st.session_state.selected_trajectory_idx < len(data) - 1:
            st.session_state.selected_trajectory_idx += 1
            idx = st.session_state.selected_trajectory_idx
            set_trajectory_idx(idx)

    # Display trajectory buttons in the sidebar
    st.sidebar.write("Select a trajectory:")
    cols = st.sidebar.columns(len(data))  # Create 10 buttons per row in the sidebar

    for idx in range(len(data)):
        # Determine button color based on session state df
        row = get_csv_row(file_path, idx)
        if not row.empty:
            has_label = pd.notna(row.iloc[0]["Label"]) and str(row.iloc[0]["Label"]).strip() != ""
        else:
            has_label = False
        
        button_color = "primary" if has_label else "secondary"
        icon = "✍️" if idx == st.session_state.selected_trajectory_idx else None
        text = f"{idx}" if idx != st.session_state.selected_trajectory_idx else ""


        # Render button with callback
        cols[idx % len(data)].button(
            label=text, 
            key=f"trajectory_{idx}", 
            type=button_color, 
            icon=icon,
            on_click=set_trajectory_idx,
            args=(idx,)
        )

    # Use the updated trajectory index
    trajectory_idx = st.session_state.selected_trajectory_idx

    # Extract the first, middle and last frames and prepare for display
    frames = data_handler.get_trajectory_images(data[trajectory_idx])
    columns = st.columns(3)
    captions = ["First Frame", "Middle Frame", "Last Frame"]

    # Display the first, middle and last frames of the trajectory side by side
    st.write(f"Trajectory {trajectory_idx} from {selected_file}")
    for i, col in enumerate(columns):
        with col:
            img = frames[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            st.image(
                Image.fromarray(img),
                caption=captions[i],
                use_container_width=True,
            )

    # Update label and show the textbox
    set_trajectory_idx(trajectory_idx)
    st.text_input(
        "Enter a label for this trajectory:",
        key="label_input",
        on_change=save_label
    )

if USE_REMOTE:   
    st.sidebar.button(
        "Save Labels",
        on_click=data_handler.save_labels,
        help="Save the current labels to the CSV file.",
        args=(st.session_state.df,)
    )
