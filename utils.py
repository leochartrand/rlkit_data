import re
import json
import pandas as pd
import numpy as np
from io import BytesIO
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from abc import ABC, abstractmethod

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

@st.cache_resource
def setup_drive_service():
    try:
        service_account_info = dict(st.secrets["google_service_account"])

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=[
                'https://www.googleapis.com/auth/drive.file',  # Access to files created by the app
                'https://www.googleapis.com/auth/drive'        # Full drive access (if needed)
            ]
        )

        return build('drive', 'v3', credentials=credentials)

    except Exception as e:
        st.error(f"Failed to setup Google Drive service: {str(e)}")
        return None

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
    
class DataHandler(ABC):    
    @abstractmethod
    def get_structure(self):
        pass
    
    @abstractmethod
    def load_npy_file(self, file_path):
        """Load trajectory data"""
        pass
    
    @abstractmethod
    def load_labels(self):
        """Load labels dataframe"""
        pass
    
    @abstractmethod
    def save_labels(self, dataframe):
        """Save labels dataframe"""
        pass
    
    @abstractmethod
    def get_trajectory_images(self, trajectory_data, trajectory_idx):
        """Extract first, middle, last images from trajectory"""
        pass

class LocalDataHandler(DataHandler):
    def __init__(self):
        self.structure = get_dataset_structure()[0]

    def get_structure(self):
        return self.structure
    
    def load_labels(_self):
        df = pd.read_csv("data/labels.csv")
        return df

    def load_npy_file(_self, file_path):
        data = np.load("data/"+file_path, allow_pickle=True)
        return data
        
    def save_labels(_self, dataframe: pd.DataFrame):
        """Update local labels csv file"""
        dataframe.to_csv("data/labels.csv", index=False)
            
    def get_trajectory_images(self, trajectory_data):
        frames = []
        observations = trajectory_data["observations"]
        frames.append(observations[0]["image_observation"]) 
        frames.append(observations[len(observations) // 2]["image_observation"]) 
        frames.append(observations[-1]["image_observation"])
        return frames

class RemoteDataHandler(DataHandler):
    def __init__(self):
        self.service = setup_drive_service()
        self.structure, self.file_mapping = get_dataset_structure()
        self.folder_id = "1PjNpmwehQpbpKCx_tf-44j1tipI6Zn9U"
        self.labels_id = "1_WWcll8dOMn0NlOap_PQkTl8C2Staa9H"

    def get_structure(self):
        return self.structure

    def download_file(self, file_id):
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = request.execute()
            return BytesIO(file_content)
        except Exception as e:
            st.error(f"Failed to download file {file_id}: {str(e)}")
            return None
        
    @st.cache_data(ttl=600, max_entries=1, show_spinner=False)
    def load_labels(_self):
        with st.spinner("Downloading labels file..."):
            # Download file content using API
            bytes = _self.download_file(_self.labels_id)
        df = pd.read_csv(bytes)
        return df

    @st.cache_data(ttl=600, max_entries=1, show_spinner=False)
    def load_npy_file(_self, file_path):

        # Load from Google Drive
        file_id = _self.file_mapping.get(file_path)
        if not file_id:
            st.error(f"File ID not found for {file_path}")
            st.stop()
        with st.spinner("Downloading file..."):
            # Download file content using API
            bytes = _self.download_file(file_id)
        # Load numpy array from bytes
        data = np.load(bytes, allow_pickle=True)
        return data
        
    def save_labels(_self, dataframe):
        """Upload or update CSV file on Google Drive"""
        try:
            # Convert dataframe to CSV string
            csv_content = dataframe.to_csv(index=False)
            
            # Create file metadata
            file_metadata = {'name': "labels.csv"}
        
            # Encode string to bytes
            csv_bytes = csv_content.encode('utf-8')
            
            media = MediaIoBaseUpload(
                BytesIO(csv_bytes),
                mimetype='text/csv',
                resumable=True
            )

            with st.spinner("Saving labels..."):
                # Update existing file
                file = _self.service.files().update(
                    fileId=_self.labels_id,
                    body=file_metadata,
                    media_body=media
                ).execute()
                return file.get('id')
                    
        except Exception as e:
            st.error(f"Failed to upload CSV: {str(e)}")
            return None
        
    def get_trajectory_images(self, trajectory_data):
        frames = []
        frames.append([
            trajectory_data[0]["hires_image_observation"],
            trajectory_data[1]["hires_image_observation"],
            trajectory_data[2]["hires_image_observation"],
        ])
        return frames[0]
        
def create_data_handler(use_remote):
    return RemoteDataHandler() if use_remote else LocalDataHandler()