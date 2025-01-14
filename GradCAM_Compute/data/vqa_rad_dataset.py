import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class VqaRadDataset(Dataset):
    def __init__(self, xlsx_file, image_path, transform=None):
        # Load the data from the Excel file
        self.data = pd.read_excel(xlsx_file)
        # Filter the dataset to include only rows where Q_TYPE is 'POS'
        self.data = self.data[self.data['Q_TYPE'] == 'POS']
        self.image_path = image_path  # Base path where images are stored
        self.transform = transform  # Optional transform (e.g., for data augmentation)

    def __len__(self):
        # Return the total number of 'POS' samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get a row of data from the filtered dataframe
        row = self.data.iloc[idx]

        # Extract IMAGEID and QUESTION
        image_id = row['IMAGEID']
        question = row['QUESTION']

        # Get the basename of the IMAGEID to use as the image filename
        image_name = os.path.basename(image_id)

        # Construct the full image file path
        img_name = os.path.join(self.image_path, image_name)
        
        return img_name, question
