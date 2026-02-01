import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FoodDataset(Dataset):
    """Custom Dataset for Food Classification"""
    
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data_frame['Food_Label'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_path = os.path.join(self.img_dir, self.data_frame.iloc[idx]['Image_Path'])
        label = self.data_frame.iloc[idx]['Food_Label']
        label_idx = self.label_to_idx[label]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
    def get_label_name(self, idx):
        """Convert index back to label name"""
        return self.idx_to_label[idx]