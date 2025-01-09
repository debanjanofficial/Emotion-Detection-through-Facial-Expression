import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
from .config import Config

class EmotionDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        try:
            image = np.array(row['pixels'].split(), dtype=np.float32).reshape(48, 48)
            label = int(row['emotion'])
        
            if self.transform:
                image = self.transform(image)
        
            return image, label
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            return None, None

# In dataset.py, modify the load_data function:
def load_data(Config):
    df = pd.read_csv(Config.DATA_PATH)
    # Calculate split sizes
    total_size = len(df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    # Define transforms for training data (with augmentation)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    # Define transforms for validation and test data
    common_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    # Create datasets
    train_dataset = EmotionDataset(train_df, transform=train_transforms)
    val_dataset = EmotionDataset(val_df, transform=common_transforms)
    test_dataset = EmotionDataset(test_df, transform=common_transforms)
    
    return (
        DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    )

