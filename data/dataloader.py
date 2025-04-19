import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import random
from tqdm import tqdm


class MIMICCXRDataset(Dataset):
    """
    Dataset class for MIMIC-CXR dataset.
    
    Args:
        csv_file (str): Path to the CSV file with image-report pairs.
        root_dir (str): Directory with all the images and reports.
        transform (callable, optional): Optional transform to be applied on images.
        max_length (int): Maximum length of the tokenized report.
        split (str): Train, val, or test split.
        tokenizer_name (str): Name of the pretrained tokenizer.
    """
    def __init__(self, csv_file, root_dir='.', transform=None, max_length=512, 
                 split='train', tokenizer_name='gpt2', test_size=0.1, val_size=0.1, seed=42):
        
        # Load the dataframe
        print(f"Loading data from {csv_file}...")
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Filter the dataframe to include only existing files
        print("Filtering dataset to include only existing files...")
        self.filter_existing_files()
        print(f"After filtering, dataset contains {len(self.data_frame)} valid image-report pairs.")
        
        # Create train/val/test splits
        if split in ['train', 'val', 'test']:
            # Set seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            
            # Get indices for each split
            indices = list(range(len(self.data_frame)))
            np.random.shuffle(indices)
            
            test_idx = int(len(indices) * test_size)
            val_idx = int(len(indices) * val_size)
            
            test_indices = indices[:test_idx]
            val_indices = indices[test_idx:test_idx + val_idx]
            train_indices = indices[test_idx + val_idx:]
            
            if split == 'train':
                self.indices = train_indices
            elif split == 'val':
                self.indices = val_indices
            else:  # test
                self.indices = test_indices
                
            # Filter dataframe for the current split
            self.data_frame = self.data_frame.iloc[self.indices].reset_index(drop=True)
            print(f"Split '{split}' contains {len(self.data_frame)} samples.")
    
    def filter_existing_files(self):
        """
        Filter the dataframe to include only image-report pairs where both files exist.
        """
        valid_indices = []
        
        for idx in tqdm(range(len(self.data_frame)), desc="Checking files"):
            image_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'])
            report_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['report_path'])
            
            if os.path.isfile(image_path) and os.path.isfile(report_path):
                valid_indices.append(idx)
        
        self.data_frame = self.data_frame.iloc[valid_indices].reset_index(drop=True)
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and report path
        image_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'])
        report_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['report_path'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load report text
        with open(report_path, 'r', encoding='utf-8') as f:
            report = f.read().strip()
        
        # Tokenize report
        tokenized_report = self.tokenizer(
            report,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized_report['input_ids'].squeeze()
        attention_mask = tokenized_report['attention_mask'].squeeze()
        
        sample = {
            'image': image,
            'report_text': report,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_path': image_path,
            'report_path': report_path
        }
        
        return sample


def get_transforms(phase):
    """
    Get image transformations based on the phase.
    
    Args:
        phase (str): 'train', 'val', or 'test'.
        
    Returns:
        transforms.Compose: Composition of transforms.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])


def get_dataloader(csv_file, root_dir='.', batch_size=32, num_workers=4, max_length=512,
                  split='train', tokenizer_name='gpt2', test_size=0.1, val_size=0.1, seed=42):
    """
    Create data loaders for training, validation and testing.
    
    Args:
        csv_file (str): Path to the CSV file with image-report pairs.
        root_dir (str): Directory with all the images and reports.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        max_length (int): Maximum length of the tokenized report.
        split (str): 'train', 'val', or 'test' split.
        tokenizer_name (str): Name of the pretrained tokenizer.
        test_size (float): Proportion of the dataset to be used as test set.
        val_size (float): Proportion of the dataset to be used as validation set.
        seed (int): Random seed for reproducibility.
        
    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    transform = get_transforms(split)
    
    dataset = MIMICCXRDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform,
        max_length=max_length,
        split=split,
        tokenizer_name=tokenizer_name,
        test_size=test_size,
        val_size=val_size,
        seed=seed
    )
    
    shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader 