"""
PyTorch Dataset for PM2.5 Vision dataset.

Loads images, extracts physics features, and returns tensors for training.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple, Dict, Optional
import pickle
from tqdm import tqdm

from features import extract_all_features


class PM25VisionDataset(Dataset):
    """
    PyTorch Dataset for PM2.5 estimation from images.
    
    Returns:
        - image_tensor: Normalized image tensor (3, 224, 224)
        - physics_features: Physics-based features vector
        - pm25_label: PM2.5 concentration (log-transformed)
    """
    
    def __init__(self, 
                 csv_path: str,
                 image_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 use_log_transform: bool = True,
                 cache_features: bool = True,
                 cache_dir: str = "./cache"):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with image paths and PM2.5 labels
            image_dir: Directory containing images
            transform: torchvision transforms for images
            use_log_transform: Whether to log-transform PM2.5 labels
            cache_features: Whether to cache physics features
            cache_dir: Directory to cache features
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.use_log_transform = use_log_transform
        self.cache_features = cache_features
        self.cache_dir = cache_dir
        
        # Load metadata
        self.metadata = pd.read_csv(csv_path)
        
        # Determine column names (flexible to handle different formats)
        self._detect_columns()
        
        # Set default transform if not provided
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # Feature extraction transform (no augmentation)
        self.feature_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Extract or load cached physics features
        self.physics_features = self._load_or_extract_features()
        
        # Get feature names
        sample_features = extract_all_features(np.zeros((224, 224, 3), dtype=np.uint8))
        self.feature_names = sorted(sample_features.keys())
        
    def _detect_columns(self):
        """Detect column names in metadata CSV."""
        cols = self.metadata.columns.tolist()
        
        # Find image column - prioritize 'filename' over 'image_id'
        if 'filename' in cols:
            self.image_col = 'filename'
        elif 'filepath' in cols:
            self.image_col = 'filepath'
        else:
            # Look for file/path columns but exclude image_id
            image_cols = [c for c in cols if ('file' in c.lower() or 'path' in c.lower()) and 'id' not in c.lower()]
            if not image_cols:
                # Look for image columns but exclude image_id
                image_cols = [c for c in cols if 'image' in c.lower() and 'id' not in c.lower()]
            if image_cols:
                self.image_col = image_cols[0]
            else:
                # Assume first column
                self.image_col = cols[0]
        
        # Find PM2.5 column
        pm25_cols = [c for c in cols if 'pm' in c.lower() and '2' in c]
        if pm25_cols:
            self.pm25_col = pm25_cols[0]
        elif 'pm2_5' in cols:
            self.pm25_col = 'pm2_5'
        elif 'pm25' in cols:
            self.pm25_col = 'pm25'
        elif 'target' in cols:
            self.pm25_col = 'target'
        else:
            raise ValueError(f"Cannot find PM2.5 column in {cols}")
    
    def _filter_missing_files(self):
        """Remove rows with missing image files."""
        original_count = len(self.metadata)
        valid_indices = []
        
        for idx in range(len(self.metadata)):
            img_path = self._get_image_path(idx)
            if os.path.exists(img_path):
                valid_indices.append(idx)
        
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        removed = original_count - len(self.metadata)
        
        if removed > 0:
            print(f"Filtered out {removed} missing files ({len(self.metadata)} valid images remaining)")
        
        print(f"Detected columns - Image: '{self.image_col}', PM2.5: '{self.pm25_col}'")
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transform with ImageNet normalization."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_or_extract_features(self) -> np.ndarray:
        """Load cached features or extract them from images."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, 
                                  f"{os.path.basename(self.csv_path)}_features.pkl")
        
        if self.cache_features and os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            with open(cache_file, 'rb') as f:
                features = pickle.load(f)
            return features
        
        print(f"Extracting physics features from {len(self.metadata)} images...")
        features_list = []
        
        for idx in tqdm(range(len(self.metadata))):
            img_path = self._get_image_path(idx)
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
            
            # Extract features
            features_dict = extract_all_features(image_np)
            features_vector = [features_dict[k] for k in sorted(features_dict.keys())]
            features_list.append(features_vector)
        
        features = np.array(features_list, dtype=np.float32)
        
        # Cache features
        if self.cache_features:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            print(f"Cached features to {cache_file}")
        
        return features
    
    def _get_image_path(self, idx: int) -> str:
        """Get full image path for index."""
        img_name = str(self.metadata.iloc[idx][self.image_col]).strip()
        
        # Handle different path formats
        if os.path.isabs(img_name):
            img_path = os.path.normpath(img_name)
        else:
            img_path = os.path.normpath(os.path.join(self.image_dir, img_name))
        
        # Handle .jpg vs .JPG, etc.
        if not os.path.exists(img_path):
            # Try different extensions
            base = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                test_path = os.path.normpath(base + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        return img_path
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            (image_tensor, physics_features, pm25_label)
        """
        # Load image
        img_path = self._get_image_path(idx)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transform
        image_tensor = self.transform(image)
        
        # Get physics features
        physics_features = torch.tensor(self.physics_features[idx], dtype=torch.float32)
        
        # Get PM2.5 label
        pm25 = self.metadata.iloc[idx][self.pm25_col]
        
        # Log transform if enabled
        if self.use_log_transform:
            pm25 = np.log1p(pm25)
        
        pm25_label = torch.tensor(pm25, dtype=torch.float32)
        
        return image_tensor, physics_features, pm25_label
    
    def get_raw_data(self, idx: int) -> Dict:
        """
        Get raw data for explainability (no transforms).
        
        Returns:
            Dictionary with image, features, label
        """
        img_path = self._get_image_path(idx)
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        pm25 = self.metadata.iloc[idx][self.pm25_col]
        
        features_dict = extract_all_features(image_np)
        
        return {
            'image': image_np,
            'image_path': img_path,
            'features': features_dict,
            'pm25': pm25
        }


def get_train_transforms() -> transforms.Compose:
    """Get training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms() -> transforms.Compose:
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(train_csv: str, val_csv: str, test_csv: str,
                      train_img_dir: str, val_img_dir: str, test_img_dir: str,
                      batch_size: int = 16,
                      num_workers: int = 2,
                      cache_dir: str = "./cache") -> Tuple:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_csv, val_csv, test_csv: Paths to split CSV files
        train_img_dir, val_img_dir, test_img_dir: Image directories
        batch_size: Batch size
        num_workers: Number of workers for data loading
        cache_dir: Directory for caching features
        
    Returns:
        (train_loader, val_loader, test_loader, feature_names)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = PM25VisionDataset(
        train_csv, train_img_dir, 
        transform=get_train_transforms(),
        cache_dir=cache_dir
    )
    
    val_dataset = PM25VisionDataset(
        val_csv, val_img_dir,
        transform=get_val_transforms(),
        cache_dir=cache_dir
    )
    
    test_dataset = PM25VisionDataset(
        test_csv, test_img_dir,
        transform=get_val_transforms(),
        cache_dir=cache_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.feature_names


if __name__ == "__main__":
    # Test dataset
    print("Testing PM25VisionDataset...")
    
    # Create a dummy CSV for testing
    test_csv = "test_metadata.csv"
    test_data = pd.DataFrame({
        'image': ['img1.jpg', 'img2.jpg'],
        'pm2_5': [25.5, 67.3]
    })
    test_data.to_csv(test_csv, index=False)
    
    print(f"Created test CSV: {test_csv}")
    print("Note: To fully test, provide actual image paths and directory.")
    
    # Clean up
    os.remove(test_csv)
    print("Dataset module test completed!")
