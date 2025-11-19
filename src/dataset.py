"""
PyTorch Dataset class for multimodal sentiment analysis
Handles loading and preprocessing of image-text pairs
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')



class MultimodalSentimentDataset(Dataset):
    
    def __init__(self,
                 dataframe: pd.DataFrame,
                 image_transform=None,
                 max_text_length: int = 128,
                 filter_missing: bool = True):

        self.df = dataframe.copy()
        self.image_transform = image_transform
        self.max_text_length = max_text_length
        self._fallback = self._make_fallback(image_transform)
        
        self.df = self._filter_valid_samples()
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        print(f"Dataset initialized with {len(self.df)} samples")
    
    def _make_fallback(self, tfm):
        img = Image.new("RGB", (224, 224), (0, 0, 0))
        if tfm is not None:
            return tfm(img)
        return transforms.ToTensor()(img)

    def _filter_valid_samples(self):
        df = self.df.copy()
        # Filter rows where image path is invalid
        def path_ok(img_path):
            try:
                return isinstance(img_path, str) and Path(img_path).exists()
            except Exception:
                print(f'{img_path} does not exist')
                return False
        initial_count = len(df)
        df = df[df["image_path"].apply(path_ok)]
        filtered_count = len(df)
        print(f"[Data] Kept {filtered_count}/{initial_count} rows with valid image files.")

        # Filter out rows with invalid images
        def valid_image(p: str) -> bool:
            try:
                with Image.open(p) as im:
                    im.verify()
                return True
            except (UnidentifiedImageError, OSError, ValueError):
                return False
        valid_df = df["image_path"].apply(valid_image)
        bad_count = int((~valid_df).sum())
        if bad_count:
            bad_examples = df.loc[~valid_df, "image_path"].head(5).tolist()
            print(f"[Data] Removing {bad_count} corrupted images. Examples: {bad_examples}")
        df = df[valid_df].reset_index(drop=True)

        # Filter rows without sentiment_label
        df = df.copy()
        sentiment_map = {'positive': 2, 'pos': 2, 'neutral': 1, 'neu': 1, 'negative': 0, 'neg': 0}
        sentiment_label = df['image_sentiment'].astype(str).str.lower().map(sentiment_map)
        df['sentiment_label'] = sentiment_label
        filtered_df = df['sentiment_label'].notna()
        df = df[filtered_df]
        df['sentiment_label'] = df['sentiment_label'].astype(int)
        return df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict:
        row = self.df.iloc[idx]

        # Image load with robust fallback
        try:
            with Image.open(row["image_path"]) as img:
                img = img.convert("RGB")
            image = self.image_transform(img) if self.image_transform else transforms.ToTensor()(img)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            # Use transformed black image; keeps batch stats sane
            image = self._fallback

        # Text (kept for future multimodal use; ignored in image-only training)
        text = str(row.get("text", ""))[: self.max_text_length]

        # Label (ensure long dtype in collate)
        label = int(row["sentiment_label"])
        dataset_name = row['dataset'] if 'dataset' in self.df.columns else 'unknown'

        return {"image": image, "text": text, "label": label, "image_path": row["image_path"], 'dataset': dataset_name}
  
    def get_label_distribution(self):
        """Get distribution of labels in the dataset"""
        return self.df['sentiment_label'].value_counts().to_dict()
    
    def get_dataset_info(self):
        """Get dataset statistics"""
        info = {
            'total_samples': len(self.df),
            'label_distribution': self.get_label_distribution(),
            'datasets': self.df['dataset'].value_counts().to_dict() if 'dataset' in self.df.columns else {},
            'avg_text_length': self.df['text'].str.len().mean(),
            'max_text_length': self.df['text'].str.len().max(),
            'min_text_length': self.df['text'].str.len().min()
        }
        return info


class MultimodalDataModule:
    
    def __init__(self,
                 dataframe: pd.DataFrame,
                 image_transform,
                 batch_size: int = 32,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 random_seed: int = 42,
                 num_workers: int = 4):

        self.df = dataframe
        self.image_transform = image_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        # Split data
        self.train_df, self.val_df, self.test_df = self._split_data(
            train_split, val_split, test_split
        )
        
        # Create datasets
        self.train_dataset = MultimodalSentimentDataset(
            self.train_df, image_transform
        )
        self.val_dataset = MultimodalSentimentDataset(
            self.val_df, image_transform
        )
        self.test_dataset = MultimodalSentimentDataset(
            self.test_df, image_transform
        )

        self._pin_memory = torch.cuda.is_available()
    
    def _split_data(self, train_split, val_split, test_split):
        """Split data into train/val/test"""
        # Shuffle
        df_shuffled = self.df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        n = len(df_shuffled)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)
        
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df)} samples ({train_split*100:.1f}%)")
        print(f"  Val:   {len(val_df)} samples ({val_split*100:.1f}%)")
        print(f"  Test:  {len(test_df)} samples ({test_split*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def train_dataloader(self):
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_all_dataloaders(self):
        """Get all dataloaders"""
        return {
            'train': self.train_dataloader(),
            'val': self.val_dataloader(),
            'test': self.test_dataloader()
        }
    
    def get_dataset_info(self):
        """Get information about all datasets"""
        return {
            'train': self.train_dataset.get_dataset_info(),
            'val': self.val_dataset.get_dataset_info(),
            'test': self.test_dataset.get_dataset_info()
        }


def collate_multimodal_batch(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'texts': texts,
        'labels': labels,
        'image_paths': image_paths
    }


def test_dataset():
    print("Testing MultimodalSentimentDataset...")
    
    # Create dummy dataframe
    dummy_data = {
        'image_path': ['test1.jpg', 'test2.jpg', 'test3.jpg'],
        'text': ['This is great!', 'Not good', 'Just okay'],
        'sentiment_label': [2, 0, 1],
        'dataset': ['test', 'test', 'test']
    }
    df = pd.DataFrame(dummy_data)
    
    # Create dataset (will fail on images, but tests structure)
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = MultimodalSentimentDataset(
            df, 
            image_transform=transform,
            filter_missing=False
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Dataset info: {dataset.get_dataset_info()}")
        
    except Exception as e:
        print(f"Expected error (dummy data): {e}")
    
    print("\nDataset test completed!")


if __name__ == "__main__":
    test_dataset()