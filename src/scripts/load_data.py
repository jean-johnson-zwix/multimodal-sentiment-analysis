# -*- coding: utf-8 -*-

import kagglehub
import os, glob, math
import shutil
import pandas as pd
from PIL import Image
from pathlib import Path
import warnings
from collections import Counter
import numpy as np

warnings.filterwarnings('ignore')

class DataCollector:
    
    def __init__(self, base_path='data'):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / 'raw'
        self.processed_path = self.base_path / 'processed'
        
        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def download_datasets(self):
        print("Starting dataset downloads...")
        
        # Download datasets
        mvsa_single = kagglehub.dataset_download('vincemarcs/mvsasingle')
        mvsa_multiple = kagglehub.dataset_download('vincemarcs/mvsamultiple')
        
        print(f'Downloaded mvsa-single to {mvsa_single}')
        print(f'Downloaded mvsa-multiple to {mvsa_multiple}')
        
        # Copy to organized structure
        self._copy_dataset(mvsa_single, 'mvsa-single')
        self._copy_dataset(mvsa_multiple, 'mvsa-multiple')
       
        print("All datasets downloaded and organized!")
        return True
    
    def _copy_dataset(self, source, dest_name):
        dest = self.raw_path / dest_name
        if dest.exists():
            print(f"  {dest_name} already exists, skipping copy...")
            return
        
        try:
            shutil.copytree(source, dest)
            print(f"  Copied {dest_name} successfully")
        except Exception as e:
            print(f"  Error copying {dest_name}: {e}")


class DataLoader:
    
    def __init__(self, base_path='data'):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / 'raw'
        
        # Sentiment label mapping
        self.sentiment_map = {
            'positive': 2,
            'neutral': 1,
            'negative': 0,
            'pos': 2,
            'neu': 1,
            'neg': 0
        }
    
    def load_mvsa_single(self, max_samples=None) -> pd.DataFrame:

        print("Loading MVSA-Single dataset...")
        dataset_path = self.raw_path / 'mvsa-single' / 'MVSA_Single'
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"MVSA-Single not found at {dataset_path}")
        
        data_list = []
        label_file = dataset_path / 'labelResultAll.txt'
        
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        # Read labels
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines[1:]:  # Skip header
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_id = parts[0].strip()
                    labels = parts[1].strip()
                    
                    # Parse "text,image" format
                    label_parts = labels.split(',')
                    if len(label_parts) >= 2:
                        text_sentiment = label_parts[0].strip()
                        image_sentiment = label_parts[1].strip()
                    else:
                        text_sentiment = labels
                        image_sentiment = labels
                    
                    # Get paths
                    text_path = dataset_path / 'data' / f"{img_id}.txt"
                    img_path = dataset_path / 'data' / f"{img_id}.jpg"
                    
                    # Read text content
                    text_content = ""
                    if text_path.exists():
                        try:
                            with open(text_path, 'r', encoding='utf-8', errors='ignore') as tf:
                                text_content = tf.read().strip()
                        except Exception as e:
                            print(f"  Error reading text file {img_id}: {e}")
                    
                    # Check if image exists
                    if img_path.exists() and text_content:
                        data_list.append({
                            'image_path': str(img_path),
                            'text': text_content,
                            'text_sentiment': text_sentiment.lower(),
                            'image_sentiment': image_sentiment.lower(),
                            'sentiment_label': self.sentiment_map.get(text_sentiment.lower(), 1),
                            'dataset': 'mvsa-single'
                        })
                
                # Limit samples if specified
                if max_samples and len(data_list) >= max_samples:
                    break
        
        df = pd.DataFrame(data_list)
        print(f"  Loaded {len(df)} samples from MVSA-Single")
        return df
    
    def load_mvsa_multiple(self, max_samples=None) -> pd.DataFrame:
        print("Loading MVSA-Multiple dataset...")
        dataset_path = self.raw_path / 'mvsa-multiple' / 'MVSA'
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"MVSA-Multiple not found at {dataset_path}")
        
        data_list = []
        label_file = dataset_path / 'labelResultAll.txt'
        
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        # Read labels
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines[1:]:  # Skip header
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_id = parts[0].strip()
                    
                    # Parse multiple annotations (use first annotator's text sentiment)
                    first_annotation = parts[1].strip()
                    label_parts = first_annotation.split(',')
                    
                    if len(label_parts) >= 2:
                        text_sentiment = label_parts[0].strip()
                        image_sentiment = label_parts[1].strip()
                    else:
                        text_sentiment = first_annotation
                        image_sentiment = first_annotation
                    
                    # Get paths
                    text_path = dataset_path / 'data' / f"{img_id}.txt"
                    img_path = dataset_path / 'data' / f"{img_id}.jpg"
                    
                    # Read text content
                    text_content = ""
                    if text_path.exists():
                        try:
                            with open(text_path, 'r', encoding='utf-8', errors='ignore') as tf:
                                text_content = tf.read().strip()
                        except Exception as e:
                            print(f"  Error reading text file {img_id}: {e}")
                    
                    # Check if image exists
                    if img_path.exists() and text_content:
                        data_list.append({
                            'image_path': str(img_path),
                            'text': text_content,
                            'text_sentiment': text_sentiment.lower(),
                            'image_sentiment': image_sentiment.lower(),
                            'sentiment_label': self.sentiment_map.get(text_sentiment.lower(), 1),
                            'dataset': 'mvsa-multiple'
                        })
                
                # Limit samples if specified
                if max_samples and len(data_list) >= max_samples:
                    break
        
        df = pd.DataFrame(data_list)
        print(f"  Loaded {len(df)} samples from MVSA-Multiple")
        return df
    
    
    def load_all_datasets(self, max_samples_per_dataset=None) -> pd.DataFrame:
        print("\n" + "="*50)
        print("Loading All Datasets")
        print("="*50 + "\n")
        
        datasets = []
        
        try:
            df_single = self.load_mvsa_single(max_samples=max_samples_per_dataset)
            datasets.append(df_single)
        except Exception as e:
            print(f"Failed to load MVSA-Single: {e}")
        
        try:
            df_multiple = self.load_mvsa_multiple(max_samples=max_samples_per_dataset)
            datasets.append(df_multiple)
        except Exception as e:
            print(f"Failed to load MVSA-Multiple: {e}")
        
        if not datasets:
            raise ValueError("No datasets were loaded successfully")
        
        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        
        print("\n" + "="*50)
        print(f"Total samples loaded: {len(combined_df)}")
        print("\nDataset distribution:")
        print(combined_df['dataset'].value_counts())
        print("\nSentiment distribution:")
        print(combined_df['sentiment_label'].value_counts())
        print("="*50 + "\n")
        
        return combined_df
    
    def validate_dataset(self, df: pd.DataFrame) -> dict:
        print("Validating dataset...")
        
        stats = {
            'total_samples': len(df),
            'missing_images': 0,
            'missing_text': 0,
            'valid_samples': 0,
            'corrupted_images': []
        }
        
        for idx, row in df.iterrows():
            # Check image
            if pd.isna(row['image_path']) or row['image_path'] == '':
                stats['missing_images'] += 1
            elif row['image_path']:
                if not Path(row['image_path']).exists():
                    stats['missing_images'] += 1
                else:
                    # Try to open image
                    try:
                        img = Image.open(row['image_path'])
                        img.verify()
                    except Exception as e:
                        stats['corrupted_images'].append(row['image_path'])
            
            # Check text
            if pd.isna(row['text']) or row['text'] == '':
                stats['missing_text'] += 1
            
            # Valid sample has both image and text
            if (not pd.isna(row['image_path']) and 
                row['image_path'] != '' and 
                Path(row['image_path']).exists() and
                not pd.isna(row['text']) and 
                row['text'] != ''):
                stats['valid_samples'] += 1
        
        print(f"\nValidation Results:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Valid samples: {stats['valid_samples']}")
        print(f"  Missing images: {stats['missing_images']}")
        print(f"  Missing text: {stats['missing_text']}")
        print(f"  Corrupted images: {len(stats['corrupted_images'])}")
        
        return stats
    
    def save_processed_data(self, df: pd.DataFrame, filename='combined_dataset.csv'):
        """Save processed dataset to CSV"""
        output_path = self.base_path / 'processed' / filename
        df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
        return output_path


def main():
    print("="*60)
    print("MULTIMODAL SENTIMENT ANALYSIS - DATA COLLECTION (FIXED)")
    print("="*60 + "\n")
    
    # Step 1: Download datasets
    collector = DataCollector()
    collector.download_datasets()
    
    print("\n" + "="*60 + "\n")
    
    # Step 2: Load datasets (limit to 1000 per dataset for testing)
    loader = DataLoader()
    combined_df = loader.load_all_datasets(max_samples_per_dataset=1000)
    
    # Step 3: Validate data
    stats = loader.validate_dataset(combined_df)
    
    # Step 4: Save processed data
    output_path = loader.save_processed_data(combined_df)
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    
    return combined_df, stats


if __name__ == "__main__":
    df, stats = main()