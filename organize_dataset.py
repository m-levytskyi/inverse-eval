#!/usr/bin/env python3
"""
Organize MARIA_VIPR_dataset by layer count using existing analysis data.
"""

import shutil
import pandas as pd
from pathlib import Path

def organize_dataset():
    """Organize dataset by copying files to subdirectories based on layer count."""
    
    # Load analysis data
    analysis_file = Path("/home/levytskyi/Documents/reflectorch api playground/maria_dataset_layers.csv")
    df = pd.read_csv(analysis_file)
    
    # Dataset path
    dataset_path = Path("/home/levytskyi/Documents/reflectorch api playground/data/MARIA_VIPR_dataset")
    
    # Create subdirectories
    for layer_count in [0, 1, 2]:
        (dataset_path / str(layer_count)).mkdir(exist_ok=True)
    
    # Organize experiments
    for _, row in df.iterrows():
        experiment_id = row['experiment_id']
        layer_count = row['num_layers']
        
        # Get all files for this experiment
        experiment_files = list(dataset_path.glob(f"{experiment_id}*"))
        
        # Copy files to appropriate subdirectory
        dest_dir = dataset_path / str(layer_count)
        for file_path in experiment_files:
            if file_path.is_file():
                dest_file = dest_dir / file_path.name
                if not dest_file.exists():
                    shutil.copy2(file_path, dest_file)
    
    print(f"Dataset organized into subdirectories 0, 1, 2")

if __name__ == "__main__":
    organize_dataset()
