#!/usr/bin/env python3
"""
Run batch inference on all 1-layer experiments using theoretical curves
with 30% narrow priors and all SLDs fixed.
"""

import pandas as pd
from batch_pipeline import BatchInferencePipeline

def get_1_layer_experiment_ids():
    """Extract all 1-layer experiment IDs from the dataset."""
    print("Loading experiment metadata...")
    df = pd.read_csv('maria_dataset_layers.csv')
    
    # Get unique experiment IDs for 1-layer experiments
    one_layer_experiments = df[df['num_layers'] == 1]['experiment_id'].unique()
    
    print(f"Found {len(one_layer_experiments)} 1-layer experiments")
    return sorted(one_layer_experiments.tolist())

def main():
    """Run batch processing on all 1-layer theoretical experiments."""
    
    # Get all 1-layer experiment IDs
    experiment_ids = get_1_layer_experiment_ids()
    
    print("Starting batch processing...")
    print(f"Processing {len(experiment_ids)} 1-layer experiments")
    print("Configuration:")
    print("  - Using theoretical curves")
    print("  - 30% narrow priors")
    print("  - All SLDs fixed")
    print("  - No preprocessing (theoretical data is clean)")
    
    # Create and run batch pipeline
    pipeline = BatchInferencePipeline(
        experiment_ids=experiment_ids,
        layer_count=1,
        enable_preprocessing=False,  # Theoretical data doesn't need preprocessing
        apply_constraints=True,
        use_narrow_priors=True,
        narrow_priors_deviation=0.3,  # 30% deviation
        fix_sld_mode="all",  # Fix all SLDs
        use_prominent_features=True
    )
    
    # Run the batch processing
    pipeline.run()

if __name__ == "__main__":
    main()