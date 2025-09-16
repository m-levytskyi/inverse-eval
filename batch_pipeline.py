#!/usr/bin/env python3
"""
Batch inference pipeline for reflectometry experiments.

This module provides a clean interface for running batch processing on multiple
experiments, building upon the single experiment processing in simple_pipeline.
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Import core functionality
from simple_pipeline import run_single_experiment
from parameter_discovery import discover_batch_experiments
from batch_analysis import (
    create_summary_statistics, 
    print_summary_statistics, 
    print_mape_distribution,
    detect_edge_cases
)
from plotting_utils import create_batch_analysis_plots
from utils import convert_to_json_serializable, ensure_directory_exists

# =============================================================================
# CONFIGURATION PARAMETERS - Adjust these as needed
# =============================================================================

# Batch processing configuration
DEFAULT_NUM_EXPERIMENTS = 10
DEFAULT_LAYER_COUNT = 1
DEFAULT_OUTPUT_DIR = "batch_results"
DEFAULT_DATA_DIRECTORY = "data"

# Preprocessing configuration  
DEFAULT_ENABLE_PREPROCESSING = True
DEFAULT_PREPROCESSING_THRESHOLD = 0.75     # Relative error threshold (75%)
DEFAULT_PREPROCESSING_CONSECUTIVE = 5      # Consecutive high-error points
DEFAULT_PREPROCESSING_REMOVE_SINGLES = False

# Constraints configuration
DEFAULT_APPLY_CONSTRAINTS = True           # Apply physical constraints

# SLD fixing configuration
DEFAULT_FIX_SLD_MODE = "all"              # SLD fixing mode: "none", "fronting_backing", "all"

# Prior bounds configuration
USE_NARROW_PRIORS = True                   # Set to False to use broad priors
NARROW_PRIORS_DEVIATION = 0.99            # Deviation for narrow priors
PRIORS_TYPE = "narrow" if USE_NARROW_PRIORS else "broad"

# Prominent features configuration
DEFAULT_USE_PROMINENT_FEATURES = False    # Prominent features analysis

# =============================================================================


class BatchInferencePipeline:
    """
    Batch processing pipeline that leverages simple_pipeline functions.
    """
    
    def __init__(self, num_experiments=DEFAULT_NUM_EXPERIMENTS, layer_count=DEFAULT_LAYER_COUNT, 
                 output_dir=DEFAULT_OUTPUT_DIR, data_directory=DEFAULT_DATA_DIRECTORY, 
                 enable_preprocessing=DEFAULT_ENABLE_PREPROCESSING, apply_constraints=DEFAULT_APPLY_CONSTRAINTS,
                 use_narrow_priors=USE_NARROW_PRIORS, narrow_priors_deviation=NARROW_PRIORS_DEVIATION, 
                 experiment_ids=None, fix_sld_mode=DEFAULT_FIX_SLD_MODE, 
                 use_prominent_features=DEFAULT_USE_PROMINENT_FEATURES):
        """
        Initialize the batch inference pipeline.
        
        Args:
            num_experiments: Number of experiments to process (ignored if experiment_ids provided)
            layer_count: Number of layers (1 or 2)
            output_dir: Output directory for results
            data_directory: Directory containing experimental data
            enable_preprocessing: Whether to enable data preprocessing
            apply_constraints: Whether to apply physical constraints to parameters
            use_narrow_priors: Whether to use narrow priors (requires true parameters)
            narrow_priors_deviation: Deviation for narrow priors (e.g., 0.3 for 30%)
            fix_sld_mode: SLD fixing mode - "none", "fronting_backing", or "all"
            experiment_ids: List of specific experiment IDs to process (optional)
            use_prominent_features: Whether to use prominent features analysis
        """
        self.experiment_ids = experiment_ids
        self.num_experiments = len(experiment_ids) if experiment_ids else num_experiments
        self.layer_count = layer_count
        self.data_directory = Path(data_directory)
        self.enable_preprocessing = enable_preprocessing
        self.apply_constraints = apply_constraints
        self.use_narrow_priors = use_narrow_priors
        self.narrow_priors_deviation = narrow_priors_deviation
        self.fix_sld_mode = fix_sld_mode
        self.use_prominent_features = use_prominent_features
        self.priors_type = "narrow" if use_narrow_priors else "broad"
        
        # Create timestamped output directory in batch_inference_results
        timestamp = datetime.now().strftime("%d%B%Y_%H_%M").lower()
        folder_name = self._generate_folder_name(timestamp)
        self.output_dir = Path("batch_inference_results") / folder_name
        ensure_directory_exists(self.output_dir)
        
        # Create organized output directories
        self.plots_dir = self.output_dir / "plots"
        ensure_directory_exists(self.plots_dir)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Layer count: {self.layer_count}")
        if experiment_ids:
            print(f"Processing specific experiments: {experiment_ids}")
        else:
            print(f"Processing first {self.num_experiments} experiments")
        print(f"Preprocessing: {'enabled' if self.enable_preprocessing else 'disabled'}")
        print(f"Physical constraints: {'enabled' if self.apply_constraints else 'disabled'}")
        print(f"Prior bounds: {self.priors_type}")
        print(f"SLD fixing mode: {self.fix_sld_mode}")
        if self.use_narrow_priors:
            print(f"Narrow priors deviation: ±{self.narrow_priors_deviation*100:.1f}%")
        print(f"Prominent features: {'enabled' if self.use_prominent_features else 'disabled'}")
    
    def _get_next_index(self):
        """Get the next available index by scanning existing directories."""
        batch_results_dir = Path("batch_inference_results")
        if not batch_results_dir.exists():
            return 1
        
        existing_indices = []
        for folder in batch_results_dir.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                # Extract index from folder name (assuming it starts with digits followed by underscore)
                if '_' in folder_name:
                    index_part = folder_name.split('_')[0]
                    if index_part.isdigit():
                        existing_indices.append(int(index_part))
        
        return max(existing_indices, default=0) + 1
    
    def _format_prior_info(self):
        """Format prior information for folder name."""
        if self.use_narrow_priors:
            # Convert deviation to percentage (e.g., 0.99 -> 99)
            percentage = int(self.narrow_priors_deviation * 100)
            return f"{percentage}priors"
        else:
            return "broadpriors"
    
    def _format_sld_fix_info(self):
        """Format SLD fix information for folder name."""
        if self.fix_sld_mode == "all":
            return "allSLDfix"
        elif self.fix_sld_mode == "fronting_backing":
            return "partSLDfix"
        else:  # "none"
            return ""
    
    def _format_prominent_info(self):
        """Format prominent features information for folder name."""
        return "PROMINENT" if self.use_prominent_features else ""
    
    def _generate_folder_name(self, timestamp):
        """Generate the complete folder name with all components."""
        index = self._get_next_index()
        
        # Base experiment info
        if self.experiment_ids:
            exp_info = f"custom_{len(self.experiment_ids)}exps"
        else:
            exp_info = f"{self.num_experiments}exps"
        
        layers_info = f"{self.layer_count}layers"
        prior_info = self._format_prior_info()
        sld_fix_info = self._format_sld_fix_info()
        prominent_info = self._format_prominent_info()
        
        # Build folder name components
        components = [
            f"{index:03d}",  # Zero-padded index (001, 002, etc.)
            exp_info,
            layers_info,
            prior_info
        ]
        
        # Add optional components if not empty
        if sld_fix_info:
            components.append(sld_fix_info)
        if prominent_info:
            components.append(prominent_info)
        
        components.append(timestamp)
        
        return "_".join(components)
    
    def discover_experiments(self):
        """
        Discover available experiments using the parameter_discovery module.
        
        Returns:
            List of experiment IDs
        """
        return discover_batch_experiments(
            data_directory=str(self.data_directory),
            layer_count=self.layer_count,
            num_experiments=self.num_experiments,
            experiment_ids=self.experiment_ids
        )
    
    def process_single_experiment_wrapper(self, experiment_id):
        """
        Wrapper for processing a single experiment with error handling and fallback logic.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with results and metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing experiment: {experiment_id}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # First attempt: Use configured priors (narrow if enabled)
            results = run_single_experiment(
                experiment_id=experiment_id,
                layer_count=self.layer_count,
                enable_preprocessing=self.enable_preprocessing,
                preprocessing_threshold=DEFAULT_PREPROCESSING_THRESHOLD,
                preprocessing_consecutive=DEFAULT_PREPROCESSING_CONSECUTIVE,
                preprocessing_remove_singles=DEFAULT_PREPROCESSING_REMOVE_SINGLES,
                apply_constraints=self.apply_constraints,
                priors_type=self.priors_type,
                priors_deviation=self.narrow_priors_deviation if self.use_narrow_priors else 0.5,
                fix_sld_mode=self.fix_sld_mode
            )
            
            # Success with primary priors
            processing_time = time.time() - start_time
            results['experiment_id'] = experiment_id
            results['processing_time'] = processing_time
            results['success'] = True
            results['priors_used'] = self.priors_type
            results['fallback_applied'] = False
            
            print(f"  ✅ {experiment_id} completed successfully in {processing_time:.1f}s")
            return results
            
        except Exception as primary_error:
            primary_error_msg = str(primary_error)
            print(f"  ❌ Failed with {self.priors_type} priors: {primary_error_msg}")
            
            # Check if it's a negative parameter error and we can fallback
            if (self.use_narrow_priors and 
                ("Negative roughness encountered" in primary_error_msg or 
                 "Negative thickness encountered" in primary_error_msg)):
                
                print(f"  🔄 Detected negative parameter error - falling back to broad priors...")
                
                # Second attempt: Broad priors fallback
                try:
                    results = run_single_experiment(
                        experiment_id=experiment_id,
                        layer_count=self.layer_count,
                        enable_preprocessing=self.enable_preprocessing,
                        preprocessing_threshold=DEFAULT_PREPROCESSING_THRESHOLD,
                        preprocessing_consecutive=DEFAULT_PREPROCESSING_CONSECUTIVE,
                        preprocessing_remove_singles=DEFAULT_PREPROCESSING_REMOVE_SINGLES,
                        apply_constraints=self.apply_constraints,
                        priors_type="broad",
                        priors_deviation=0.5,
                        fix_sld_mode="none"  # Disable SLD fixing in fallback mode
                    )
                    
                    # Success with broad priors fallback
                    processing_time = time.time() - start_time
                    results['experiment_id'] = experiment_id
                    results['processing_time'] = processing_time
                    results['success'] = True
                    results['priors_used'] = "broad"
                    results['fallback_applied'] = True
                    results['primary_error'] = primary_error_msg
                    
                    print(f"  ✅ {experiment_id} completed with broad priors (fallback) in {processing_time:.1f}s")
                    return results
                    
                except Exception as broad_error:
                    print(f"  ❌ Failed with broad priors too: {str(broad_error)}")
                    # Fall through to general error handling
            
            # Complete failure
            processing_time = time.time() - start_time
            return {
                'experiment_id': experiment_id,
                'processing_time': processing_time,
                'success': False,
                'error': primary_error_msg,
                'error_type': type(primary_error).__name__
            }
    
    def process_experiments_sequential(self, experiments):
        """
        Process experiments sequentially.
        
        Args:
            experiments: List of experiment IDs
            
        Returns:
            Dictionary of results
        """
        print(f"\nProcessing {len(experiments)} experiments sequentially...")
        
        all_results = {}
        successful_count = 0
        failed_count = 0
        start_time = time.time()
        
        for i, exp_id in enumerate(experiments, 1):
            print(f"\nProgress: {i}/{len(experiments)} experiments")
            
            result = self.process_single_experiment_wrapper(exp_id)
            all_results[exp_id] = result
            
            if result.get('success', False):
                successful_count += 1
            else:
                failed_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\nSequential processing completed!")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Successful: {successful_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Average time per experiment: {total_time/len(experiments):.1f} seconds")
        
        return all_results
    
    def save_results(self, all_results):
        """Save batch processing results to files."""
        print(f"Saving results to {self.output_dir}")
        
        # Save detailed results as JSON
        results_file = self.output_dir / "batch_results.json"
        
        # Convert to JSON serializable format
        json_results = convert_to_json_serializable(all_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  Detailed results saved to: {results_file}")
        
        # Save failed experiments to separate file
        failed_results = {k: v for k, v in all_results.items() if not v.get('success', False)}
        
        if failed_results:
            failed_file = self.output_dir / "failed_experiments.json"
            failed_json_results = convert_to_json_serializable(failed_results)
            
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_json_results, f, indent=2)
            
            print(f"  Failed experiments saved to: {failed_file}")
            print(f"  Total failed experiments: {len(failed_results)}")
        else:
            print("  No failed experiments to save")
        
        # Create and save summary statistics
        successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
        
        if successful_results:
            summary = create_summary_statistics(
                successful_results, 
                self.layer_count, 
                self.enable_preprocessing, 
                self.priors_type, 
                self.narrow_priors_deviation if self.use_narrow_priors else None
            )
            
            # Save summary
            summary_file = self.output_dir / f"batch_summary_{self.layer_count}layer.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            print(f"  Summary statistics saved to: {summary_file}")
            
            # Print summary to console
            print_summary_statistics(summary)
            
            # Print MAPE distribution
            print_mape_distribution(successful_results)
        
        return results_file, len(successful_results), len(all_results)
    
    def run(self):
        """Run the complete batch processing pipeline."""
        print("STARTING BATCH INFERENCE PIPELINE")
        print("="*60)
        
        start_time = time.time()
        
        # Discover experiments
        experiments = self.discover_experiments()
        
        if not experiments:
            print("No experiments found!")
            return None
        
        # Process experiments sequentially
        all_results = self.process_experiments_sequential(experiments)
        
        # Save results
        _, successful_count, total_count = self.save_results(all_results)
        
        # Detect edge cases
        successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
        if successful_results:
            detect_edge_cases(successful_results)
        
        # Create plots
        if successful_results:
            try:
                print(f"\nCreating analysis plots...")
                plot_paths = create_batch_analysis_plots(
                    successful_results, 
                    layer_count=self.layer_count, 
                    output_dir=str(self.output_dir), 
                    save=True
                )
                print(f"Analysis plots completed")
            except Exception as e:
                print(f"Warning: Failed to create plots: {e}")
        
        end_time = time.time()
        total_pipeline_time = end_time - start_time
        
        # Final summary
        print(f"\n{'='*60}")
        print("BATCH PIPELINE COMPLETED")
        print(f"{'='*60}")
        print(f"Total time: {total_pipeline_time:.1f} seconds")
        print(f"Successful experiments: {successful_count}/{total_count}")
        print(f"Success rate: {successful_count/total_count*100:.1f}%")
        print(f"Results saved to: {self.output_dir}")
        
        return all_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch inference on reflectometry experiments")
    parser.add_argument('--num-experiments', type=int, default=10,
                       help='Number of experiments to process (default: 10)')
    parser.add_argument('--layer-count', type=int, choices=[1, 2], default=1,
                       help='Number of layers (1 or 2, default: 1)')
    parser.add_argument('--data-directory', type=str, default='data',
                       help='Data directory path (default: data)')
    parser.add_argument('--disable-preprocessing', action='store_true',
                       help='Disable data preprocessing')
    parser.add_argument('--disable-constraints', action='store_true',
                       help='Disable physical constraints application')
    parser.add_argument('--output-dir', type=str, default='batch_results',
                       help='Output directory (default: batch_results)')
    parser.add_argument('--experiment-ids', type=str, nargs='+',
                       help='Specific experiment IDs to process (e.g., s005156 s004141)')
    parser.add_argument('--fix-sld-mode', type=str, choices=['none', 'fronting_backing', 'all'], 
                       default=DEFAULT_FIX_SLD_MODE,
                       help=f'SLD fixing mode: none, fronting_backing, or all (default: {DEFAULT_FIX_SLD_MODE})')
    parser.add_argument('--use-prominent-features', action='store_true',
                       help='Enable prominent features analysis')
    
    return parser.parse_args()


def main():
    """Main function to run the batch inference pipeline."""
    args = parse_arguments()
    
    # Create batch pipeline
    batch_pipeline = BatchInferencePipeline(
        num_experiments=args.num_experiments,
        layer_count=args.layer_count,
        output_dir=args.output_dir,
        data_directory=args.data_directory,
        enable_preprocessing=not args.disable_preprocessing,
        apply_constraints=not args.disable_constraints,
        fix_sld_mode=args.fix_sld_mode,
        experiment_ids=args.experiment_ids,
        use_prominent_features=args.use_prominent_features
    )
    
    # Run the pipeline
    try:
        batch_pipeline.run()
        print("\nBatch pipeline completed successfully!")
    except KeyboardInterrupt:
        print("\nBatch pipeline interrupted by user")
    except Exception as e:
        print(f"\nBatch pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
