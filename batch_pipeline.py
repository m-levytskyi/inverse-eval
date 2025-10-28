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
from parameter_discovery import (
    discover_batch_experiments,
    parse_true_parameters_from_model_file,
    discover_experiment_files,
    check_experiment_within_constraints
)
from batch_analysis import (
    create_summary_statistics, 
    print_summary_statistics, 
    print_mape_distribution,
    detect_edge_cases
)
from plotting_utils import create_batch_analysis_plots
from utils import convert_to_json_serializable, ensure_directory_exists
from find_prominent_peaks import find_experiments_with_prominent_peaks

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
DEFAULT_FIX_SLD_MODE = "none"              # SLD fixing mode: "none", "backing", "all"

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
                 use_prominent_features=DEFAULT_USE_PROMINENT_FEATURES, priors_type=None):
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
            narrow_priors_deviation: Deviation for narrow priors (e.g., 0.3 for 30%) or 
                                     constraint percentage for constraint_based priors
            fix_sld_mode: SLD fixing mode - "none", "backing", or "all"
            experiment_ids: List of specific experiment IDs to process (optional)
            use_prominent_features: Whether to use prominent features analysis
            priors_type: Explicit priors type - "broad", "narrow", or "constraint_based" 
                        (overrides use_narrow_priors if provided)
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
        
        # Determine priors type - explicit parameter takes precedence
        if priors_type is not None:
            self.priors_type = priors_type
            self.use_narrow_priors = (priors_type in ["narrow", "constraint_based"])
        else:
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
        if self.priors_type == "constraint_based":
            # Convert deviation to percentage (e.g., 0.99 -> 99)
            percentage = int(self.narrow_priors_deviation * 100)
            return f"{percentage}constraint"
        elif self.priors_type == "narrow":
            # Convert deviation to percentage (e.g., 0.99 -> 99)
            percentage = int(self.narrow_priors_deviation * 100)
            return f"{percentage}priors"
        else:
            return "broadpriors"
    
    def _format_sld_fix_info(self):
        """Format SLD fix information for folder name."""
        if self.fix_sld_mode == "all":
            return "allSLDfix"
        elif self.fix_sld_mode == "backing":
            return "backSLDfix"
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
        If use_prominent_features is enabled, filter experiments by prominent peaks.
        
        Returns:
            List of experiment IDs
        """
        if self.use_prominent_features:
            print(f"\n🔍 PROMINENT FEATURES MODE ENABLED")
            print("=" * 50)
            
            # Find experiments with prominent peaks
            experiments_with_peaks = find_experiments_with_prominent_peaks(
                layer_count=self.layer_count,
                data_directory=str(self.data_directory),
                verbose=True
            )
            
            if not experiments_with_peaks:
                print("⚠️  No experiments with prominent peaks found!")
                return []
            
            # Update experiment count based on filtered results
            original_num = self.num_experiments
            self.num_experiments = len(experiments_with_peaks)
            
            print(f"\n📊 PROMINENT FEATURES FILTERING RESULTS:")
            print(f"  Found {len(experiments_with_peaks)} experiments with prominent peaks")
            print(f"  Original request: {original_num} experiments")
            print(f"  Updated count: {self.num_experiments} experiments")
            
            # If specific experiment_ids were provided, filter them
            if self.experiment_ids:
                # Intersect provided IDs with experiments that have peaks
                filtered_ids = [exp_id for exp_id in self.experiment_ids if exp_id in experiments_with_peaks]
                print(f"  Filtering provided experiment IDs...")
                print(f"  Provided: {len(self.experiment_ids)} experiments")
                print(f"  With prominent peaks: {len(filtered_ids)} experiments")
                
                if not filtered_ids:
                    print("⚠️  None of the provided experiment IDs have prominent peaks!")
                    return []
                
                return filtered_ids
            else:
                # Use the first N experiments with peaks
                selected_experiments = experiments_with_peaks[:self.num_experiments]
                print(f"  Selected first {len(selected_experiments)} experiments with peaks")
                print(f"  Examples: {selected_experiments[:5]}")
                return selected_experiments
        
        else:
            # Standard experiment discovery
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
        Process experiments sequentially with outlier detection for constraint-based priors.
        
        Args:
            experiments: List of experiment IDs
            
        Returns:
            Dictionary of results
        """
        print(f"\nProcessing {len(experiments)} experiments sequentially...")
        
        all_results = {}
        successful_count = 0
        failed_count = 0
        outlier_count = 0
        start_time = time.time()
        
        for i, exp_id in enumerate(experiments, 1):
            print(f"\nProgress: {i}/{len(experiments)} experiments")
            
            # Check for outliers if using constraint-based priors
            if self.priors_type == "constraint_based":
                # Try to get true parameters for outlier checking
                try:
                    data_file, model_file, _ = discover_experiment_files(
                        exp_id, str(self.data_directory), self.layer_count
                    )
                    
                    if model_file:
                        true_params_dict = parse_true_parameters_from_model_file(str(model_file))
                        
                        # Check if experiment is within constraints
                        is_within, outlier_params = check_experiment_within_constraints(
                            exp_id, true_params_dict, self.layer_count, self.narrow_priors_deviation
                        )
                        
                        if not is_within:
                            print(f"  OUTLIER DETECTED: {exp_id}")
                            print(f"  True parameters outside constraint bounds:")
                            for param_name, value, min_bound, max_bound in outlier_params:
                                print(f"    {param_name}: {value:.3f} not in [{min_bound:.3f}, {max_bound:.3f}]")
                            
                            # Record as outlier
                            all_results[exp_id] = {
                                'experiment_id': exp_id,
                                'success': False,
                                'excluded_as_outlier': True,
                                'outlier_parameters': [
                                    {
                                        'parameter': param_name,
                                        'true_value': float(value),
                                        'constraint_min': float(min_bound),
                                        'constraint_max': float(max_bound)
                                    }
                                    for param_name, value, min_bound, max_bound in outlier_params
                                ],
                                'processing_time': 0
                            }
                            outlier_count += 1
                            continue  # Skip processing this experiment
                    
                except Exception as e:
                    print(f"  Warning: Could not check for outliers: {e}")
                    # Continue with normal processing if outlier check fails
            
            # Process experiment normally
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
        if outlier_count > 0:
            print(f"  Excluded as outliers: {outlier_count}")
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
        
        # Separate failed experiments and outliers
        failed_results = {k: v for k, v in all_results.items() 
                         if not v.get('success', False) and not v.get('excluded_as_outlier', False)}
        outlier_results = {k: v for k, v in all_results.items() 
                          if v.get('excluded_as_outlier', False)}
        
        # Save failed experiments (excluding outliers) to separate file
        if failed_results:
            failed_file = self.output_dir / "failed_experiments.json"
            failed_json_results = convert_to_json_serializable(failed_results)
            
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_json_results, f, indent=2)
            
            print(f"  Failed experiments saved to: {failed_file}")
            print(f"  Total failed experiments: {len(failed_results)}")
        else:
            print("  No failed experiments to save")
        
        # Save outliers to separate file
        if outlier_results:
            outlier_file = self.output_dir / "outlier_experiments.json"
            outlier_json_results = convert_to_json_serializable(outlier_results)
            
            with open(outlier_file, 'w', encoding='utf-8') as f:
                json.dump(outlier_json_results, f, indent=2)
            
            print(f"  Outlier experiments saved to: {outlier_file}")
            print(f"  Total outlier experiments: {len(outlier_results)}")
        
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
        
        return results_file, len(successful_results), len(all_results), len(failed_results), len(outlier_results)
    
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
        _, successful_count, total_count, failed_count, outlier_count = self.save_results(all_results)
        
        # Detect edge cases
        successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
        if successful_results:
            detect_edge_cases(successful_results)
        
        # Create plots with outlier and failure statistics
        if successful_results:
            try:
                print(f"\nCreating analysis plots...")
                plot_paths = create_batch_analysis_plots(
                    successful_results, 
                    layer_count=self.layer_count, 
                    output_dir=str(self.output_dir), 
                    save=True,
                    use_prominent_features=self.use_prominent_features,
                    narrow_priors_deviation=self.narrow_priors_deviation if self.use_narrow_priors else 0.5,
                    failed_count=failed_count,
                    outlier_count=outlier_count
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
        if outlier_count > 0:
            print(f"Excluded as outliers: {outlier_count}")
        if failed_count > 0:
            print(f"Failed experiments: {failed_count}")
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
    parser.add_argument('--fix-sld-mode', type=str, choices=['none', 'backing', 'all'], 
                       default=DEFAULT_FIX_SLD_MODE,
                       help=f'SLD fixing mode: none, backing, or all (default: {DEFAULT_FIX_SLD_MODE})')
    parser.add_argument('--use-prominent-features', action='store_true',
                       help='Enable prominent features analysis')
    parser.add_argument('--priors-deviation', type=int, choices=[5, 30, 99], default=99,
                       help='Prior bounds deviation percentage: 5, 30, or 99 (default: 99)')
    parser.add_argument('--priors-type', type=str, choices=['broad', 'narrow', 'constraint_based'], 
                       help='Prior bounds type: broad, narrow, or constraint_based')
    
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
        use_prominent_features=args.use_prominent_features,
        narrow_priors_deviation=args.priors_deviation / 100.0,  # Convert percentage to decimal
        priors_type=args.priors_type
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
