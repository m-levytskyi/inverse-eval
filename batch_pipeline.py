"""
Batch inference pipeline for reflectometry experiments.
Uses the simple_pipeline functions for individual experiment processing.
"""

import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

# Import the simple pipeline functions
from simple_pipeline import run_single_experiment
from plotting_utils import create_batch_analysis_plots

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
DEFAULT_PREPROCESSING_THRESHOLD = 0.75     # Relative error threshold (75%) - Prevents tensor concatenation error
DEFAULT_PREPROCESSING_CONSECUTIVE = 5      # Consecutive high-error points - More conservative truncation
DEFAULT_PREPROCESSING_REMOVE_SINGLES = False

# Prior bounds configuration
USE_NARROW_PRIORS = True                   # Set to False to use broad priors
NARROW_PRIORS_DEVIATION = 0.3              # Deviation for narrow priors (30%)
PRIORS_TYPE = "narrow" if USE_NARROW_PRIORS else "broad"

# =============================================================================


class BatchInferencePipeline:
    """
    Batch processing pipeline that leverages simple_pipeline functions.
    """
    
    def __init__(self, num_experiments=DEFAULT_NUM_EXPERIMENTS, layer_count=DEFAULT_LAYER_COUNT, 
                 output_dir=DEFAULT_OUTPUT_DIR, data_directory=DEFAULT_DATA_DIRECTORY, 
                 enable_preprocessing=DEFAULT_ENABLE_PREPROCESSING, use_narrow_priors=USE_NARROW_PRIORS,
                 narrow_priors_deviation=NARROW_PRIORS_DEVIATION, experiment_ids=None):
        """
        Initialize the batch inference pipeline.
        
        Args:
            num_experiments: Number of experiments to process (ignored if experiment_ids provided)
            layer_count: Number of layers (1 or 2)
            output_dir: Output directory for results
            data_directory: Directory containing experimental data
            enable_preprocessing: Whether to enable data preprocessing
            use_narrow_priors: Whether to use narrow priors (requires true parameters)
            narrow_priors_deviation: Deviation for narrow priors (e.g., 0.3 for 30%)
            experiment_ids: List of specific experiment IDs to process (optional)
        """
        self.experiment_ids = experiment_ids
        self.num_experiments = len(experiment_ids) if experiment_ids else num_experiments
        self.layer_count = layer_count
        self.output_dir = Path(output_dir)
        self.data_directory = Path(data_directory)
        self.enable_preprocessing = enable_preprocessing
        self.use_narrow_priors = use_narrow_priors
        self.narrow_priors_deviation = narrow_priors_deviation
        self.priors_type = "narrow" if use_narrow_priors else "broad"
        
        # Create timestamped output directory in batch_inference_results
        timestamp = datetime.now().strftime("%d%B%Y_%H_%M").lower()
        if experiment_ids:
            folder_name = f"custom_{len(experiment_ids)}experiments_{layer_count}_layer_{timestamp}"
        else:
            folder_name = f"{num_experiments}experiments_{layer_count}_layer_{timestamp}"
        self.output_dir = Path("batch_inference_results") / folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized output directories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Layer count: {self.layer_count}")
        if experiment_ids:
            print(f"Processing specific experiments: {experiment_ids}")
        else:
            print(f"Processing first {self.num_experiments} experiments")
        print(f"Preprocessing: {'enabled' if self.enable_preprocessing else 'disabled'}")
        print(f"Prior bounds: {self.priors_type}")
        if self.use_narrow_priors:
            print(f"Narrow priors deviation: ±{self.narrow_priors_deviation*100:.1f}%")
    
    def _convert_to_json_serializable(self, obj):
        """
        Recursively convert objects to JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(self._convert_to_json_serializable(list(obj)))
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            # Handle objects with attributes (like BasicParams)
            return {k: self._convert_to_json_serializable(v) for k, v in obj.__dict__.items()}
        else:
            # Try to handle other types
            try:
                import json
                json.dumps(obj)  # Test if it's already JSON serializable
                return obj
            except (TypeError, ValueError):
                # If all else fails, convert to string
                return str(obj)
    
    def discover_experiments(self):
        """
        Discover available experiments in the data directory.
        If specific experiment IDs were provided, validate and return those.
        Otherwise, discover experiments from the data directory.
        
        Returns:
            List of experiment IDs
        """
        if self.experiment_ids:
            print(f"\nUsing provided experiment IDs: {self.experiment_ids}")
            
            # Validate that the experiments exist
            validated_experiments = []
            for exp_id in self.experiment_ids:
                if self._experiment_exists(exp_id):
                    validated_experiments.append(exp_id)
                    print(f"  ✓ {exp_id}: found")
                else:
                    print(f"  ✗ {exp_id}: not found - skipping")
            
            if not validated_experiments:
                raise FileNotFoundError("None of the provided experiment IDs were found")
            
            print(f"Validated {len(validated_experiments)}/{len(self.experiment_ids)} experiments")
            return validated_experiments
        
        # Original discovery logic
        print(f"\nDiscovering experiments in {self.data_directory}")
        
        experiments = []
        
        # Try MARIA dataset structure first
        maria_dataset_path = self.data_directory / "MARIA_VIPR_dataset"
        if maria_dataset_path.exists():
            print(f"  Found MARIA dataset: {maria_dataset_path}")
            
            # Search in subdirectories (e.g., 0/, 1/, 2/)
            for layer_dir in maria_dataset_path.iterdir():
                if layer_dir.is_dir() and layer_dir.name.isdigit():
                    print(f"  Checking layer directory: {layer_dir.name}")
                    
                    # Find all experimental data files
                    exp_files = list(layer_dir.glob("s*_experimental_curve.dat"))
                    
                    for exp_file in exp_files:
                        exp_id = exp_file.name.replace('_experimental_curve.dat', '')
                        
                        # Check if model file exists
                        model_file = layer_dir / f"{exp_id}_model.txt"
                        if not model_file.exists():
                            model_file = layer_dir / f"{exp_id}_model.dat"
                        
                        if model_file.exists():
                            experiments.append(exp_id)
                        else:
                            print(f"  Skipping {exp_id}: no model file found")
        
        else:
            print(f"  Directory not found: {maria_dataset_path}")
            
            # Fallback: try to find experiments in test data
            test_data_path = Path(self.data_directory) / "test_data" / str(self.layer_count)
            if test_data_path.exists():
                print(f"  Using test data: {test_data_path}")
                exp_files = list(test_data_path.glob("s*_experimental_curve.dat"))
                for exp_file in exp_files:
                    exp_id = exp_file.name.replace('_experimental_curve.dat', '')
                    experiments.append(exp_id)
        
        print(f"Found {len(experiments)} experiments")
        
        # Limit to requested number
        if len(experiments) > self.num_experiments:
            experiments = experiments[:self.num_experiments]
            print(f"Limited to first {self.num_experiments} experiments")
        
        return experiments
    
    def _experiment_exists(self, experiment_id):
        """
        Check if an experiment exists in the data directory.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            bool: True if experiment files exist
        """
        # Check MARIA dataset structure
        maria_dataset_path = self.data_directory / "MARIA_VIPR_dataset"
        if maria_dataset_path.exists():
            for layer_dir in maria_dataset_path.iterdir():
                if layer_dir.is_dir() and layer_dir.name.isdigit():
                    exp_file = layer_dir / f"{experiment_id}_experimental_curve.dat"
                    model_file = layer_dir / f"{experiment_id}_model.txt"
                    if not model_file.exists():
                        model_file = layer_dir / f"{experiment_id}_model.dat"
                    
                    if exp_file.exists() and model_file.exists():
                        return True
        
        # Check test data structure
        test_data_path = self.data_directory / "test_data" / str(self.layer_count)
        if test_data_path.exists():
            exp_file = test_data_path / f"{experiment_id}_experimental_curve.dat"
            if exp_file.exists():
                return True
        
        return False
    
    def process_single_experiment_wrapper(self, experiment_id):
        """
        Wrapper function for processing a single experiment using simple_pipeline.
        Includes automatic fallback from narrow to broad priors for robustness.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with experiment results
        """
        try:
            print(f"Processing {experiment_id}...")
            
            # First attempt: Use preferred priors (narrow if enabled, broad otherwise)
            if self.use_narrow_priors:
                print(f"  [1/2] Attempting with narrow priors...")
                
                try:
                    results = run_single_experiment(
                        experiment_id=experiment_id,
                        layer_count=self.layer_count,
                        enable_preprocessing=self.enable_preprocessing,
                        preprocessing_threshold=DEFAULT_PREPROCESSING_THRESHOLD,
                        preprocessing_consecutive=DEFAULT_PREPROCESSING_CONSECUTIVE,
                        preprocessing_remove_singles=DEFAULT_PREPROCESSING_REMOVE_SINGLES,
                        priors_type="narrow",
                        priors_deviation=self.narrow_priors_deviation
                    )
                    
                    # Success with narrow priors
                    results['experiment_id'] = experiment_id
                    results['processing_time'] = time.time()
                    results['success'] = True
                    results['priors_used'] = "narrow"
                    results['fallback_applied'] = False
                    
                    # Add prior bounds information to logs
                    if 'prior_bounds' in results:
                        results['prior_bounds_info'] = {
                            'bounds': results['prior_bounds'],
                            'type': 'narrow',
                            'deviation': self.narrow_priors_deviation
                        }
                    
                    print(f"  ✅ {experiment_id} completed with narrow priors")
                    return results
                    
                except Exception as narrow_error:
                    narrow_error_msg = str(narrow_error)
                    print(f"  ❌ Failed with narrow priors: {narrow_error_msg}")
                    
                    # Capture narrow prior bounds for failed experiments
                    narrow_prior_bounds = None
                    narrow_priors_info = None
                    true_params_dict = None
                    try:
                        from parameter_discovery import get_prior_bounds_for_experiment, discover_experiment_files, parse_true_parameters_from_model_file
                        
                        # Get experimental files and true parameters
                        exp_data_file, model_file, layer_count = discover_experiment_files(
                            experiment_id, data_directory="data", layer_count=self.layer_count
                        )
                        
                        if model_file:
                            true_params_dict = parse_true_parameters_from_model_file(str(model_file))
                        
                        # Generate narrow prior bounds
                        narrow_prior_bounds = get_prior_bounds_for_experiment(
                            experiment_id, 
                            true_params_dict, 
                            priors_type="narrow",
                            deviation=self.narrow_priors_deviation,
                            layer_count=self.layer_count
                        )
                        
                        narrow_priors_info = {
                            'bounds': narrow_prior_bounds,
                            'type': 'narrow',
                            'deviation': self.narrow_priors_deviation,
                            'true_params': true_params_dict
                        }
                        
                    except Exception as bounds_error:
                        print(f"    Warning: Could not capture narrow prior bounds: {bounds_error}")
                    
                    # Check if it's a negative parameter error (our known issue)
                    if ("Negative roughness encountered" in narrow_error_msg or 
                        "Negative thickness encountered" in narrow_error_msg):
                        
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
                                priors_type="broad",
                                priors_deviation=0.5
                            )
                            
                            # Success with broad priors fallback
                            results['experiment_id'] = experiment_id
                            results['processing_time'] = time.time()
                            results['success'] = True
                            results['priors_used'] = "broad"
                            results['fallback_applied'] = True
                            results['narrow_error'] = narrow_error_msg
                            
                            # Add prior bounds information to logs (both narrow that failed and broad that succeeded)
                            if 'prior_bounds' in results:
                                results['prior_bounds_info'] = {
                                    'bounds': results['prior_bounds'],
                                    'type': 'broad',
                                    'deviation': 0.5,
                                    'fallback_reason': narrow_error_msg
                                }
                                
                                # Add failed narrow priors info
                                if narrow_priors_info:
                                    results['failed_narrow_priors_info'] = narrow_priors_info
                            
                            print(f"  ✅ {experiment_id} completed with broad priors (fallback)")
                            return results
                            
                        except Exception as broad_error:
                            print(f"  ❌ Failed with broad priors too: {str(broad_error)}")
                            
                            # Capture broad prior bounds for completely failed experiments
                            broad_prior_bounds = None
                            broad_priors_info = None
                            try:
                                # Generate broad prior bounds  
                                broad_prior_bounds = get_prior_bounds_for_experiment(
                                    experiment_id, 
                                    true_params_dict, 
                                    priors_type="broad",
                                    deviation=0.5,
                                    layer_count=self.layer_count
                                )
                                
                                broad_priors_info = {
                                    'bounds': broad_prior_bounds,
                                    'type': 'broad',
                                    'deviation': 0.5,
                                    'true_params': true_params_dict
                                }
                                
                            except Exception as bounds_error:
                                print(f"    Warning: Could not capture broad prior bounds: {bounds_error}")
                            
                            # Both failed - return comprehensive error info with prior bounds
                            failed_result = {
                                'experiment_id': experiment_id,
                                'success': False,
                                'narrow_error': narrow_error_msg,
                                'broad_error': str(broad_error),
                                'priors_used': None,
                                'fallback_applied': True,
                                'processing_time': time.time()
                            }
                            
                            # Add prior bounds information
                            if narrow_priors_info:
                                failed_result['failed_narrow_priors_info'] = narrow_priors_info
                            if broad_priors_info:
                                failed_result['failed_broad_priors_info'] = broad_priors_info
                                
                            return failed_result
                    else:
                        # Non-negative parameter error - don't attempt fallback
                        print(f"  ❌ Non-parameter error - not attempting fallback")
                        
                        # Prepare failed result with prior bounds information
                        failed_result = {
                            'experiment_id': experiment_id,
                            'success': False,
                            'error': narrow_error_msg,
                            'priors_used': "narrow",
                            'fallback_applied': False,
                            'processing_time': time.time()
                        }
                        
                        # Add narrow prior bounds information if captured
                        if narrow_priors_info:
                            failed_result['failed_narrow_priors_info'] = narrow_priors_info
                            
                        return failed_result
            else:
                # Direct broad priors (no fallback needed)
                results = run_single_experiment(
                    experiment_id=experiment_id,
                    layer_count=self.layer_count,
                    enable_preprocessing=self.enable_preprocessing,
                    preprocessing_threshold=DEFAULT_PREPROCESSING_THRESHOLD,
                    preprocessing_consecutive=DEFAULT_PREPROCESSING_CONSECUTIVE,
                    preprocessing_remove_singles=DEFAULT_PREPROCESSING_REMOVE_SINGLES,
                    priors_type="broad",
                    priors_deviation=0.5
                )
                
                results['experiment_id'] = experiment_id
                results['processing_time'] = time.time()
                results['success'] = True
                results['priors_used'] = "broad"
                results['fallback_applied'] = False
                
                # Add prior bounds information to logs
                if 'prior_bounds' in results:
                    results['prior_bounds_info'] = {
                        'bounds': results['prior_bounds'],
                        'type': 'broad',
                        'deviation': 0.5
                    }
                
                print(f"  ✅ {experiment_id} completed with broad priors")
                return results
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"  💥 {experiment_id} failed unexpectedly: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'success': False,
                'error': str(e),
                'priors_used': None,
                'fallback_applied': False,
                'processing_time': time.time()
            }
    
    def process_experiments_sequential(self, experiments):
        """Process experiments sequentially."""
        print(f"\nProcessing {len(experiments)} experiments sequentially...")
        
        all_results = {}
        successful_count = 0
        failed_count = 0
        
        start_time = time.time()
        
        for i, exp_id in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Processing {exp_id}...")
            
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
        json_results = self._convert_to_json_serializable(all_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  Detailed results saved to: {results_file}")
        
        # Save failed experiments to separate file
        failed_results = {k: v for k, v in all_results.items() if not v.get('success', False)}
        
        if failed_results:
            failed_file = self.output_dir / "failed_experiments.json"
            failed_json_results = self._convert_to_json_serializable(failed_results)
            
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_json_results, f, indent=2)
            
            print(f"  Failed experiments saved to: {failed_file}")
            print(f"  Total failed experiments: {len(failed_results)}")
        else:
            print("  No failed experiments to save")
        
        # Create summary statistics
        successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
        
        if successful_results:
            summary = self.create_summary_statistics(successful_results)
            
            # Save with naming convention from batch_inference_pipeline.py  
            summary_file = self.output_dir / f"batch_summary_{self.layer_count}layer.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            print(f"  Summary statistics saved to: {summary_file}")
            
            # Print summary to console
            self.print_summary_statistics(summary)
            
            # Print MAPE distribution
            self.print_mape_distribution(successful_results)
        
        return results_file, len(successful_results), len(all_results)
    
    def create_summary_statistics(self, successful_results):
        """Create simplified summary statistics focused on MAPE with detailed debugging."""
        print("\nGenerating summary statistics...")
        
        # Collect MAPE values with detailed debugging
        mape_values = []
        debug_info = []
        
        for exp_id, result in successful_results.items():
            if 'param_metrics' in result and result['param_metrics']:
                param_metrics = result['param_metrics']
                
                # Debug: Show what data structure we have
                print(f"\nDEBUG - Experiment {exp_id}:")
                print(f"  param_metrics keys: {list(param_metrics.keys())}")
                
                # Check for overall MAPE in different formats
                overall_mape = None
                if 'overall_mape' in param_metrics:
                    overall_mape = param_metrics['overall_mape']
                    print(f"  Found overall_mape: {overall_mape:.2f}%")
                elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                    if 'mape' in param_metrics['overall']:
                        overall_mape = param_metrics['overall']['mape']
                        print(f"  Found overall.mape: {overall_mape:.2f}%")
                
                if overall_mape is not None:
                    mape_values.append(overall_mape)
                    
                    # Debug: Show by_type breakdown if available
                    if 'by_type' in param_metrics:
                        print(f"  Parameter breakdown:")
                        by_type = param_metrics['by_type']
                        for param_type, metrics in by_type.items():
                            if isinstance(metrics, dict) and 'mape' in metrics:
                                print(f"    {param_type}: {metrics['mape']:.2f}%")
                    
                    # Debug: Show individual parameter details if available
                    if 'by_parameter' in param_metrics:
                        print(f"  Individual parameters:")
                        for param_name, metrics in param_metrics['by_parameter'].items():
                            if isinstance(metrics, dict):
                                pred = metrics.get('predicted', 'N/A')
                                true = metrics.get('true', 'N/A')
                                rel_err = metrics.get('relative_error_percent', 'N/A')
                                print(f"    {param_name}: pred={pred}, true={true}, error={rel_err}%")
                
                debug_info.append({
                    'exp_id': exp_id,
                    'overall_mape': overall_mape,
                    'param_metrics': param_metrics
                })
        
        print(f"\nCollected {len(mape_values)} MAPE values from {len(successful_results)} successful experiments")
        
        summary = {
            'total_experiments': len(successful_results),
            'layer_count': self.layer_count,
            'preprocessing_enabled': self.enable_preprocessing,
            'priors_type': self.priors_type,
            'narrow_priors_deviation': self.narrow_priors_deviation if self.use_narrow_priors else None,
            'debug_info': debug_info  # Add debug info to summary
        }
        
        if mape_values:
            summary['parameter_accuracy'] = {
                'overall_mape': {
                    'median': float(np.median(mape_values)),
                    'mean': float(np.mean(mape_values)),
                    'std': float(np.std(mape_values)),
                    'min': float(np.min(mape_values)),
                    'max': float(np.max(mape_values)),
                    'count': len(mape_values)
                }
            }
        
        return summary
    
    def print_summary_statistics(self, summary):
        """Print simplified summary statistics focusing on MAPE."""
        print("\nBATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total successful experiments: {summary['total_experiments']}")
        print(f"Layer count: {summary['layer_count']}")
        print(f"Preprocessing enabled: {summary['preprocessing_enabled']}")
        print(f"Prior bounds: {summary.get('priors_type', 'unknown')}")
        
        if summary.get('narrow_priors_deviation'):
            deviation_percent = summary['narrow_priors_deviation'] * 100
            print(f"Narrow priors deviation: ±{deviation_percent:.1f}%")
        
        if 'parameter_accuracy' in summary and summary['parameter_accuracy']:
            print("\nParameter Accuracy (MAPE):")
            mape_stats = summary['parameter_accuracy']['overall_mape']
            print(f"  Median: {mape_stats['median']:.2f}%")
            print(f"  Mean: {mape_stats['mean']:.2f}% ± {mape_stats['std']:.2f}%")
            print(f"  Range: {mape_stats['min']:.2f}% - {mape_stats['max']:.2f}%")
            print(f"  Experiments: {mape_stats['count']}")
        else:
            print("\nNo MAPE data available")
    
    def print_mape_distribution(self, successful_results):
        """Print MAPE distribution summary using real overall MAPE values."""
        # Collect real overall MAPE values
        mape_values = []
        
        for result in successful_results.values():
            if 'param_metrics' in result and result['param_metrics']:
                param_metrics = result['param_metrics']
                
                # Get real overall MAPE
                overall_mape = None
                if 'overall_mape' in param_metrics:
                    overall_mape = param_metrics['overall_mape']
                elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                    if 'mape' in param_metrics['overall']:
                        overall_mape = param_metrics['overall']['mape']
                
                if overall_mape is not None:
                    mape_values.append(overall_mape)
        
        if not mape_values:
            print("\nNo MAPE data available")
            return
        
        print("\nREAL MAPE DISTRIBUTION:")
        print("-" * 35)
        
        # Count experiments in quality ranges
        excellent = sum(1 for mape in mape_values if mape < 5)
        good = sum(1 for mape in mape_values if 5 <= mape < 10)
        acceptable = sum(1 for mape in mape_values if 10 <= mape < 20)
        poor = sum(1 for mape in mape_values if mape >= 20)
        
        total = len(mape_values)
        print(f"Excellent (< 5%): {excellent} ({100*excellent/total:.1f}%)")
        print(f"Good (5-10%): {good} ({100*good/total:.1f}%)")
        print(f"Acceptable (10-20%): {acceptable} ({100*acceptable/total:.1f}%)")
        print(f"Poor (≥ 20%): {poor} ({100*poor/total:.1f}%)")
        
        print(f"\nStatistics:")
        print(f"Mean: {np.mean(mape_values):.1f}% ± {np.std(mape_values):.1f}%")
        print(f"Median: {np.median(mape_values):.1f}%")
        print(f"Range: {np.min(mape_values):.1f}% - {np.max(mape_values):.1f}%")
    
    
    def create_mape_distribution_plot(self, successful_results):
        """Create MAPE distribution plot showing real overall MAPE values with debugging."""
        import matplotlib.pyplot as plt
        
        # Collect real overall MAPE values with debugging
        mape_data = {'narrow': []}
        
        print("\nDEBUG - MAPE distribution collection:")
        
        for exp_id, result in successful_results.items():
            if 'param_metrics' in result and result['param_metrics']:
                param_metrics = result['param_metrics']
                
                # Get the real overall MAPE - no artificial calculations
                overall_mape = None
                if 'overall_mape' in param_metrics:
                    overall_mape = param_metrics['overall_mape']
                    print(f"  {exp_id}: overall_mape = {overall_mape:.2f}%")
                elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                    if 'mape' in param_metrics['overall']:
                        overall_mape = param_metrics['overall']['mape']
                        print(f"  {exp_id}: overall.mape = {overall_mape:.2f}%")
                
                if overall_mape is not None:
                    mape_data['narrow'].append(overall_mape)
        
        if not mape_data['narrow']:
            print("No MAPE data available for plotting")
            return
        
        mapes = mape_data['narrow']
        print(f"\nCollected {len(mapes)} real MAPE values")
        print(f"MAPE range: {np.min(mapes):.1f}% - {np.max(mapes):.1f}%")
        print(f"Mean MAPE: {np.mean(mapes):.1f}% ± {np.std(mapes):.1f}%")
        print(f"Median MAPE: {np.median(mapes):.1f}%")
        
        # Create distribution plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'REAL MAPE Distribution Analysis - {len(successful_results)} {self.layer_count}-Layer Experiments\n'
                    f'(Narrow Priors ±{int(self.narrow_priors_deviation * 100)}%) - No Artificial Scaling', 
                    fontsize=16, fontweight='bold')
        
        # Define MAPE ranges
        mape_ranges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
        range_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-40%', '40-50%', '50%+']
        
        # Count experiments in each MAPE range
        counts = []
        for i in range(len(mape_ranges) - 1):
            count = sum(1 for mape in mapes if mape_ranges[i] <= mape < mape_ranges[i+1])
            counts.append(count)
        
        # Add count for 50%+ range
        counts.append(sum(1 for mape in mapes if mape >= 50))
        
        # Create bar chart
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(counts)))
        bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                percentage = (count / len(mapes)) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('MAPE Range')
        ax.set_ylabel('Number of Experiments')
        ax.set_xticks(range(len(range_labels)))
        ax.set_xticklabels(range_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        if mapes:
            stats_text = f'Total: {len(mapes)} experiments\n'
            stats_text += f'Mean MAPE: {np.mean(mapes):.1f}%\n'
            stats_text += f'Median MAPE: {np.median(mapes):.1f}%\n'
            stats_text += f'Std Dev: {np.std(mapes):.1f}%\n'
            stats_text += f'Min MAPE: {np.min(mapes):.1f}%\n'
            stats_text += f'Max MAPE: {np.max(mapes):.1f}%'
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   ha='right', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / f"mape_distribution_{self.layer_count}layer.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MAPE distribution plot saved to: {plot_file}")
        
        return plot_file
    
    def create_parameter_breakdown_plot(self, successful_results):
        """Create parameter-specific MAPE breakdown plot with proper debugging."""
        import matplotlib.pyplot as plt
        
        # Collect parameter-specific MAPE values from by_type structure
        param_mapes = {
            'thickness': [],
            'roughness': [], 
            'sld': [],
            'overall': []
        }
        
        print("\nDEBUG - Parameter breakdown collection:")
        
        for exp_id, result in successful_results.items():
            if 'param_metrics' in result and result['param_metrics']:
                param_metrics = result['param_metrics']
                
                print(f"\nExperiment {exp_id}:")
                
                # Overall MAPE
                if 'overall_mape' in param_metrics:
                    overall_mape = param_metrics['overall_mape']
                    param_mapes['overall'].append(overall_mape)
                    print(f"  Overall MAPE: {overall_mape:.2f}%")
                elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                    if 'mape' in param_metrics['overall']:
                        overall_mape = param_metrics['overall']['mape']
                        param_mapes['overall'].append(overall_mape)
                        print(f"  Overall MAPE: {overall_mape:.2f}%")
                
                # Individual parameter MAPEs from by_type structure
                if 'by_type' in param_metrics:
                    by_type = param_metrics['by_type']
                    print(f"  by_type data:")
                    for param_type in ['thickness', 'roughness', 'sld']:
                        if param_type in by_type and isinstance(by_type[param_type], dict):
                            if 'mape' in by_type[param_type]:
                                mape_val = by_type[param_type]['mape']
                                param_mapes[param_type].append(mape_val)
                                print(f"    {param_type}: {mape_val:.2f}%")
                            else:
                                print(f"    {param_type}: no MAPE data")
                        else:
                            print(f"    {param_type}: not found in by_type")
        
        # Filter out empty parameter types
        param_mapes = {k: v for k, v in param_mapes.items() if v}
        
        print(f"\nFinal parameter counts:")
        for param_type, values in param_mapes.items():
            print(f"  {param_type}: {len(values)} values")
        
        if not param_mapes:
            print("No parameter-specific MAPE data available for plotting")
            return
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        param_names = list(param_mapes.keys())
        param_values = [param_mapes[name] for name in param_names]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'][:len(param_names)]
        
        # Create box plot
        box_plot = ax.boxplot(param_values, tick_labels=param_names, patch_artist=True,
                             showfliers=True, flierprops=dict(marker='o', markerfacecolor='red', 
                             markersize=5, alpha=0.5))
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Parameter Type', fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title(f'Parameter-Specific MAPE Distribution - {len(successful_results)} {self.layer_count}-Layer Experiments\n'
                    f'(Narrow Priors ±{int(self.narrow_priors_deviation * 100)}%) - DEBUG MODE', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistical annotations
        for i, (name, values) in enumerate(zip(param_names, param_values)):
            if values:  # Only add annotation if there are values
                median_val = np.median(values)
                mean_val = np.mean(values)
                ax.text(i + 1, ax.get_ylim()[1] * 0.95, 
                       f'Med: {median_val:.1f}%\nMean: {mean_val:.1f}%\nN: {len(values)}', 
                       ha='center', va='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / f"parameter_breakdown_{self.layer_count}layer.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter breakdown plot saved to: {plot_file}")
        
        return plot_file
    
    def detect_edge_cases(self, successful_results):
        """Detect edge cases with poor performance using real MAPE values."""
        edge_cases = []
        
        print("\nDEBUG - Edge case detection:")
        
        for exp_name, result in successful_results.items():
            if 'param_metrics' not in result or not result['param_metrics']:
                continue
                
            param_metrics = result['param_metrics']
            
            # Get real overall MAPE
            overall_mape = None
            if 'overall_mape' in param_metrics:
                overall_mape = param_metrics['overall_mape']
            elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                if 'mape' in param_metrics['overall']:
                    overall_mape = param_metrics['overall']['mape']
            
            if overall_mape is not None:
                print(f"  {exp_name}: {overall_mape:.1f}% MAPE")
                
                # Flag as edge case if MAPE > 50%
                if overall_mape > 50:
                    # Get individual parameter details if available
                    thickness_mape = None
                    roughness_mape = None
                    sld_mape = None
                    
                    if 'by_type' in param_metrics:
                        by_type = param_metrics['by_type']
                        thickness_mape = by_type.get('thickness', {}).get('mape', 0)
                        roughness_mape = by_type.get('roughness', {}).get('mape', 0)
                        sld_mape = by_type.get('sld', {}).get('mape', 0)
                    
                    edge_cases.append({
                        'experiment': exp_name,
                        'overall_mape': overall_mape,
                        'thickness_mape': thickness_mape,
                        'roughness_mape': roughness_mape,
                        'sld_mape': sld_mape
                    })
        
        # Sort by worst performance
        edge_cases.sort(key=lambda x: x['overall_mape'], reverse=True)
        
        if edge_cases:
            print(f"\n🚨 Edge Cases Detected ({len(edge_cases)} experiments with MAPE > 50%):")
            print("-" * 80)
            for i, case in enumerate(edge_cases[:5], 1):  # Show top 5 worst
                print(f"{i}. {case['experiment']}")
                print(f"   Overall MAPE: {case['overall_mape']:.1f}%")
                if case['thickness_mape'] is not None:
                    print(f"   Thickness: {case['thickness_mape']:.1f}%")
                if case['roughness_mape'] is not None:
                    print(f"   Roughness: {case['roughness_mape']:.1f}%")
                if case['sld_mape'] is not None:
                    print(f"   SLD: {case['sld_mape']:.1f}%")
                print()
        else:
            print("\n✅ No edge cases detected (all experiments < 50% MAPE)")
        
        return edge_cases
    
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
        if successful_count > 0:
            successful_results = {k: v for k, v in all_results.items() if v.get('success', False)}
            self.detect_edge_cases(successful_results)
        
        # Create batch analysis plots if there are successful results
        if successful_count > 0:
            print("\nCreating batch analysis plots...")
            try:
                # Create MAPE distribution plot
                self.create_mape_distribution_plot({k: v for k, v in all_results.items() if v.get('success', False)})
                
                # Create parameter breakdown plot
                self.create_parameter_breakdown_plot({k: v for k, v in all_results.items() if v.get('success', False)})
                
                # Also create the plotting_utils plots
                create_batch_analysis_plots(
                    all_results, 
                    layer_count=self.layer_count, 
                    output_dir=str(self.output_dir), 
                    save=True
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"Warning: Failed to create plots: {e}")
        else:
            print("No successful results - skipping plot generation")
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\nBATCH PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Successful experiments: {successful_count}/{total_count}")
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
    parser.add_argument('--output-dir', type=str, default='batch_results',
                       help='Output directory (default: batch_results)')
    parser.add_argument('--experiment-ids', type=str, nargs='+',
                       help='Specific experiment IDs to process (e.g., s005156 s004141)')
    
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
        experiment_ids=args.experiment_ids
    )
    
    # Run the pipeline
    try:
        results = batch_pipeline.run()
        sys.exit(0 if results else 1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
