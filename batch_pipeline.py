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
DEFAULT_PREPROCESSING_THRESHOLD = 0.5      # Relative error threshold (50%)
DEFAULT_PREPROCESSING_CONSECUTIVE = 3      # Consecutive high-error points
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
                 narrow_priors_deviation=NARROW_PRIORS_DEVIATION):
        """
        Initialize the batch inference pipeline.
        
        Args:
            num_experiments: Number of experiments to process
            layer_count: Number of layers (1 or 2)
            output_dir: Output directory for results
            data_directory: Directory containing experimental data
            enable_preprocessing: Whether to enable data preprocessing
            use_narrow_priors: Whether to use narrow priors (requires true parameters)
            narrow_priors_deviation: Deviation for narrow priors (e.g., 0.3 for 30%)
        """
        self.num_experiments = num_experiments
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
        
        Returns:
            List of experiment IDs
        """
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
    
    def process_single_experiment_wrapper(self, experiment_id):
        """
        Wrapper function for processing a single experiment using simple_pipeline.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with experiment results
        """
        try:
            print(f"Processing {experiment_id}...")
            
            # Use the simple pipeline function
            results = run_single_experiment(
                experiment_id=experiment_id,
                layer_count=self.layer_count,
                enable_preprocessing=self.enable_preprocessing,
                preprocessing_threshold=0.5,
                preprocessing_consecutive=3,
                preprocessing_remove_singles=False,
                priors_type=self.priors_type,
                priors_deviation=self.narrow_priors_deviation
            )
            
            # Add experiment metadata
            results['experiment_id'] = experiment_id
            results['processing_time'] = time.time()
            results['success'] = True
            
            print(f"  {experiment_id} completed successfully")
            return results
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"  {experiment_id} failed: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'success': False,
                'error': str(e),
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
            
            # Print MAPE distribution summary
            self.print_mape_summary(successful_results)
        
        return results_file, len(successful_results), len(all_results)
    
    def create_summary_statistics(self, successful_results):
        """Create summary statistics from successful results."""
        print("\nGenerating summary statistics...")
        
        # Collect fit metrics
        fit_metrics_list = []
        param_metrics_list = []
        
        for result in successful_results.values():
            if 'fit_metrics' in result and result['fit_metrics']:
                fit_metrics_list.append(result['fit_metrics'])
            
            if 'param_metrics' in result and result['param_metrics']:
                param_metrics_list.append(result['param_metrics'])
        
        summary = {
            'total_experiments': len(successful_results),
            'layer_count': self.layer_count,
            'preprocessing_enabled': self.enable_preprocessing,
            'fit_quality': {},
            'parameter_accuracy': {}
        }
        
        if fit_metrics_list:
            # Calculate fit quality statistics
            r_squared_values = [fm['r_squared'] for fm in fit_metrics_list]
            mse_values = [fm['mse'] for fm in fit_metrics_list]
            
            summary['fit_quality'] = {
                'r_squared': {
                    'mean': float(np.mean(r_squared_values)),
                    'std': float(np.std(r_squared_values)),
                    'min': float(np.min(r_squared_values)),
                    'max': float(np.max(r_squared_values)),
                    'median': float(np.median(r_squared_values))
                },
                'mse': {
                    'mean': float(np.mean(mse_values)),
                    'std': float(np.std(mse_values)),
                    'min': float(np.min(mse_values)),
                    'max': float(np.max(mse_values)),
                    'median': float(np.median(mse_values))
                }
            }
        
        if param_metrics_list:
            # Calculate parameter accuracy statistics
            mape_values = [pm['overall_mape'] for pm in param_metrics_list if 'overall_mape' in pm]
            
            if mape_values:
                summary['parameter_accuracy'] = {
                    'overall_mape': {
                        'mean': float(np.mean(mape_values)),
                        'std': float(np.std(mape_values)),
                        'min': float(np.min(mape_values)),
                        'max': float(np.max(mape_values)),
                        'median': float(np.median(mape_values))
                    }
                }
        
        return summary
    
    def print_summary_statistics(self, summary):
        """Print summary statistics to console."""
        print("\nBATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total successful experiments: {summary['total_experiments']}")
        print(f"Layer count: {summary['layer_count']}")
        print(f"Preprocessing enabled: {summary['preprocessing_enabled']}")
        
        if 'fit_quality' in summary and summary['fit_quality']:
            print("\nFit Quality (R²):")
            r2_stats = summary['fit_quality']['r_squared']
            print(f"  Mean: {r2_stats['mean']:.6f}")
            print(f"  Std:  {r2_stats['std']:.6f}")
            print(f"  Range: {r2_stats['min']:.6f} - {r2_stats['max']:.6f}")
        
        if 'parameter_accuracy' in summary and summary['parameter_accuracy']:
            print("\nParameter Accuracy (MAPE):")
            mape_stats = summary['parameter_accuracy']['overall_mape']
            print(f"  Mean: {mape_stats['mean']:.2f}%")
            print(f"  Std:  {mape_stats['std']:.2f}%")
            print(f"  Range: {mape_stats['min']:.2f}% - {mape_stats['max']:.2f}%")
    
    def print_mape_summary(self, successful_results):
        """Print detailed MAPE distribution summary."""
        # Collect MAPE values
        mape_values = []
        
        for result in successful_results.values():
            if 'param_metrics' in result and result['param_metrics']:
                param_metrics = result['param_metrics']
                if 'overall_mape' in param_metrics:
                    mape_values.append(param_metrics['overall_mape'])
        
        if not mape_values:
            return
        
        print("MAPE DISTRIBUTION SUMMARY:")
        print("-" * 50)
        print(f"Total experiments: {len(mape_values)}")
        print(f"Mean MAPE: {np.mean(mape_values):.1f}% ± {np.std(mape_values):.1f}%")
        print(f"Median MAPE: {np.median(mape_values):.1f}%")
        print(f"Range: {np.min(mape_values):.1f}% - {np.max(mape_values):.1f}%")
        
        # Count experiments in quality ranges
        excellent = sum(1 for mape in mape_values if mape < 5)
        good = sum(1 for mape in mape_values if 5 <= mape < 10)
        acceptable = sum(1 for mape in mape_values if 10 <= mape < 20)
        poor = sum(1 for mape in mape_values if mape >= 20)
        
        print(f"Excellent (< 5%): {excellent} ({100*excellent/len(mape_values):.1f}%)")
        print(f"Good (5-10%): {good} ({100*good/len(mape_values):.1f}%)")
        print(f"Acceptable (10-20%): {acceptable} ({100*acceptable/len(mape_values):.1f}%)")
        print(f"Poor (≥ 20%): {poor} ({100*poor/len(mape_values):.1f}%)")
    
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
        
        # Create batch analysis plots if there are successful results
        if successful_count > 0:
            print("\nCreating batch analysis plots...")
            try:
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
        enable_preprocessing=not args.disable_preprocessing
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
