#!/usr/bin/env python3
"""
Batch Inference Pipeline for ReflecTorch Models

Usage:
    python batch_inference_pipeline.py [--num-experiments 25] [--layer-count 2]
"""

import argparse
import json
import random
import pickle
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
from functools import lru_cache
import psutil

# Import the existing inference pipeline
from inference_pipeline import InferencePipeline, NARROW_PRIORS_DEVIATION

def validate_experiment(exp_dir):
    """Standalone function to validate experiment completeness for multiprocessing."""
    exp_dir_path = Path(exp_dir)
    exp_files = list(exp_dir_path.glob('s*_experimental_curve.dat'))
    valid_experiments = []
    
    for exp_file in exp_files:
        exp_id = exp_file.name.replace('_experimental_curve.dat', '')
        model_file = exp_dir_path / f"{exp_id}_model.txt"
        if model_file.exists():
            valid_experiments.append(exp_id)
    
    return valid_experiments

def process_experiment_worker(args):
    """
    Worker function for parallel processing of experiments.
    This runs in a separate process to avoid GIL limitations.
    """
    (exp_id, models, data_directory, layer_count, output_dir) = args
    
    try:
        exp_results = {
            'experiment_id': exp_id,
            'layer_count': layer_count,
            'priors': {},
            'model_times': {}
        }
        
        # Process both broad and narrow priors for this experiment
        for priors_type in ['broad', 'narrow']:
            start_time = time.time()
            
            # Call the original inference
            result = InferencePipeline.run_experiment_inference(
                experiment_id=exp_id,
                models_list=models,
                data_directory=str(data_directory),
                priors_type=priors_type,
                output_dir=str(output_dir),
                layer_count=layer_count
            )
            
            end_time = time.time()
            result['processing_time'] = end_time - start_time
            exp_results['priors'][priors_type] = result
            
            # Extract timing data
            if result['success'] and 'models_results' in result:
                for model_name, model_result in result['models_results'].items():
                    if 'inference_time' in model_result:
                        if model_name not in exp_results['model_times']:
                            exp_results['model_times'][model_name] = {}
                        if priors_type not in exp_results['model_times'][model_name]:
                            exp_results['model_times'][model_name][priors_type] = []
                        exp_results['model_times'][model_name][priors_type].append(
                            model_result['inference_time']
                        )
        
        return exp_results
        
    except Exception as e:
        print(f"    ✗ {exp_id} worker error: {e}")
        return {
            'experiment_id': exp_id,
            'layer_count': layer_count,
            'priors': {},
            'model_times': {},
            'error': str(e),
            'success': False
        }

def process_experiment_worker(args):
    """
    Worker function for parallel processing of experiments.
    This runs in a separate process to avoid GIL limitations.
    """
    (exp_id, models, data_directory, layer_count, output_dir) = args
    
    try:
        exp_results = {
            'experiment_id': exp_id,
            'layer_count': layer_count,
            'priors': {},
            'model_times': {}  # Track individual model times
        }
        
        # Process both broad and narrow priors for this experiment
        for priors_type in ['broad', 'narrow']:
            start_time = time.time()
            result = InferencePipeline.run_experiment_inference(
                experiment_id=exp_id,
                models_list=models,
                data_directory=str(data_directory),
                priors_type=priors_type,
                output_dir=str(output_dir),
                layer_count=layer_count
            )
            end_time = time.time()
            
            result['processing_time'] = end_time - start_time
            exp_results['priors'][priors_type] = result
            
            # Extract individual model times if available
            if result['success'] and 'models_results' in result:
                if 'model_times' not in exp_results:
                    exp_results['model_times'] = {}
                for model_name, model_result in result['models_results'].items():
                    if 'inference_time' in model_result:
                        if model_name not in exp_results['model_times']:
                            exp_results['model_times'][model_name] = {}
                        if priors_type not in exp_results['model_times'][model_name]:
                            exp_results['model_times'][model_name][priors_type] = []
                        exp_results['model_times'][model_name][priors_type].append(model_result['inference_time'])
            
            if result['success']:
                print(f"    ✓ {exp_id} ({priors_type}) completed in {end_time-start_time:.1f}s")
            else:
                print(f"    ✗ {exp_id} ({priors_type}) failed: {result.get('error', 'Unknown error')}")
        
        return exp_results
        
    except Exception as e:
        print(f"    ✗ {exp_id} worker error: {e}")
        return {
            'experiment_id': exp_id,
            'layer_count': layer_count,
            'priors': {},
            'model_times': {},
            'error': str(e),
            'success': False
        }

class BatchInferencePipeline:
    """Significantly optimized batch pipeline with pre-loading and caching."""
    
    def __init__(self, num_experiments=25, layer_count=2, data_directory="data", 
                 enable_parallel=True, max_workers=None, enable_caching=True,
                 batch_size=5, memory_limit_gb=48):
        self.num_experiments = num_experiments
        self.layer_count = layer_count
        self.data_directory = Path(data_directory)
        self.maria_dataset_path = Path(data_directory) / "MARIA_VIPR_dataset"
        
        # Create timestamped output directory
        self.timestamp = datetime.now().strftime("%d%B%Y_%H_%M").lower()
        folder_name = f"{num_experiments}experiments_{layer_count}_layer_{self.timestamp}"
        self.output_dir = Path("batch_inference_results") / folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results will be saved to: {self.output_dir}")
        
        # Performance optimization parameters
        self.enable_parallel = enable_parallel
        self.enable_caching = enable_caching
        self.batch_size = batch_size  # Process experiments in batches to manage memory
        avail_gb = psutil.virtual_memory().available / (1024**3)
        self.memory_limit_gb = memory_limit_gb or (avail_gb * 0.9)
        
        # Auto-detect optimal workers based on system resources
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = mp.cpu_count()
        
        # Conservative worker calculation: 1 worker per 2GB RAM + CPU constraint
        max_memory_workers = max(1, int(available_memory_gb / 2))
        max_cpu_workers = min(cpu_count, 12)  # Cap at 12 for stability
        
        self.max_workers = max_workers or min(max_memory_workers, max_cpu_workers)
        
        # Model sets for different layer counts
        self.model_sets = {
            1: [
                "b_mc_point_neutron_conv_standard_L1_comp",
                "b_mc_point_neutron_conv_standard_L1_InputQDq",
                "b_mc_point_xray_conv_standard_L2" # this xray model is designed for 2 layers. however, there are no xray models for 1 layer.
            ],
            2: [
                "b_mc_point_neutron_conv_standard_L2_comp",
                "b_mc_point_neutron_conv_standard_L2_InputQDq", 
                "b_mc_point_xray_conv_standard_L2"
            ]
        }
        
        # Get models for current layer count
        if self.layer_count not in self.model_sets:
            raise ValueError(f"Layer count {self.layer_count} not supported. Available: {list(self.model_sets.keys())}")
            
        self.models = self.model_sets[self.layer_count]
        
        # Pre-compiled experiment data cache
        self.experiment_cache = {}
        self.model_cache = {}  # Cache loaded models to avoid reloading
        
        # Performance tracking
        self.model_timing_stats = {}
        self.cache_hit_stats = defaultdict(int)
        self.memory_usage_stats = []
        self.batch_results = {}

    def setup_logging(self):
        """Setup logging to file instead of stdout."""
        # Create logs directory if it doesn't exist
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"BatchInference_{self.timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler
        log_file = log_dir / f"batch_inference_{self.timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler (minimal output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Also print important info to console
        print(f"Logging to: {log_file}")

    def create_output_directories(self):
        """Create organized output directory structure."""
        # Create subdirectories for different types of outputs
        self.inference_results_dir = self.output_dir / "inference_results"
        self.debug_dir = self.output_dir / "debug"
        self.plots_dir = self.output_dir / "plots"
        
        # Create directories
        self.inference_results_dir.mkdir(exist_ok=True)
        self.debug_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Created output directories:")
        self.logger.info(f"  Inference results: {self.inference_results_dir}")
        self.logger.info(f"  Debug files: {self.debug_dir}")
        self.logger.info(f"  Plots: {self.plots_dir}")

    @lru_cache(maxsize=1000)
    def load_experiment_data_cached(self, exp_id):
        """Cache experiment data loading to avoid repeated file I/O."""
        try:
            layer_dir = self.maria_dataset_path / str(self.layer_count)
            exp_curve_file = layer_dir / f"{exp_id}_experimental_curve.dat"
            exp_model_file = layer_dir / f"{exp_id}_model.txt"
            
            if not exp_curve_file.exists() or not exp_model_file.exists():
                return None
            
            # Use memory mapping for large files
            with open(exp_curve_file, 'r') as f:
                curve_data = f.read()
            
            with open(exp_model_file, 'r') as f:
                model_data = f.read()
            
            self.cache_hit_stats['experiment_data'] += 1
            return {
                'curve_data': curve_data,
                'model_data': model_data,
                'exp_id': exp_id
            }
            
        except Exception as e:
            print(f"Error loading experiment {exp_id}: {e}")
            return None

    def discover_experiments_optimized(self):
        """Optimized experiment discovery with parallel file checking."""
        layer_dir = self.maria_dataset_path / str(self.layer_count)
        
        if not layer_dir.exists():
            self.logger.warning(f"MARIA dataset directory not found: {layer_dir}")
            return []
        
        self.logger.info(f"Optimized search for {self.layer_count}-layer experiments in: {layer_dir}")
        
        try:
            # Use find with multiple conditions in one call
            result = subprocess.run([
                'find', str(layer_dir), 
                '-name', '*_experimental_curve.dat', 
                '-type', 'f',
                '-exec', 'dirname', '{}', ';'
            ], capture_output=True, text=True, check=True)
            
            experiment_dirs = set(result.stdout.strip().split('\n'))
            experiments = []
            
            # Parallel validation of experiment completeness
            if self.enable_parallel:
                with ProcessPoolExecutor(max_workers=min(4, self.max_workers)) as executor:
                    future_to_dir = {executor.submit(validate_experiment, exp_dir): exp_dir 
                                   for exp_dir in experiment_dirs if exp_dir}
                    
                    for future in as_completed(future_to_dir):
                        valid_exps = future.result()
                        experiments.extend(valid_exps)
            else:
                for exp_dir in experiment_dirs:
                    if exp_dir:
                        experiments.extend(validate_experiment(exp_dir))
            
            # Remove duplicates and filter
            experiments = list(set([exp for exp in experiments if exp.startswith('s')]))
            print(f"Found {len(experiments)} valid experiments")
            
            # Sample if we have more than requested
            if len(experiments) > self.num_experiments:
                experiments = random.sample(experiments, self.num_experiments)
                print(f"Randomly selected {len(experiments)} experiments for processing")
            
            return experiments
            
        except subprocess.CalledProcessError as e:
            print(f"Error finding experiments: {e}")
            return []

    def preload_experiment_batch(self, experiment_batch):
        """Preload a batch of experiments to minimize I/O during processing."""
        print(f"Preloading batch of {len(experiment_batch)} experiments...")
        
        preloaded_data = {}
        
        if self.enable_parallel:
            # Parallel preloading
            with ProcessPoolExecutor(max_workers=min(4, len(experiment_batch))) as executor:
                future_to_exp = {
                    executor.submit(self.load_experiment_data_cached, exp_id): exp_id 
                    for exp_id in experiment_batch
                }
                
                for future in as_completed(future_to_exp):
                    exp_id = future_to_exp[future]
                    data = future.result()
                    if data:
                        preloaded_data[exp_id] = data
        else:
            # Sequential preloading
            for exp_id in experiment_batch:
                data = self.load_experiment_data_cached(exp_id)
                if data:
                    preloaded_data[exp_id] = data
        
        print(f"Successfully preloaded {len(preloaded_data)}/{len(experiment_batch)} experiments")
        return preloaded_data

    def run_experiment_inference(self, exp_id, priors_type="broad"):
        """Run inference for a single experiment using the new parameterized interface."""
        try:
            result = InferencePipeline.run_experiment_inference(
                experiment_id=exp_id,
                models_list=self.models,
                data_directory=str(self.data_directory),
                priors_type=priors_type,
                output_dir=str(self.output_dir),
                layer_count=self.layer_count
            )
            
            if result['success']:
                self.logger.info(f"✓ {exp_id} completed successfully")
                return result
            else:
                self.logger.error(f"✗ {exp_id} failed: {result['error']}")
                return result
            
        except Exception as e:
            self.logger.error(f"✗ {exp_id} error: {e}")
            return {
                'exp_id': exp_id,
                'success': False,
                'error': str(e),
                'models_results': {}
            }

    def run_optimized_experiment_inference(self, exp_id, models, data_directory, 
                                         priors_type, output_dir, layer_count, preloaded_data=None):
        """Optimized inference using cached data and models."""
        try:
            # Create inference pipeline instance
            pipeline = InferencePipeline(
                experiment_id=exp_id,
                models_list=models,
                data_directory=data_directory,
                priors_type=priors_type,
                output_dir=output_dir,
                layer_count=layer_count
            )
            
            # Load experimental data and true parameters
            pipeline.discover_experiment_files()
            pipeline.load_experimental_data_from_files()
            pipeline.load_true_parameters_from_files()
            pipeline.generate_model_configurations()
            
            # Run inference for all models (don't show plots to save time)
            pipeline.run_all_models(show_plots=False)
            
            # Extract results with SLD profiles included BEFORE they get serialized/stripped
            models_results = {}
            best_model_name = None
            best_mape = float('inf')
            
            # Access the results from the pipeline's internal results storage
            # IMPORTANT: Extract data here BEFORE save_results() strips the numpy arrays
            for model_name, model_result in pipeline.results.items():
                if model_result['success']:
                    # Extract essential data including SLD profiles and reflectivity curves
                    models_results[model_name] = {
                        'success': True,
                        'parameter_metrics': model_result['parameter_metrics'],
                        'fit_metrics': model_result['fit_metrics'],
                        # Include SLD profiles for plotting - properly convert numpy arrays to lists
                        'sld_profile_x': model_result.get('sld_profile_x', []).tolist() if isinstance(model_result.get('sld_profile_x'), np.ndarray) else list(model_result.get('sld_profile_x', [])),
                        'sld_profile_predicted': model_result.get('sld_profile_predicted', []).tolist() if isinstance(model_result.get('sld_profile_predicted'), np.ndarray) else list(model_result.get('sld_profile_predicted', [])),
                        'sld_profile_polished': model_result.get('sld_profile_polished', []).tolist() if isinstance(model_result.get('sld_profile_polished'), np.ndarray) else list(model_result.get('sld_profile_polished', [])),
                        # Include reflectivity curve data - properly convert numpy arrays to lists
                        'q_model': model_result.get('q_model', []).tolist() if isinstance(model_result.get('q_model'), np.ndarray) else list(model_result.get('q_model', [])),
                        'predicted_curve': model_result.get('predicted_curve', []).tolist() if isinstance(model_result.get('predicted_curve'), np.ndarray) else list(model_result.get('predicted_curve', [])),
                        'polished_curve': model_result.get('polished_curve', []).tolist() if isinstance(model_result.get('polished_curve'), np.ndarray) else list(model_result.get('polished_curve', [])),
                        # Include timing if available
                        'inference_time': model_result.get('inference_time', 0)
                    }
                    
                    # Track best model (lowest overall MAPE)
                    if model_result['parameter_metrics'] and 'overall' in model_result['parameter_metrics']:
                        overall_mape = model_result['parameter_metrics']['overall']['mape']
                        if overall_mape < best_mape:
                            best_mape = overall_mape
                            best_model_name = model_name
                else:
                    models_results[model_name] = {'success': False}
            
            return {
                'success': len([r for r in models_results.values() if r.get('success', False)]) > 0,
                'models_results': models_results,
                'best_model_name': best_model_name,
                'best_mape': best_mape
            }
            
        except Exception as e:
            print(f"    ✗ Error in {exp_id} ({priors_type}): {e}")
            return {
                'success': False,
                'error': str(e),
                'models_results': {}
            }

    def process_experiment_batch_worker(self, args):
        """Optimized worker function that processes multiple experiments in one call."""
        (experiment_batch, models, data_directory, layer_count, output_dir, preloaded_data) = args
        
        batch_results = {}
        
        for exp_id in experiment_batch:
            try:
                # Check memory usage
                current_memory_gb = psutil.Process().memory_info().rss / (1024**3)
                if current_memory_gb > self.memory_limit_gb:
                    print(f"Warning: Memory usage ({current_memory_gb:.1f}GB) exceeding limit ({self.memory_limit_gb}GB)")
                
                exp_results = {
                    'experiment_id': exp_id,
                    'layer_count': layer_count,
                    'priors': {},
                    'model_times': {}
                }
                
                # Use preloaded data if available
                if exp_id in preloaded_data:
                    self.cache_hit_stats['preloaded_data'] += 1
                
                # Process both priors types
                for priors_type in ['broad', 'narrow']:
                    start_time = time.time()
                    
                    # Use optimized inference call
                    result = self.run_optimized_experiment_inference(
                        exp_id, models, data_directory, priors_type, 
                        output_dir, layer_count, preloaded_data.get(exp_id)
                    )
                    
                    end_time = time.time()
                    result['processing_time'] = end_time - start_time
                    exp_results['priors'][priors_type] = result
                    
                    # Extract timing data
                    if result['success'] and 'models_results' in result:
                        for model_name, model_result in result['models_results'].items():
                            if 'inference_time' in model_result:
                                if model_name not in exp_results['model_times']:
                                    exp_results['model_times'][model_name] = {}
                                if priors_type not in exp_results['model_times'][model_name]:
                                    exp_results['model_times'][model_name][priors_type] = []
                                exp_results['model_times'][model_name][priors_type].append(
                                    model_result['inference_time']
                                )
                
                batch_results[exp_id] = exp_results
                
            except Exception as e:
                print(f"Batch worker error for {exp_id}: {e}")
                batch_results[exp_id] = {
                    'experiment_id': exp_id,
                    'layer_count': layer_count,
                    'priors': {},
                    'error': str(e),
                    'success': False
                }
        
        return batch_results

    def run_batched_processing(self, experiments):
        """Process experiments in batches to optimize memory usage and I/O."""
        print(f"Running {len(experiments)} experiments in batches of {self.batch_size}...")
        
        all_results = {}
        total_batches = (len(experiments) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(experiments), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            experiment_batch = experiments[batch_idx:batch_idx + self.batch_size]
            
            print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(experiment_batch)} experiments...")
            
            # Preload this batch
            preloaded_data = self.preload_experiment_batch(experiment_batch)
            
            # Track memory usage
            memory_before = psutil.Process().memory_info().rss / (1024**3)
            
            if self.enable_parallel:
                # Process batch in parallel
                batch_results = self.process_batch_parallel(experiment_batch, preloaded_data)
            else:
                # Process batch sequentially
                batch_results = self.process_batch_sequential(experiment_batch, preloaded_data)
            
            # Track memory usage after batch
            memory_after = psutil.Process().memory_info().rss / (1024**3)
            self.memory_usage_stats.append({
                'batch': batch_num,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before
            })
            
            # Merge batch results
            all_results.update(batch_results)
            
            # Save intermediate results for crash recovery
            if batch_num % 3 == 0:  # Save every 3 batches
                self.save_intermediate_results(all_results, batch_num)
            
            # Optional: Clear caches if memory usage is too high
            if memory_after > self.memory_limit_gb * 0.8:
                print(f"High memory usage ({memory_after:.1f}GB), clearing caches...")
                self.load_experiment_data_cached.cache_clear()
                
            print(f"[Batch {batch_num}/{total_batches}] Completed. Memory: {memory_after:.1f}GB")
        
        return all_results

    def process_batch_parallel(self, experiment_batch, preloaded_data):
        """Process a batch of experiments in parallel."""
        batch_results = {}
        
        # Create args for individual experiments
        worker_args = []
        for exp_id in experiment_batch:
            worker_args.append((
                exp_id,
                self.models,
                self.data_directory,
                self.layer_count,
                self.output_dir
            ))
        
        with ProcessPoolExecutor(max_workers=min(self.max_workers, len(experiment_batch))) as executor:
            future_to_exp = {
                executor.submit(process_experiment_worker, args): args[0]
                for args in worker_args
            }
            
            for future in as_completed(future_to_exp):
                try:
                    exp_id = future_to_exp[future]
                    result = future.result()
                    batch_results[exp_id] = result
                    print(f"    ✓ {exp_id} completed")
                except Exception as e:
                    exp_id = future_to_exp[future]
                    print(f"    ✗ {exp_id} failed: {e}")
                    batch_results[exp_id] = {
                        'experiment_id': exp_id,
                        'layer_count': self.layer_count,
                        'priors': {},
                        'error': str(e),
                        'success': False
                    }
        
        return batch_results

    def process_batch_sequential(self, experiment_batch, preloaded_data):
        """Process a batch of experiments sequentially."""
        batch_results = {}
        
        for exp_id in experiment_batch:
            exp_results = {
                'experiment_id': exp_id,
                'layer_count': self.layer_count,
                'priors': {}
            }
            
            for priors_type in ['broad', 'narrow']:
                start_time = time.time()
                result = self.run_optimized_experiment_inference(
                    exp_id, self.models, self.data_directory, priors_type,
                    self.output_dir, self.layer_count, preloaded_data.get(exp_id)
                )
                end_time = time.time()
                result['processing_time'] = end_time - start_time
                exp_results['priors'][priors_type] = result
            
            batch_results[exp_id] = exp_results
        
        return batch_results

    def save_intermediate_results(self, results, batch_num):
        """Save intermediate results for crash recovery."""
        intermediate_file = self.output_dir / f"intermediate_results_batch_{batch_num}_{self.timestamp}.pkl"
        
        try:
            with open(intermediate_file, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"Intermediate results saved to: {intermediate_file}")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate results: {e}")

    def print_optimization_stats(self):
        """Print optimization performance statistics."""
        print(f"\nOPTIMIZATION PERFORMANCE STATISTICS:")
        print("=" * 50)
        
        # Cache hit statistics
        if self.cache_hit_stats:
            print("Cache Hit Statistics:")
            for cache_type, hits in self.cache_hit_stats.items():
                print(f"  {cache_type}: {hits} hits")
        
        # Memory usage statistics
        if self.memory_usage_stats:
            print("\nMemory Usage by Batch:")
            for stat in self.memory_usage_stats:
                print(f"  Batch {stat['batch']}: {stat['memory_before']:.1f}GB → {stat['memory_after']:.1f}GB "
                      f"(Δ{stat['memory_delta']:+.1f}GB)")
        
        # Estimated performance improvement
        cache_efficiency = sum(self.cache_hit_stats.values()) / max(1, len(self.cache_hit_stats))
        print(f"\nEstimated cache efficiency: {cache_efficiency:.1f} hits per cache type")
    
    def run_experiments(self):
        """Run the batch inference experiments and return results."""
        self.logger.info("Starting Optimized Batch Inference Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Target experiments: {self.num_experiments}")
        self.logger.info(f"Layer count: {self.layer_count}")
        self.logger.info(f"Models: {self.models}")
        self.logger.info(f"Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}")
        self.logger.info(f"Caching: {'Enabled' if self.enable_caching else 'Disabled'}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Max workers: {self.max_workers}")
        self.logger.info(f"Memory limit: {self.memory_limit_gb}GB")
        
        # System info
        available_memory = psutil.virtual_memory().available / (1024**3)
        self.logger.info(f"Available memory: {available_memory:.1f}GB")
        self.logger.info(f"CPU cores: {mp.cpu_count()}")
        
        print(f"Starting batch inference for {self.num_experiments} experiments...")
        print(f"Check logs at: {self.output_dir / 'logs'}")
        
        # Discover experiments
        experiments = self.discover_experiments_optimized()
        
        if not experiments:
            self.logger.error("No experiments found. Exiting.")
            return None
        
        self.logger.info(f"Processing {len(experiments)} experiments...")
        
        start_time = time.time()
        
        # Run batched processing
        all_results = self.run_batched_processing(experiments)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.logger.info(f"Optimized processing completed!")
        self.logger.info(f"Total processing time: {total_time:.1f} seconds")
        self.logger.info(f"Average time per experiment: {total_time/len(experiments):.1f} seconds")
        
        # Print optimization statistics
        self.print_optimization_stats()
        
        # Create batch summary and plots (reuse existing methods)
        self.create_batch_summary(all_results)
        self.create_performance_plots(all_results)
        
        # NEW: Analyze and plot individual best/worst predictions
        self.analyze_and_plot_outliers(all_results)
        
        return all_results
        print(f"Average time per experiment: {total_time/len(experiments):.1f} seconds")
        
        # Create batch summary
        self.create_batch_summary(all_results)
        
        # Create performance plots
        self.create_performance_plots(all_results)

    def run_parallel_processing(self, experiments):
        """Run experiments in parallel using multiprocessing."""
        print(f"Running {len(experiments)} experiments in parallel with {self.max_workers} workers...")
        
        # Prepare arguments for worker processes
        worker_args = []
        for exp_id in experiments:
            worker_args.append((
                exp_id, 
                self.models, 
                self.data_directory, 
                self.layer_count, 
                self.output_dir
            ))
        
        all_results = {}
        completed_count = 0
        
        # Use ProcessPoolExecutor for better resource management
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_exp = {
                executor.submit(process_experiment_worker, args): args[0] 
                for args in worker_args
            }
            
            # Process completed jobs as they finish
            for future in as_completed(future_to_exp):
                exp_id = future_to_exp[future]
                try:
                    result = future.result()
                    all_results[exp_id] = result
                    completed_count += 1
                    
                    print(f"[{completed_count}/{len(experiments)}] Completed {exp_id}")
                    
                    # Save individual experiment results
                    exp_file = self.output_dir / f"{exp_id}_results.json"
                    with open(exp_file, 'w') as f:
                        json.dump(result, f, indent=2)
                        
                except Exception as e:
                    self.logger.error(f"[{completed_count+1}/{len(experiments)}] Failed {exp_id}: {e}")
                    all_results[exp_id] = {
                        'experiment_id': exp_id,
                        'layer_count': self.layer_count,
                        'priors': {},
                        'error': str(e),
                        'success': False
                    }
                    completed_count += 1
        
        return all_results

    def create_batch_summary(self, all_results):
        """Create summary of batch results."""
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"BATCH INFERENCE SUMMARY - {self.layer_count}-LAYER EXPERIMENTS")
        self.logger.info(f"{'='*80}")
        
        print(f"\n{'='*80}")
        print(f"BATCH INFERENCE SUMMARY - {self.layer_count}-LAYER EXPERIMENTS")
        print(f"{'='*80}")
        
        # Calculate statistics
        total_experiments = len(all_results)
        successful_broad = 0
        successful_narrow = 0
        
        # Aggregate performance by model and priors type
        model_performance = defaultdict(lambda: defaultdict(list))
        
        for exp_id, exp_result in all_results.items():
            # Collect model timing data
            if 'model_times' in exp_result:
                for model_name, priors_times in exp_result['model_times'].items():
                    if model_name not in self.model_timing_stats:
                        self.model_timing_stats[model_name] = {}
                    for priors_type, times_list in priors_times.items():
                        if priors_type not in self.model_timing_stats[model_name]:
                            self.model_timing_stats[model_name][priors_type] = []
                        self.model_timing_stats[model_name][priors_type].extend(times_list)
            
            for priors_type in ['broad', 'narrow']:
                if priors_type not in exp_result.get('priors', {}):
                    continue  # Skip if this priors type is missing
                    
                priors_result = exp_result['priors'][priors_type]
                if priors_result.get('success', False):
                    if priors_type == 'broad':
                        successful_broad += 1
                    else:
                        successful_narrow += 1
                    
                    # Collect metrics for each model
                    for model_name, model_result in priors_result.get('models_results', {}).items():
                        if model_result.get('success', False):
                            # Fit metrics
                            if 'fit_metrics' in model_result:
                                fit_metrics = model_result['fit_metrics']
                                model_performance[model_name][f'{priors_type}_r2'].append(fit_metrics.get('r_squared', 0))
                                model_performance[model_name][f'{priors_type}_mse'].append(fit_metrics.get('mse', float('inf')))
                            
                            # Parameter metrics
                            if 'parameter_metrics' in model_result and model_result['parameter_metrics']:
                                param_metrics = model_result['parameter_metrics']
                                overall_mape = param_metrics['overall']['mape']
                                model_performance[model_name][f'{priors_type}_param_mape'].append(overall_mape)
                            
                            # Timing metrics
                            if 'inference_time' in model_result:
                                if model_name not in self.model_timing_stats:
                                    self.model_timing_stats[model_name] = {}
                                if priors_type not in self.model_timing_stats[model_name]:
                                    self.model_timing_stats[model_name][priors_type] = []
                                self.model_timing_stats[model_name][priors_type].append(model_result['inference_time'])
        
        print(f"Total experiments: {total_experiments}")
        print(f"Successful with broad priors: {successful_broad}")
        print(f"Successful with narrow priors: {successful_narrow}")
        print(f"Success rate (broad): {100*successful_broad/total_experiments:.1f}%")
        print(f"Success rate (narrow): {100*successful_narrow/total_experiments:.1f}%")
        
        # Print model performance summary
        print(f"\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 80)
        
        for model_name in self.models:
            if model_name in model_performance:
                perf = model_performance[model_name]
                print(f"\n{model_name}:")
                
                for priors_type in ['broad', 'narrow']:
                    r2_key = f'{priors_type}_r2'
                    mse_key = f'{priors_type}_mse'
                    mape_key = f'{priors_type}_param_mape'
                    
                    if r2_key in perf and perf[r2_key]:
                        print(f"  {priors_type.title()} priors:")
                        print(f"    R² - Mean: {np.mean(perf[r2_key]):.3f}, Std: {np.std(perf[r2_key]):.3f}")
                        
                        mse_vals = [x for x in perf[mse_key] if x != float('inf')]
                        if mse_vals:
                            print(f"    MSE - Mean: {np.mean(mse_vals):.6f}, Std: {np.std(mse_vals):.6f}")
                        
                        if mape_key in perf and perf[mape_key]:
                            print(f"    Param MAPE - Mean: {np.mean(perf[mape_key]):.1f}%, Std: {np.std(perf[mape_key]):.1f}%")
        
        # Print model timing summary
        self.print_timing_summary()
        
        # Save batch summary
        summary = {
            'timestamp': self.timestamp,
            'layer_count': self.layer_count,
            'total_experiments': total_experiments,
            'successful_broad': successful_broad,
            'successful_narrow': successful_narrow,
            'models_tested': self.models,
            'model_performance': dict(model_performance),
            'model_timing_stats': dict(self.model_timing_stats),
            'all_results': all_results
        }
        
        summary_file = self.output_dir / f"batch_summary_{self.layer_count}layer_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBatch summary saved to: {summary_file}")
        return summary

    def create_performance_plots(self, all_results):
        """Create comprehensive 2-column performance visualization plots."""
        
        # Collect data for plotting
        model_mapes = defaultdict(lambda: defaultdict(list))
        model_param_mapes = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> priors -> param_type -> values
        experiment_mapes = defaultdict(lambda: defaultdict(list))  # priors_type -> exp_id -> mape
        
        for exp_id, exp_result in all_results.items():
            for priors_type in ['broad', 'narrow']:
                priors_result = exp_result['priors'][priors_type]
                if priors_result['success']:
                    exp_mapes_for_priors = []
                    
                    for model_name, model_result in priors_result['models_results'].items():
                        if model_result['success'] and 'parameter_metrics' in model_result:
                            param_metrics = model_result['parameter_metrics']
                            if param_metrics and 'overall' in param_metrics:
                                # Overall MAPE
                                overall_mape = param_metrics['overall']['mape']
                                model_mapes[model_name][priors_type].append(overall_mape)
                                exp_mapes_for_priors.append(overall_mape)
                                
                                # Parameter-specific MAPEs
                                by_type = param_metrics.get('by_type', {})
                                for param_type in ['thickness', 'roughness', 'sld']:
                                    mape_key = f'{param_type}_mape'
                                    if mape_key in by_type:
                                        model_param_mapes[model_name][priors_type][param_type].append(by_type[mape_key])
                    
                    # Calculate average MAPE for this experiment with this priors type
                    if exp_mapes_for_priors:
                        experiment_mapes[priors_type][exp_id] = np.mean(exp_mapes_for_priors)
        
        # Create 2-column, 3-row layout
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f'Batch Inference Performance Analysis - {len(all_results)} {self.layer_count}-Layer Experiments', fontsize=16, fontweight='bold')
        
        # Column titles with narrow priors info
        narrow_priors_percent = int(NARROW_PRIORS_DEVIATION * 100)
        axes[0, 0].text(0.5, 1.15, 'BROAD PRIORS', transform=axes[0, 0].transAxes, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[0, 1].text(0.5, 1.15, f'NARROW PRIORS (±{narrow_priors_percent}%)', transform=axes[0, 1].transAxes, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        models = list(model_mapes.keys())
        model_labels = [model.replace('b_mc_point_', '').replace('_conv_standard', '') for model in models]
        
        # Row 1: Average MAPE by Model
        for col, priors_type in enumerate(['broad', 'narrow']):
            ax = axes[0, col]
            
            means = []
            stds = []
            for model in models:
                vals = model_mapes[model][priors_type]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            
            x = np.arange(len(models))
            color = 'skyblue' if priors_type == 'broad' else 'lightcoral'
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=color, 
                         edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Average MAPE (%)')
            ax.set_title(f'Overall MAPE by Model ({priors_type.title()} Priors)')
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                if mean > 0:
                    ax.text(i, mean + std + max(means) * 0.02, f'{mean:.1f}%', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Row 2: Parameter-specific MAPE by Model
        param_types = ['thickness', 'roughness', 'sld']
        param_colors = {'thickness': '#FF6B6B', 'roughness': '#4ECDC4', 'sld': '#45B7D1'}
        
        for col, priors_type in enumerate(['broad', 'narrow']):
            ax = axes[1, col]
            
            x = np.arange(len(models))
            width = 0.25
            
            for i, param_type in enumerate(param_types):
                medians = []
                for model in models:
                    vals = model_param_mapes[model][priors_type][param_type]
                    medians.append(np.median(vals) if vals else 0)
                
                ax.bar(x + i*width - width, medians, width, 
                      label=param_type.title(), alpha=0.8, color=param_colors[param_type],
                      edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Median MAPE (%)')
            ax.set_title(f'Parameter-Specific MAPE ({priors_type.title()} Priors)')
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Row 3: Edge Case Detection
        for col, priors_type in enumerate(['broad', 'narrow']):
            ax = axes[2, col]
            
            exp_data = experiment_mapes[priors_type]
            if exp_data:
                exp_ids = list(exp_data.keys())
                exp_vals = list(exp_data.values())
                exp_indices = range(len(exp_ids))
                
                # Plot all experiments
                ax.plot(exp_indices, exp_vals, 'o-', alpha=0.7, linewidth=1, markersize=4,
                       color='darkblue', label='Experiments')
                
                # Calculate threshold for edge cases
                mean_mape = np.mean(exp_vals)
                std_mape = np.std(exp_vals)
                threshold = mean_mape + 2 * std_mape
                
                # Highlight edge cases
                edge_cases = [(i, exp_id, mape) for i, (exp_id, mape) in enumerate(zip(exp_ids, exp_vals)) if mape > threshold]
                
                if edge_cases:
                    edge_indices = [i for i, _, _ in edge_cases]
                    edge_mapes = [mape for _, _, mape in edge_cases]
                    ax.scatter(edge_indices, edge_mapes, color='red', s=80, alpha=0.8, 
                             label=f'Edge Cases (>{threshold:.1f}%)', zorder=5)
                    
                    # Annotate worst edge cases
                    worst_cases = sorted(edge_cases, key=lambda x: x[2], reverse=True)[:3]
                    for i, exp_id, mape in worst_cases:
                        ax.annotate(f'{exp_id}\n{mape:.1f}%', 
                                  xy=(i, mape), xytext=(10, 10), 
                                  textcoords='offset points', fontsize=8,
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                # Add threshold line
                ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                          label=f'Threshold (μ+2σ)')
                
                ax.set_xlabel('Experiment Index')
                ax.set_ylabel('Average MAPE (%)')
                ax.set_title(f'Edge Case Detection ({priors_type.title()} Priors)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Print edge cases for this priors type
                if edge_cases:
                    print(f"\nEdge Cases for {priors_type.title()} Priors (MAPE > {threshold:.1f}%):")
                    for i, exp_id, mape in sorted(edge_cases, key=lambda x: x[2], reverse=True):
                        print(f"  {exp_id}: {mape:.1f}% MAPE")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for main title
        
        # Save plot
        plot_file = self.output_dir / f"batch_inference_results_{self.layer_count}layer_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to: {plot_file}")
    
    def print_timing_summary(self):
        """Print detailed timing summary for each model."""
        print(f"\nMODEL TIMING PERFORMANCE SUMMARY:")
        print("=" * 80)
        
        if not self.model_timing_stats:
            print("No timing data available.")
            return
        
        # Calculate overall statistics
        all_times = []
        for model_stats in self.model_timing_stats.values():
            for times in model_stats.values():
                all_times.extend(times)
        
        if all_times:
            print(f"Overall Statistics (all models, all priors):")
            print(f"  Total inference runs: {len(all_times)}")
            print(f"  Mean inference time: {np.mean(all_times):.3f}s")
            print(f"  Std inference time: {np.std(all_times):.3f}s")
            print(f"  Min inference time: {np.min(all_times):.3f}s")
            print(f"  Max inference time: {np.max(all_times):.3f}s")
            print()
        
        # Print per-model statistics
        for model_name in self.models:
            if model_name in self.model_timing_stats:
                model_stats = self.model_timing_stats[model_name]
                print(f"{model_name}:")
                
                for priors_type in ['broad', 'narrow']:
                    if priors_type in model_stats and model_stats[priors_type]:
                        times = model_stats[priors_type]
                        print(f"  {priors_type.title()} priors ({len(times)} runs):")
                        print(f"    Mean time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
                        print(f"    Range: {np.min(times):.3f}s - {np.max(times):.3f}s")
                        
                        # Calculate percentiles
                        p25, p50, p75 = np.percentile(times, [25, 50, 75])
                        print(f"    Percentiles: 25%={p25:.3f}s, 50%={p50:.3f}s, 75%={p75:.3f}s")
                
                # Combined statistics for this model
                all_model_times = []
                for times in model_stats.values():
                    all_model_times.extend(times)
                
                if all_model_times:
                    print(f"  Overall model average: {np.mean(all_model_times):.3f}s ± {np.std(all_model_times):.3f}s")
                print()
        
        # Model comparison
        print("MODEL TIMING COMPARISON (Mean ± Std):")
        print("-" * 60)
        model_avg_times = {}
        for model_name in self.models:
            if model_name in self.model_timing_stats:
                all_model_times = []
                for times in self.model_timing_stats[model_name].values():
                    all_model_times.extend(times)
                if all_model_times:
                    model_avg_times[model_name] = np.mean(all_model_times)
                    model_label = model_name.replace('b_mc_point_', '').replace('_conv_standard', '')
                    print(f"{model_label}: {np.mean(all_model_times):.3f}s ± {np.std(all_model_times):.3f}s")
        
        # Find fastest and slowest models
        if model_avg_times:
            fastest_model = min(model_avg_times, key=model_avg_times.get)
            slowest_model = max(model_avg_times, key=model_avg_times.get)
            fastest_label = fastest_model.replace('b_mc_point_', '').replace('_conv_standard', '')
            slowest_label = slowest_model.replace('b_mc_point_', '').replace('_conv_standard', '')
            
            print(f"\nFastest model: {fastest_label} ({model_avg_times[fastest_model]:.3f}s)")
            print(f"Slowest model: {slowest_label} ({model_avg_times[slowest_model]:.3f}s)")
            speedup = model_avg_times[slowest_model] / model_avg_times[fastest_model]
            print(f"Speed difference: {speedup:.2f}x")

    def generate_sld_profile_from_params(self, params, param_names, thickness_range=None):
        """
        Generate SLD profile from predicted parameters for 1-layer system.
        
        Args:
            params: List of parameter values
            param_names: List of parameter names
            thickness_range: Optional tuple (min_depth, max_depth) for depth axis
            
        Returns:
            tuple: (depths, sld_profile) numpy arrays
        """
        try:
            # Extract parameters by name
            param_dict = dict(zip(param_names, params))
            
            # Find thickness parameter
            thickness = None
            for name in param_dict:
                if 'thickness' in name.lower() and 'l1' in name.lower():
                    thickness = param_dict[name]
                    break
            
            if thickness is None:
                print(f"Warning: Could not find thickness parameter in {param_names}")
                return None, None
            
            # Find SLD parameters
            layer_sld = None
            substrate_sld = None
            
            for name in param_dict:
                if 'sld' in name.lower() and 'l1' in name.lower():
                    layer_sld = param_dict[name]
                elif 'sld' in name.lower() and ('sub' in name.lower() or 'backing' in name.lower()):
                    substrate_sld = param_dict[name]
            
            if layer_sld is None or substrate_sld is None:
                print(f"Warning: Could not find SLD parameters in {param_names}")
                return None, None
            
            # Set depth range
            if thickness_range is None:
                # Extend beyond layer boundaries for visualization
                padding = max(thickness * 0.5, 50)  # At least 50 Å padding
                min_depth = -padding
                max_depth = thickness + padding
            else:
                min_depth, max_depth = thickness_range
            
            # Create depth axis
            depths = np.linspace(min_depth, max_depth, 1024)
            
            # Create step function SLD profile
            # Ambient (< 0): set to 0 by default (can be modified if needed)
            # Layer (0 to thickness): layer_sld
            # Substrate (> thickness): substrate_sld
            sld_profile = np.where(
                depths < 0,
                0,  # Ambient SLD (could be a parameter if available)
                np.where(
                    depths <= thickness,
                    layer_sld,
                    substrate_sld
                )
            )
            
            return depths, sld_profile
            
        except Exception as e:
            print(f"Error generating SLD profile: {e}")
            return None, None

    def analyze_and_plot_outliers(self, all_results):
        """
        Analyze experiments by MAPE and create individual plots for best/worst predictions.
        Creates SLD profiles and reflectivity curves ensuring no overlap between best and worst categories.
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING BEST AND WORST PREDICTIONS")
        print(f"{'='*80}")
        
        # Extract MAPE data for sorting
        experiment_mapes = {'broad': {}, 'narrow': {}}
        experiment_data = {'broad': {}, 'narrow': {}}
        
        for exp_id, exp_result in all_results.items():
            for priors_type in ['broad', 'narrow']:
                if priors_type not in exp_result.get('priors', {}):
                    continue
                    
                priors_result = exp_result['priors'][priors_type]
                if not priors_result.get('success', False):
                    continue
                
                # Calculate average MAPE across all models for this experiment
                model_mapes = []
                all_models_data = {}
                
                for model_name, model_result in priors_result.get('models_results', {}).items():
                    if model_result.get('success', False) and 'parameter_metrics' in model_result:
                        param_metrics = model_result['parameter_metrics']
                        if 'overall' in param_metrics and 'mape' in param_metrics['overall']:
                            mape = param_metrics['overall']['mape']
                            model_mapes.append(mape)
                            
                            # Store all model data for plotting
                            all_models_data[model_name] = {
                                'mape': mape,
                                'model_result': model_result
                            }
                
                if model_mapes and all_models_data:
                    avg_mape = np.mean(model_mapes)
                    experiment_mapes[priors_type][exp_id] = avg_mape
                    experiment_data[priors_type][exp_id] = {
                        'mape': avg_mape,
                        'all_models_data': all_models_data,
                        'exp_result': exp_result
                    }
        
        # Process each priors type independently ensuring no overlap within each priors type
        for priors_type in ['broad', 'narrow']:
            if not experiment_mapes[priors_type]:
                print(f"No valid results found for {priors_type} priors")
                continue
                
            # Sort by MAPE (ascending - best first)
            sorted_experiments = sorted(
                experiment_mapes[priors_type].items(), 
                key=lambda x: x[1]
            )
            
            # For this priors type, track experiments used within this priors type only
            priors_used_experiments = set()
            
            # Select best experiments (lowest MAPE) 
            best_experiments = []
            for exp_id, mape in sorted_experiments:
                if exp_id not in priors_used_experiments and len(best_experiments) < 3:
                    best_experiments.append((exp_id, mape))
                    priors_used_experiments.add(exp_id)
            
            # Select worst experiments (highest MAPE) ensuring no overlap with best within this priors type
            worst_experiments = []
            for exp_id, mape in reversed(sorted_experiments):
                if exp_id not in priors_used_experiments and len(worst_experiments) < 3:
                    worst_experiments.append((exp_id, mape))
                    priors_used_experiments.add(exp_id)
            
            # Reverse worst_experiments to maintain descending order by MAPE
            worst_experiments = worst_experiments[::-1]
            
            print(f"\n{priors_type.upper()} PRIORS RANKING (no overlap within {priors_type} priors):")
            print(f"Best {len(best_experiments)} experiments (lowest MAPE):")
            for i, (exp_id, mape) in enumerate(best_experiments):
                print(f"  {i+1}. {exp_id}: {mape:.2f}% MAPE")
            
            print(f"Worst {len(worst_experiments)} experiments (highest MAPE):")
            for i, (exp_id, mape) in enumerate(worst_experiments):
                print(f"  {i+1}. {exp_id}: {mape:.2f}% MAPE")
            
            # Verify no overlap within this priors type
            best_exp_ids = [exp_id for exp_id, _ in best_experiments]
            worst_exp_ids = [exp_id for exp_id, _ in worst_experiments]
            overlap = set(best_exp_ids) & set(worst_exp_ids)
            if overlap:
                print(f"  WARNING: Found overlap within {priors_type} priors: {overlap}")
            else:
                print(f"  ✓ No overlap between best and worst in {priors_type} priors")
            
            # Create plots for best and worst experiments
            if best_experiments:
                self.plot_individual_experiments(
                    best_experiments, experiment_data[priors_type], 
                    priors_type, "best", "Best Predictions"
                )
            
            if worst_experiments:
                self.plot_individual_experiments(
                    worst_experiments, experiment_data[priors_type], 
                    priors_type, "worst", "Worst Predictions"
                )
    
    def plot_individual_experiments(self, experiments_list, experiment_data, 
                                  priors_type, category, title_prefix):
        """
        Create individual SLD profile plots.
        
        Args:
            experiments_list: List of (exp_id, mape) tuples
            experiment_data: Dictionary containing model results and data
            priors_type: 'broad' or 'narrow'
            category: 'best' or 'worst'
            title_prefix: Title prefix for plots
        """
        print(f"\nCreating plots for {title_prefix.lower()} {priors_type} priors experiments...")
        
        # Create subplots: 3 rows (experiments) x 1 column (SLD only)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle(f'{title_prefix} Predictions - {priors_type.title()} Priors', 
                     fontsize=16, fontweight='bold')
        
        # Define colors for different models - more distinct colors
        model_colors = {
            'b_mc_point_neutron_conv_standard_L1_comp': '#2E8B57',  # Sea Green
            'b_mc_point_neutron_conv_standard_L1_InputQDq': '#FF6347',  # Tomato  
            'b_mc_point_neutron_conv_standard_L2_comp': '#4169E1',  # Royal Blue
            'b_mc_point_neutron_conv_standard_L2_InputQDq': '#FF8C00',  # Dark Orange
            'b_mc_point_xray_conv_standard_L2': '#9932CC'  # Dark Orchid
        }
        
        for idx, (exp_id, mape) in enumerate(experiments_list):
            if exp_id not in experiment_data:
                continue
                
            exp_data = experiment_data[exp_id]
            all_models_data = exp_data['all_models_data']
            
            # Plot SLD Profile
            ax_sld = axes[idx]  # Single column, so just use the row index
            
            # Load ground truth SLD profile
            try:
                # Load true parameters to construct ground truth SLD
                exp_model_file = self.maria_dataset_path / f"{exp_id}_model.txt"
                if exp_model_file.exists():
                    with open(exp_model_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse the true parameters for SLD construction
                    true_thickness = None
                    true_layer_sld = None
                    true_sub_sld = None
                    
                    for line in lines:
                        if 'layer1' in line and 'thickness' not in line:
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                try:
                                    true_layer_sld = float(parts[1]) * 1e6  # Convert to 10^-6 units
                                    true_thickness = float(parts[2])
                                except (ValueError, IndexError):
                                    continue
                        elif 'backing' in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    true_sub_sld = float(parts[1]) * 1e6  # Convert to 10^-6 units
                                except (ValueError, IndexError):
                                    continue
                    
                    print(f"  DEBUG: Parsed true params - thickness: {true_thickness}, layer_sld: {true_layer_sld}, sub_sld: {true_sub_sld}")
                    
                    # Generate ground truth SLD profile
                    if true_thickness and true_layer_sld is not None and true_sub_sld is not None:
                        depths_true, sld_profile_true = self.generate_sld_profile_from_params(
                            [true_thickness, 0, 0, true_layer_sld, true_sub_sld],
                            ['Thickness L1', 'Roughness L1', 'Roughness sub', 'SLD L1', 'SLD sub']
                        )
                        
                        if depths_true is not None and sld_profile_true is not None:
                            print(f"  DEBUG: Ground truth SLD range: {min(sld_profile_true):.2f} to {max(sld_profile_true):.2f}")
                            ax_sld.plot(depths_true, sld_profile_true, 'k--', linewidth=4, 
                                       label='Ground Truth', alpha=0.9, zorder=10)
                
            except Exception as e:
                print(f"  Warning: Could not load ground truth SLD for {exp_id}: {e}")
            
            # Plot predicted SLD profiles from all models
            models_plotted = 0
            for model_name, model_data in all_models_data.items():
                model_result = model_data['model_result']
                model_mape = model_data['mape']
                
                print(f"  DEBUG: Processing model {model_name} for {exp_id}")
                
                # Try to get SLD data from stored results first (if available)
                has_stored_sld = ('sld_profile_x' in model_result and 'sld_profile_polished' in model_result and
                                 len(model_result['sld_profile_x']) > 0 and len(model_result['sld_profile_polished']) > 0)
                
                print(f"    Has stored SLD data: {has_stored_sld}")
                
                if has_stored_sld:
                    # Use stored SLD data
                    sld_x = np.array(model_result['sld_profile_x'])
                    sld_profile = np.array(model_result['sld_profile_polished'])
                    
                    print(f"    Using stored SLD X range: {min(sld_x):.2f} to {max(sld_x):.2f}")
                    print(f"    Using stored SLD profile range: {min(sld_profile):.6f} to {max(sld_profile):.6f}")
                    
                    # Convert to proper units if needed
                    if np.max(np.abs(sld_profile)) < 1e-4:
                        sld_profile = sld_profile * 1e6
                        print(f"    Converted SLD profile range: {min(sld_profile):.6f} to {max(sld_profile):.6f}")
                    
                    model_label = model_name.replace('b_mc_point_', '').replace('_conv_standard', '')
                    color = model_colors.get(model_name, '#333333')
                    
                    ax_sld.plot(sld_x, sld_profile, '-', linewidth=3, 
                               color=color, label=f'{model_label} ({model_mape:.1f}%)', alpha=0.85, zorder=5)
                    models_plotted += 1
                    
                elif 'polished_params' in model_result and 'param_names' in model_result:
                    # Generate SLD profile from predicted parameters
                    params = model_result['polished_params']
                    param_names = model_result['param_names']
                    
                    print(f"    Generating SLD from parameters: {params}")
                    print(f"    Parameter names: {param_names}")
                    
                    depths, sld_profile = self.generate_sld_profile_from_params(params, param_names)
                    
                    if depths is not None and sld_profile is not None:
                        print(f"    Generated SLD depth range: {min(depths):.2f} to {max(depths):.2f}")
                        print(f"    Generated SLD profile range: {min(sld_profile):.6f} to {max(sld_profile):.6f}")
                        
                        model_label = model_name.replace('b_mc_point_', '').replace('_conv_standard', '')
                        color = model_colors.get(model_name, '#333333')
                        
                        ax_sld.plot(depths, sld_profile, '-', linewidth=3, 
                                   color=color, label=f'{model_label} ({model_mape:.1f}%)', alpha=0.85, zorder=5)
                        models_plotted += 1
                    else:
                        print(f"    WARNING: Failed to generate SLD profile for {model_name}")
                else:
                    print(f"    WARNING: No SLD data or parameters available for {model_name}")
            
            print(f"  DEBUG: Plotted {models_plotted} SLD profiles for {exp_id}")
            
            ax_sld.set_xlabel('Depth (Å)', fontsize=11)
            ax_sld.set_ylabel('SLD (×10⁻⁶ Å⁻²)', fontsize=11)
            ax_sld.set_title(f'SLD Profile\n{exp_id} - Avg MAPE: {mape:.2f}%', fontsize=12, fontweight='bold')
            ax_sld.grid(True, alpha=0.3, linestyle=':')
            ax_sld.legend(fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Export plotting data to JSON for debugging
        debug_data = {
            'category': category,
            'priors_type': priors_type,
            'experiments': {}
        }
        
        for exp_id, mape in experiments_list:
            if exp_id not in experiment_data:
                continue
                
            exp_data = experiment_data[exp_id]
            exp_debug = {
                'mape': mape,
                'true_params': exp_data.get('true_params', {}),
                'models': {}
            }
            
            # Add ground truth data
            # Note: Ground truth data needs to be loaded separately since it's not stored in experiment results
            try:
                # Load true parameters for this experiment
                model_file_path = self.maria_dataset_path / f"{exp_id}_model.txt"
                if model_file_path.exists():
                    with open(model_file_path, 'r') as f:
                        content = f.read()
                    # Parse parameters from the model file content
                    exp_debug['ground_truth_params'] = content[:200]  # First 200 chars for debugging
                else:
                    exp_debug['ground_truth_params'] = "Model file not found"
                
                # Load experimental curve data for ground truth reflectivity
                exp_file_path = self.maria_dataset_path / f"{exp_id}_experimental_curve.dat"
                if exp_file_path.exists():
                    exp_data_raw = np.loadtxt(exp_file_path)
                    if exp_data_raw.shape[1] >= 3:
                        q_exp = exp_data_raw[:, 0]
                        r_exp = exp_data_raw[:, 1]
                        exp_debug['ground_truth_refl'] = {
                            'q': q_exp[:10].tolist() if len(q_exp) > 10 else q_exp.tolist(),
                            'r': r_exp[:10].tolist() if len(r_exp) > 10 else r_exp.tolist(),
                            'length': len(r_exp),
                            'range': [float(min(r_exp)), float(max(r_exp))]
                        }
                        
                        # Generate true SLD profile from model parameters for comparison
                        # This is a placeholder - would need actual SLD generation logic
                        exp_debug['ground_truth_sld'] = {
                            'note': 'SLD profile generation needed',
                            'q_range': [float(min(q_exp)), float(max(q_exp))]
                        }
                else:
                    exp_debug['ground_truth_refl'] = "Experimental curve file not found"
                    exp_debug['ground_truth_sld'] = "Cannot generate without experimental data"
                    
            except Exception as e:
                exp_debug['ground_truth_error'] = str(e)
            
            # Add model data
            all_models_data = exp_data['all_models_data']
            for model_name, model_data in all_models_data.items():
                model_result = model_data['model_result']
                model_debug = {
                    'mape': model_data['mape'],
                    'has_sld_x': 'sld_profile_x' in model_result,
                    'has_sld_profile': 'sld_profile_polished' in model_result,
                    'has_q_model': 'q_model' in model_result,
                    'has_polished_curve': 'polished_curve' in model_result
                }
                
                if 'sld_profile_x' in model_result and len(model_result['sld_profile_x']) > 0:
                    sld_x = model_result['sld_profile_x']
                    model_debug['sld_x_length'] = len(sld_x)
                    model_debug['sld_x_range'] = [float(min(sld_x)), float(max(sld_x))]
                
                if 'sld_profile_polished' in model_result and len(model_result['sld_profile_polished']) > 0:
                    sld_prof = model_result['sld_profile_polished']
                    model_debug['sld_profile_length'] = len(sld_prof)
                    model_debug['sld_profile_range'] = [float(min(sld_prof)), float(max(sld_prof))]
                
                if 'q_model' in model_result and len(model_result['q_model']) > 0:
                    q_vals = model_result['q_model']
                    model_debug['q_model_length'] = len(q_vals)
                    model_debug['q_model_range'] = [float(min(q_vals)), float(max(q_vals))]
                
                if 'polished_curve' in model_result and len(model_result['polished_curve']) > 0:
                    r_vals = model_result['polished_curve']
                    model_debug['polished_curve_length'] = len(r_vals)
                    model_debug['polished_curve_range'] = [float(min(r_vals)), float(max(r_vals))]
                
                exp_debug['models'][model_name] = model_debug
            
            debug_data['experiments'][exp_id] = exp_debug
        
        # Save debug data
        debug_filename = f"debug_{category}_{priors_type}_plotting_data.json"
        debug_path = self.output_dir / debug_filename
        with open(debug_path, 'w') as f:
            json.dump(debug_data, f, indent=2)
        print(f"  DEBUG: Exported plotting data to {debug_path}")
        
        # Save plot
        plot_filename = f"individual_{category}_{priors_type}_predictions.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_path}")
    
    # ...existing code...
def parse_arguments():
    """Enhanced argument parsing with optimization options."""
    parser = argparse.ArgumentParser(description="Run optimized batch inference on MARIA experiments")
    parser.add_argument('--num-experiments', type=int, default=25,
                       help='Number of experiments to process (default: 25)')
    parser.add_argument('--layer-count', type=int, choices=[1, 2], default=2,
                       help='Number of layers to process (1 or 2, default: 2)')
    parser.add_argument('--data-directory', type=str, default='data',
                       help='Data directory path (default: data)')
    
    # Parallel processing options
    parser.add_argument('--disable-parallel', action='store_true',
                       help='Disable parallel processing (run sequentially)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto-detect)')
    
    # Optimization options
    parser.add_argument('--disable-caching', action='store_true',
                       help='Disable caching optimizations')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of experiments to process per batch (default: 5)')
    parser.add_argument('--memory-limit-gb', type=float, default=48.0,
                       help='Memory limit in GB (default: 48.0)')
    
    return parser.parse_args()


def main():
    """Main function to run the optimized batch inference pipeline."""
    args = parse_arguments()
    
    print(f"Running OPTIMIZED batch processing for {args.layer_count}-layer experiments")
    
    # Run optimized batch inference pipeline
    batch_pipeline = BatchInferencePipeline(
        num_experiments=args.num_experiments,
        layer_count=args.layer_count,
        data_directory=args.data_directory,
        enable_parallel=not args.disable_parallel,
        max_workers=args.max_workers,
        enable_caching=not args.disable_caching,
        batch_size=args.batch_size,
        memory_limit_gb=args.memory_limit_gb
    )
    
    batch_pipeline.run()


if __name__ == "__main__":
    main()
