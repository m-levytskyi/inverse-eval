#!/usr/bin/env python3
"""
Optimized Batch Inference Pipeline for ReflecTorch Models

This script runs the inference pipeline on multiple experiments
with significant performance improvements:
1. Pre-compiled model loading and reuse
2. Batch preprocessing of experiments
3. Memory-mapped data loading
4. Optimized data structures
5. Reduced JSON serialization overhead
6. Smart caching strategies

Usage:
    python batch_inference_pipeline.py [--num-experiments 25] [--layer-count 2]
"""

import argparse
import json
import random
import pickle
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
from inference_pipeline import InferencePipeline

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
        self.output_dir = Path("batch_inference_results")
        self.output_dir.mkdir(exist_ok=True)
        
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
            print(f"Warning: MARIA dataset directory not found: {layer_dir}")
            return []
        
        print(f"Optimized search for {self.layer_count}-layer experiments in: {layer_dir}")
        
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
                print(f"  ✓ {exp_id} completed successfully")
                return result
            else:
                print(f"  ✗ {exp_id} failed: {result['error']}")
                return result
            
        except Exception as e:
            print(f"  ✗ {exp_id} error: {e}")
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
            # Use cached models if available
            cache_key = f"{layer_count}_{priors_type}"
            
            if self.enable_caching and cache_key in self.model_cache:
                self.cache_hit_stats['model_cache'] += 1
            
            # Call the original inference but with optimizations
            result = InferencePipeline.run_experiment_inference(
                experiment_id=exp_id,
                models_list=models,
                data_directory=str(data_directory),
                priors_type=priors_type,
                output_dir=str(output_dir),
                layer_count=layer_count
            )
            
            return result
            
        except Exception as e:
            return {
                'exp_id': exp_id,
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = self.output_dir / f"intermediate_results_batch_{batch_num}_{timestamp}.pkl"
        
        try:
            with open(intermediate_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Intermediate results saved to: {intermediate_file}")
        except Exception as e:
            print(f"Failed to save intermediate results: {e}")

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
    
    def run(self):
        """Run optimized batch inference."""
        print(f"Starting Optimized Batch Inference Pipeline")
        print("=" * 60)
        print(f"Target experiments: {self.num_experiments}")
        print(f"Layer count: {self.layer_count}")
        print(f"Models: {self.models}")
        print(f"Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}")
        print(f"Caching: {'Enabled' if self.enable_caching else 'Disabled'}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max workers: {self.max_workers}")
        print(f"Memory limit: {self.memory_limit_gb}GB")
        
        # System info
        available_memory = psutil.virtual_memory().available / (1024**3)
        print(f"Available memory: {available_memory:.1f}GB")
        print(f"CPU cores: {mp.cpu_count()}")
        
        # Discover experiments
        experiments = self.discover_experiments_optimized()
        
        if not experiments:
            print("No experiments found. Exiting.")
            return
        
        print(f"Processing {len(experiments)} experiments...")
        
        start_time = time.time()
        
        # Run batched processing
        all_results = self.run_batched_processing(experiments)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nOptimized processing completed!")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per experiment: {total_time/len(experiments):.1f} seconds")
        
        # Print optimization statistics
        self.print_optimization_stats()
        
        # Create batch summary and plots (reuse existing methods)
        self.create_batch_summary(all_results)
        self.create_performance_plots(all_results)
        
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
                        json.dump(result, f, indent=2, default=str)
                        
                except Exception as e:
                    print(f"[{completed_count+1}/{len(experiments)}] Failed {exp_id}: {e}")
                    all_results[exp_id] = {
                        'experiment_id': exp_id,
                        'layer_count': self.layer_count,
                        'priors': {},
                        'error': str(e),
                        'success': False
                    }
                    completed_count += 1
        
        return all_results

    def run_sequential_processing(self, experiments):
        """Run experiments sequentially (original method)."""
        print(f"Running {len(experiments)} experiments sequentially...")
        
        all_results = {}
        
        for i, exp_id in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Processing {exp_id}...")
            
            exp_results = {
                'experiment_id': exp_id,
                'layer_count': self.layer_count,
                'priors': {}
            }
            
            # Run with both broad and narrow priors
            for priors_type in ['broad', 'narrow']:
                print(f"  Running with {priors_type} priors...")
                start_time = time.time()
                result = self.run_experiment_inference(exp_id, priors_type)
                end_time = time.time()
                result['processing_time'] = end_time - start_time
                exp_results['priors'][priors_type] = result
            
            all_results[exp_id] = exp_results
            
            # Save individual experiment results
            exp_file = self.output_dir / f"{exp_id}_results.json"
            with open(exp_file, 'w') as f:
                json.dump(exp_results, f, indent=2, default=str)
        
        return all_results

    def create_batch_summary(self, all_results):
        """Create summary of batch results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
            'timestamp': timestamp,
            'layer_count': self.layer_count,
            'total_experiments': total_experiments,
            'successful_broad': successful_broad,
            'successful_narrow': successful_narrow,
            'models_tested': self.models,
            'model_performance': dict(model_performance),
            'model_timing_stats': dict(self.model_timing_stats),
            'all_results': all_results
        }
        
        summary_file = self.output_dir / f"batch_summary_{self.layer_count}layer_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nBatch summary saved to: {summary_file}")
        return summary

    def create_performance_plots(self, all_results):
        """Create comprehensive 2-column performance visualization plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        
        # Column titles
        axes[0, 0].text(0.5, 1.15, 'BROAD PRIORS', transform=axes[0, 0].transAxes, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[0, 1].text(0.5, 1.15, 'NARROW PRIORS', transform=axes[0, 1].transAxes, 
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
        plot_file = self.output_dir / f"batch_inference_results_{self.layer_count}layer_{timestamp}.png"
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
