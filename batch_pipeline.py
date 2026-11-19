#!/usr/bin/env python3
"""
Batch inference pipeline for reflectometry experiments.

This module provides a clean interface for running batch processing on multiple
experiments, building upon the single experiment processing in simple_pipeline.
"""

import sys
import argparse
import hashlib
import math
import time
import logging
import subprocess
from importlib import metadata
from pathlib import Path
from datetime import datetime
import torch

# Import core functionality
from simple_pipeline import (
    create_inference_model,
    resolve_inference_config,
    run_single_experiment,
)
from parameter_discovery import (
    discover_batch_experiments,
    parse_true_parameters_from_model_file,
    discover_experiment_files,
    check_experiment_within_constraints,
)
from batch_analysis import (
    create_summary_statistics,
    print_summary_statistics,
    print_constraint_mape_summary,
    detect_edge_cases,
)
from plotting_utils import create_batch_analysis_plots
from utils import ensure_directory_exists, write_json_atomic
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
DEFAULT_PREPROCESSING_THRESHOLD = 0.75  # Relative error threshold (75%)
DEFAULT_PREPROCESSING_CONSECUTIVE = 5  # Consecutive high-error points
DEFAULT_PREPROCESSING_REMOVE_SINGLES = False

# Constraints configuration
DEFAULT_APPLY_CONSTRAINTS = True  # Apply physical constraints

# SLD fixing configuration
DEFAULT_FIX_SLD_MODE = "none"  # SLD fixing mode: "none", "backing", "all"

# Prior bounds configuration
USE_NARROW_PRIORS = True  # Set to False to use broad priors
NARROW_PRIORS_DEVIATION = 0.99  # Deviation for narrow priors
PRIORS_TYPE = "narrow" if USE_NARROW_PRIORS else "broad"

# Prominent features configuration
DEFAULT_USE_PROMINENT_FEATURES = False  # Prominent features analysis

# =============================================================================


logger = logging.getLogger(__name__)


def _file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _collect_provenance():
    root = Path(__file__).resolve().parent
    provenance = {"python": sys.version.split()[0], "packages": {}}
    for package in ("reflectorch", "torch", "numpy", "scipy"):
        try:
            provenance["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            provenance["packages"][package] = None
    for name, path in (
        ("repository", root),
        ("framework", root / "vendor/nflows_reflectorch"),
    ):
        try:
            commit = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            dirty = subprocess.run(
                [
                    "git",
                    "-C",
                    str(path),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            provenance[name] = {"commit": commit, "dirty": bool(dirty)}
        except (OSError, subprocess.SubprocessError):
            provenance[name] = {"commit": None, "dirty": None}
    provenance["constraints"] = _file_identity(root / "model_constraints.json")
    return provenance


class BatchInferencePipeline:
    """
    Batch processing pipeline that leverages simple_pipeline functions.
    """

    def __init__(
        self,
        num_experiments=DEFAULT_NUM_EXPERIMENTS,
        layer_count=DEFAULT_LAYER_COUNT,
        output_dir=DEFAULT_OUTPUT_DIR,
        data_directory=DEFAULT_DATA_DIRECTORY,
        enable_preprocessing=DEFAULT_ENABLE_PREPROCESSING,
        apply_constraints=DEFAULT_APPLY_CONSTRAINTS,
        use_narrow_priors=USE_NARROW_PRIORS,
        narrow_priors_deviation=NARROW_PRIORS_DEVIATION,
        experiment_ids=None,
        fix_sld_mode=DEFAULT_FIX_SLD_MODE,
        use_prominent_features=DEFAULT_USE_PROMINENT_FEATURES,
        priors_type=None,
        use_theoretical=False,
        inference_backend="predict",
        config_name=None,
        nf_num_samples=1000,
        nf_enable_importance_sampling=True,
        use_sigmas_input=False,
    ):
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
            priors_type: Explicit priors type - "narrow" or "constraint_based"
                        (overrides use_narrow_priors if provided)
            use_theoretical: If True, use theoretical curves; if False (default), use experimental curves
            inference_backend: Inference backend: "predict" (default) or "nf"
            config_name: Optional reflectorch config name (overrides backend default)
            nf_num_samples: NF backend only: number of samples
            nf_enable_importance_sampling: NF backend only: importance sampling toggle
            use_sigmas_input: Supply measured uncertainties to the inference backend
        """
        if num_experiments is not None and num_experiments < 0:
            raise ValueError("num_experiments must be nonnegative or None")
        if nf_num_samples <= 0:
            raise ValueError("nf_num_samples must be positive")
        self.experiment_ids = (
            list(dict.fromkeys(experiment_ids)) if experiment_ids else None
        )
        self.num_experiments = (
            len(self.experiment_ids) if self.experiment_ids else num_experiments
        )
        self.layer_count = layer_count
        self.data_directory = Path(data_directory)
        self.enable_preprocessing = enable_preprocessing
        self.apply_constraints = apply_constraints
        self.use_narrow_priors = use_narrow_priors
        self.narrow_priors_deviation = narrow_priors_deviation
        self.fix_sld_mode = fix_sld_mode
        self.use_prominent_features = use_prominent_features
        self.use_theoretical = use_theoretical
        self.inference_backend = inference_backend
        self.config_name = resolve_inference_config(
            layer_count, inference_backend, config_name
        )
        self.nf_num_samples = nf_num_samples
        self.nf_enable_importance_sampling = nf_enable_importance_sampling
        self.use_sigmas_input = use_sigmas_input

        # Determine priors type - explicit parameter takes precedence
        if priors_type is not None:
            self.priors_type = priors_type
            self.use_narrow_priors = priors_type in ["narrow", "constraint_based"]
        else:
            self.priors_type = "narrow" if use_narrow_priors else "constraint_based"

        # Create timestamped output directory under the configured output_dir
        timestamp = datetime.now().strftime("%d%B%Y_%H_%M").lower()
        self.base_output_dir = Path(output_dir)
        folder_name = self._generate_folder_name(timestamp)
        self.output_dir = self.base_output_dir / folder_name
        ensure_directory_exists(self.output_dir)
        self.results = {}
        self._inference_model = None
        self._model_error = None
        self.run_metadata = {
            "config": {
                "num_experiments": self.num_experiments,
                "experiment_ids": self.experiment_ids,
                "layer_count": self.layer_count,
                "data_directory": str(self.data_directory.resolve()),
                "enable_preprocessing": self.enable_preprocessing,
                "preprocessing_threshold": DEFAULT_PREPROCESSING_THRESHOLD,
                "preprocessing_consecutive": DEFAULT_PREPROCESSING_CONSECUTIVE,
                "preprocessing_remove_singles": DEFAULT_PREPROCESSING_REMOVE_SINGLES,
                "apply_constraints": self.apply_constraints,
                "priors_type": self.priors_type,
                "priors_deviation": self.narrow_priors_deviation
                if self.use_narrow_priors
                else 0.5,
                "fix_sld_mode": self.fix_sld_mode,
                "use_prominent_features": self.use_prominent_features,
                "use_theoretical": self.use_theoretical,
                "inference_backend": self.inference_backend,
                "config_name": self.config_name,
                "nf_num_samples": self.nf_num_samples,
                "nf_enable_importance_sampling": self.nf_enable_importance_sampling,
                "use_sigmas_input": self.use_sigmas_input,
                "clip_prediction": True,
                "q_resolution": 0.1,
                "metric_policy": "constraint_mape_global_max_log_likelihood"
                if self.inference_backend == "nf"
                else "constraint_mape_polished_prediction",
                "noise_policy": "provided_sigmas"
                if self.use_sigmas_input
                else (
                    "relative_0.2_absolute_floor_1e-12"
                    if self.inference_backend == "nf"
                    else "model_default"
                ),
            },
            "provenance": _collect_provenance(),
            "experiment_ids": [],
            "started_at": datetime.now().astimezone().isoformat(),
        }
        self._write_metadata("initialized")

        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Layer count: {self.layer_count}")
        if experiment_ids:
            logger.info(f"Processing specific experiments: {experiment_ids}")
        else:
            logger.info(f"Processing first {self.num_experiments} experiments")
        logger.info(
            f"Preprocessing: {'enabled' if self.enable_preprocessing else 'disabled'}"
        )
        logger.info(
            f"Physical constraints: {'enabled' if self.apply_constraints else 'disabled'}"
        )
        logger.info(f"Prior bounds: {self.priors_type}")
        logger.info(f"Inference backend: {self.inference_backend}")
        if self.config_name:
            logger.info(f"Model config: {self.config_name}")
        if self.inference_backend == "nf":
            logger.info(f"NF num samples: {self.nf_num_samples}")
            logger.info(
                f"NF importance sampling: {'enabled' if self.nf_enable_importance_sampling else 'disabled'}"
            )
        logger.info(
            f"Data source: {'THEORETICAL curves' if self.use_theoretical else 'EXPERIMENTAL curves'}"
        )
        logger.info(f"SLD fixing mode: {self.fix_sld_mode}")
        if self.use_narrow_priors:
            logger.info(
                f"Narrow priors deviation: ±{self.narrow_priors_deviation * 100:.1f}%"
            )
        logger.info(
            f"Prominent features: {'enabled' if self.use_prominent_features else 'disabled'}"
        )

    def _write_metadata(self, status):
        self.run_metadata["status"] = status
        self.run_metadata["counts"] = {
            "completed": len(self.results),
            "successful": sum(
                bool(value.get("success")) for value in self.results.values()
            ),
            "failed": sum(
                not value.get("success") and not value.get("excluded_as_outlier")
                for value in self.results.values()
            ),
            "outliers": sum(
                bool(value.get("excluded_as_outlier"))
                for value in self.results.values()
            ),
        }
        write_json_atomic(self.output_dir / "run_metadata.json", self.run_metadata)

    def _get_inference_model(self):
        if self._model_error is not None:
            raise RuntimeError(self._model_error)
        if self._inference_model is None:
            try:
                self._inference_model = create_inference_model(
                    layer_count=self.layer_count,
                    inference_backend=self.inference_backend,
                    config_name=self.config_name,
                )
                model = self._inference_model
                self.run_metadata["config"]["inference_device"] = str(model.device)
                self.run_metadata["provenance"]["model_config"] = _file_identity(
                    Path(model.config_dir) / model.config_name
                )
                self.run_metadata["provenance"]["model_weights"] = _file_identity(
                    Path(model.model_dir) / model.model_name
                )
                self._write_metadata("running")
            except Exception as error:
                self._model_error = (
                    f"Model initialization failed: {type(error).__name__}: {error}"
                )
                raise RuntimeError(self._model_error) from error
        return self._inference_model

    def _record_result(self, experiment_id, result):
        if (
            not experiment_id
            or experiment_id in (".", "..")
            or any(char in experiment_id for char in ("/", "\\"))
        ):
            raise ValueError(
                f"Invalid experiment ID for a result filename: {experiment_id!r}"
            )
        write_json_atomic(
            self.output_dir / "experiments" / f"{experiment_id}.json", result
        )
        self.results[experiment_id] = result

    def _get_next_index(self):
        """Get the next available index by scanning existing directories."""
        batch_results_dir = self.base_output_dir
        if not batch_results_dir.exists():
            return 1

        existing_indices = []
        for folder in batch_results_dir.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                # Extract index from folder name (assuming it starts with digits followed by underscore)
                if "_" in folder_name:
                    index_part = folder_name.split("_")[0]
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
            # Default case (shouldn't be reached with current code)
            return "defaultpriors"

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
            prior_info,
        ]

        # Add optional components if not empty
        if sld_fix_info:
            components.append(sld_fix_info)
        if prominent_info:
            components.append(prominent_info)

        components.append(timestamp)

        return "_".join(components)

    def discover_experiments(self):
        """Select a deterministic, deduplicated set, then apply the requested limit."""
        experiments = discover_batch_experiments(
            data_directory=str(self.data_directory),
            layer_count=self.layer_count,
            num_experiments=None,
            experiment_ids=self.experiment_ids,
            use_theoretical=self.use_theoretical,
        )
        experiments = list(dict.fromkeys(experiments))
        if not self.experiment_ids:
            experiments.sort()
        if self.use_prominent_features:
            experiments_with_peaks = set(
                find_experiments_with_prominent_peaks(
                    layer_count=self.layer_count,
                    data_directory=str(self.data_directory),
                    verbose=True,
                )
            )
            experiments = [
                exp_id for exp_id in experiments if exp_id in experiments_with_peaks
            ]
        return experiments[: self.num_experiments]

    def process_single_experiment_wrapper(self, experiment_id):
        """
        Wrapper for processing a single experiment with error handling and fallback logic.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary with results and metadata
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing experiment: {experiment_id}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        try:
            results = run_single_experiment(
                experiment_id=experiment_id,
                layer_count=self.layer_count,
                enable_preprocessing=self.enable_preprocessing,
                preprocessing_threshold=DEFAULT_PREPROCESSING_THRESHOLD,
                preprocessing_consecutive=DEFAULT_PREPROCESSING_CONSECUTIVE,
                preprocessing_remove_singles=DEFAULT_PREPROCESSING_REMOVE_SINGLES,
                apply_constraints=self.apply_constraints,
                priors_type=self.priors_type,
                priors_deviation=self.narrow_priors_deviation
                if self.use_narrow_priors
                else 0.5,
                fix_sld_mode=self.fix_sld_mode,
                use_theoretical=self.use_theoretical,
                inference_backend=self.inference_backend,
                config_name=self.config_name,
                nf_num_samples=self.nf_num_samples,
                nf_enable_importance_sampling=self.nf_enable_importance_sampling,
                use_sigmas_input=self.use_sigmas_input,
                data_directory=str(self.data_directory),
                inference_model=self._get_inference_model(),
            )

            # Success with primary priors
            processing_time = time.time() - start_time
            results["experiment_id"] = experiment_id
            results["processing_time"] = processing_time
            results["success"] = True
            results["priors_used"] = self.priors_type
            results["fallback_applied"] = False

            logger.info(
                f"  {experiment_id} completed successfully in {processing_time:.1f}s"
            )
            return results

        except Exception as error:
            error_msg = str(error)
            logger.exception("  Failed: %s", error_msg)

            # No fallback - fail immediately
            processing_time = time.time() - start_time
            return {
                "experiment_id": experiment_id,
                "processing_time": processing_time,
                "success": False,
                "error": error_msg,
                "error_type": type(error).__name__,
            }

    def process_experiments_sequential(self, experiments):
        """
        Process experiments sequentially with outlier detection for constraint-based priors.

        Args:
            experiments: List of experiment IDs

        Returns:
            Dictionary of results
        """
        logger.info(f"\nProcessing {len(experiments)} experiments sequentially...")

        all_results = self.results
        successful_count = 0
        failed_count = 0
        outlier_count = 0
        start_time = time.time()

        for i, exp_id in enumerate(experiments, 1):
            logger.info(f"\nProgress: {i}/{len(experiments)} experiments")
            result = None

            # Check for outliers if using constraint-based priors
            if self.priors_type == "constraint_based":
                # Try to get true parameters for outlier checking
                try:
                    data_file, model_file, _ = discover_experiment_files(
                        exp_id,
                        str(self.data_directory),
                        self.layer_count,
                        use_theoretical=self.use_theoretical,
                    )

                    if model_file:
                        true_params_dict = parse_true_parameters_from_model_file(
                            str(model_file)
                        )

                        # Check if experiment is within constraints
                        is_within, outlier_params = check_experiment_within_constraints(
                            exp_id, true_params_dict, self.layer_count
                        )

                        if not is_within:
                            logger.info(f"  OUTLIER DETECTED: {exp_id}")
                            logger.info("  True parameters outside constraint bounds:")
                            for (
                                param_name,
                                value,
                                min_bound,
                                max_bound,
                            ) in outlier_params:
                                logger.info(
                                    f"    {param_name}: {value:.3f} not in [{min_bound:.3f}, {max_bound:.3f}]"
                                )

                            # Record as outlier
                            result = {
                                "experiment_id": exp_id,
                                "success": False,
                                "excluded_as_outlier": True,
                                "outlier_parameters": [
                                    {
                                        "parameter": param_name,
                                        "true_value": float(value),
                                        "constraint_min": float(min_bound),
                                        "constraint_max": float(max_bound),
                                    }
                                    for param_name, value, min_bound, max_bound in outlier_params
                                ],
                                "processing_time": 0,
                            }
                            invalid = [
                                f"{name}={value}"
                                for name, value, _, _ in outlier_params
                                if not math.isfinite(value)
                            ]
                            if invalid:
                                result = {
                                    "experiment_id": exp_id,
                                    "success": False,
                                    "processing_time": 0,
                                    "error_type": "ValueError",
                                    "error": "Non-finite true parameters: "
                                    + ", ".join(invalid),
                                }
                except (OSError, ValueError, KeyError) as e:
                    logger.warning("  Could not check for outliers: %s", e)
                    # Continue with normal processing if outlier check fails

            # Process experiment normally
            if result is None:
                result = self.process_single_experiment_wrapper(exp_id)
            self._record_result(exp_id, result)

            if result.get("excluded_as_outlier", False):
                outlier_count += 1
            elif result.get("success", False):
                successful_count += 1
            else:
                failed_count += 1

        end_time = time.time()
        total_time = end_time - start_time

        logger.info("\nSequential processing completed!")
        logger.info(f"  Total time: {total_time:.1f} seconds")
        logger.info(f"  Successful: {successful_count}")
        logger.info(f"  Failed: {failed_count}")
        if outlier_count > 0:
            logger.info(f"  Excluded as outliers: {outlier_count}")
        logger.info(
            f"  Average time per experiment: {total_time / len(experiments) if experiments else 0:.1f} seconds"
        )

        return all_results

    def save_results(self, all_results):
        """Save batch processing results to files."""
        logger.info(f"Saving results to {self.output_dir}")

        # Save detailed results as JSON
        results_file = self.output_dir / "batch_results.json"

        write_json_atomic(results_file, all_results)

        logger.info(f"  Detailed results saved to: {results_file}")

        # Separate failed experiments and outliers
        failed_results = {
            k: v
            for k, v in all_results.items()
            if not v.get("success", False) and not v.get("excluded_as_outlier", False)
        }
        outlier_results = {
            k: v for k, v in all_results.items() if v.get("excluded_as_outlier", False)
        }

        # Save failed experiments (excluding outliers) to separate file
        if failed_results:
            failed_file = self.output_dir / "failed_experiments.json"
            write_json_atomic(failed_file, failed_results)

            logger.info(f"  Failed experiments saved to: {failed_file}")
            logger.info(f"  Total failed experiments: {len(failed_results)}")
        else:
            logger.info("  No failed experiments to save")

        # Save outliers to separate file
        if outlier_results:
            outlier_file = self.output_dir / "outlier_experiments.json"
            write_json_atomic(outlier_file, outlier_results)

            logger.info(f"  Outlier experiments saved to: {outlier_file}")
            logger.info(f"  Total outlier experiments: {len(outlier_results)}")

        # Create and save summary statistics
        successful_results = {
            k: v for k, v in all_results.items() if v.get("success", False)
        }

        summary = create_summary_statistics(
            all_results,
            self.layer_count,
            self.enable_preprocessing,
            self.priors_type,
            self.narrow_priors_deviation if self.use_narrow_priors else 0.5,
        )
        summary_file = self.output_dir / f"batch_summary_{self.layer_count}layer.json"
        write_json_atomic(summary_file, summary)
        logger.info(f"  Summary statistics saved to: {summary_file}")
        print_summary_statistics(summary)
        if successful_results:
            print_constraint_mape_summary(successful_results)

        return (
            results_file,
            len(successful_results),
            len(all_results),
            len(failed_results),
            len(outlier_results),
        )

    def run(self):
        """Run the complete batch processing pipeline."""
        logger.info("STARTING BATCH INFERENCE PIPELINE")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            self.run_metadata["provenance"]["torch_initial_seed"] = torch.initial_seed()
            experiments = self.discover_experiments()
            self.run_metadata["experiment_ids"] = experiments
            self._write_metadata("running")
            if not experiments:
                logger.error("No experiments found!")
                self.save_results({})
                self._write_metadata("failed")
                return None
            all_results = self.process_experiments_sequential(experiments)
            _, successful_count, total_count, failed_count, outlier_count = (
                self.save_results(all_results)
            )
        except KeyboardInterrupt:
            self.save_results(self.results)
            self._write_metadata("interrupted")
            raise
        except Exception as error:
            self.run_metadata["error"] = f"{type(error).__name__}: {error}"
            self._write_metadata("failed")
            raise

        status = (
            "failed"
            if failed_count
            else ("completed" if successful_count else "completed_no_eligible_results")
        )
        self.run_metadata["finished_at"] = datetime.now().astimezone().isoformat()
        self._write_metadata(status)

        # Detect edge cases
        successful_results = {
            k: v for k, v in all_results.items() if v.get("success", False)
        }
        if successful_results:
            detect_edge_cases(successful_results)

        # Create plots with outlier and failure statistics
        if successful_results:
            try:
                logger.info("\nCreating analysis plots...")
                create_batch_analysis_plots(
                    successful_results,
                    layer_count=self.layer_count,
                    output_dir=str(self.output_dir),
                    save=True,
                    use_prominent_features=self.use_prominent_features,
                )
                logger.info("Analysis plots completed")
            except (OSError, ValueError, RuntimeError) as e:
                logger.exception("Failed to create plots: %s", e)

        end_time = time.time()
        total_pipeline_time = end_time - start_time

        # Final summary
        logger.info(f"\n{'=' * 60}")
        logger.info("BATCH PIPELINE COMPLETED")
        logger.info(f"{'=' * 60}")
        logger.info("Batch status: %s", status)
        logger.info(f"Total time: {total_pipeline_time:.1f} seconds")
        logger.info(f"Successful experiments: {successful_count}/{total_count}")
        if outlier_count > 0:
            logger.info(f"Excluded as outliers: {outlier_count}")
        if failed_count > 0:
            logger.info(f"Failed experiments: {failed_count}")
        logger.info(f"Success rate: {successful_count / total_count * 100:.1f}%")
        logger.info(f"Results saved to: {self.output_dir}")

        return all_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference on reflectometry experiments"
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=None,
        help="Number of experiments to process (default: all)",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        choices=[1, 2],
        default=1,
        help="Number of layers (1 or 2, default: 1)",
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        default="data",
        help="Data directory path (default: data)",
    )
    parser.add_argument(
        "--disable-preprocessing",
        action="store_true",
        help="Disable data preprocessing",
    )
    parser.add_argument(
        "--disable-constraints",
        action="store_true",
        help="Disable physical constraints application",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_results",
        help="Output directory (default: batch_results)",
    )
    parser.add_argument(
        "--experiment-ids",
        type=str,
        nargs="+",
        help="Specific experiment IDs to process (e.g., s005156 s004141)",
    )
    parser.add_argument(
        "--fix-sld-mode",
        type=str,
        choices=["none", "backing", "all"],
        default=DEFAULT_FIX_SLD_MODE,
        help=f"SLD fixing mode: none, backing, or all (default: {DEFAULT_FIX_SLD_MODE})",
    )
    parser.add_argument(
        "--use-prominent-features",
        action="store_true",
        help="Enable prominent features analysis",
    )
    parser.add_argument(
        "--priors-deviation",
        type=int,
        choices=[5, 30, 60, 99],
        default=99,
        help="Prior bounds deviation percentage: 5, 30, 60, or 99 (default: 99)",
    )
    parser.add_argument(
        "--priors-type",
        type=str,
        choices=["narrow", "constraint_based"],
        help="Prior bounds type: narrow or constraint_based",
    )

    parser.add_argument(
        "--inference-backend",
        type=str,
        choices=["predict", "nf"],
        default="predict",
        help="Inference backend: predict (default) or nf",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Optional reflectorch config name (overrides backend default)",
    )
    parser.add_argument(
        "--nf-num-samples",
        type=int,
        default=1000,
        help="NF backend: number of samples (default: 1000)",
    )
    parser.add_argument(
        "--nf-disable-importance-sampling",
        action="store_true",
        help="NF backend: disable importance sampling",
    )
    parser.add_argument(
        "--use-sigmas-input",
        action="store_true",
        help="Supply measured uncertainties to the inference backend",
    )
    parser.add_argument(
        "--use-theoretical",
        action="store_true",
        help="Use theoretical curves instead of experimental curves",
    )

    return parser.parse_args()


def main():
    """Main function to run the batch inference pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_arguments()

    try:
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
            narrow_priors_deviation=args.priors_deviation / 100.0,
            priors_type=args.priors_type,
            inference_backend=args.inference_backend,
            config_name=args.config_name,
            nf_num_samples=args.nf_num_samples,
            nf_enable_importance_sampling=not args.nf_disable_importance_sampling,
            use_sigmas_input=args.use_sigmas_input,
            use_theoretical=args.use_theoretical,
        )
        results = batch_pipeline.run()
        if results is None:
            logger.info("\nBatch pipeline failed: No experiments found")
            return 1
        if any(
            not value.get("success") and not value.get("excluded_as_outlier")
            for value in results.values()
        ):
            logger.error("\nBatch pipeline completed with processing failures")
            return 1
        logger.info("\nBatch pipeline completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.warning("\nBatch pipeline interrupted by user")
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        logger.exception("\nBatch pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
