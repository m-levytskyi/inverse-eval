#!/usr/bin/env python3
"""
Consolidated batch pipeline parameter sweep script - CONSTRAINT-BASED PRIORS SWEEP.

This script runs the batch pipeline with constraint-based prior bounds method
across all available experiments in the dataset, testing different parameter combinations.

Configuration can be provided via:
1. Sweep config YAML files in sweep_configs/ directory
2. Command-line arguments

Usage examples:
    python batch_sweep_runner.py --config sweep_configs/baseline.yaml
    python batch_sweep_runner.py --sweep-name nf_anaklasis --nf-config nf_config_mixed_sigmas_qweighted.yaml
    python batch_sweep_runner.py --prior-deviations 5 30 99 --sld-modes none backing all
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import json
import yaml

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Parameter combinations to test
    "prior_deviations": [5, 30, 99],
    "sld_modes": ["none", "backing", "all"],
    "prominent_features": [False, True],
    "priors_type": "constraint_based",
    
    # Inference backend configuration
    "inference_backend": "nf",
    "nf_config_name": "example_nf_config_reflectorch.yaml",
    "nf_num_samples": 1000,
    "nf_disable_importance_sampling": False,
    "use_sigmas_as_input": False,
    
    # Experiment configuration
    "num_experiments": None,
    "layer_count": 1,
    "data_directory": "../reflectorch_devvm/reflectorch/dataset/test",
    
    # Output configuration
    "sweep_results_dir": "sweep_results",
}


# =============================================================================


class BatchPipelineSweep:
    """Automated parameter sweep for batch pipeline."""

    def __init__(self, config=None):
        """Initialize sweep with configuration."""
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.num_experiments = self.config["num_experiments"]
        self.layer_count = self.config["layer_count"]
        self.data_directory = self.config["data_directory"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sweep_dir = (
            Path(self.config["sweep_results_dir"])
            / f"sweep_{timestamp}"
        )
        self.sweep_dir.mkdir(parents=True, exist_ok=True)

        self.results_summary = []
        self.total_runs = (
            len(self.config["prior_deviations"])
            * len(self.config["sld_modes"])
            * len(self.config["prominent_features"])
        )
        self.run_count = 0

        print(f"Initialized sweep with {self.total_runs} total parameter combinations")
        print(f"Results will be saved to: {self.sweep_dir}")

    def run_single_batch(
        self, prior_deviation, sld_mode, use_prominent_features
    ):
        """Run a single batch pipeline instance."""
        self.run_count += 1
        print(f"\n{'='*80}")
        print(f"RUN {self.run_count}/{self.total_runs}")
        print(f"Prior deviation: {prior_deviation}%")
        print(f"SLD mode: {sld_mode}")
        print(f"Prominent features: {use_prominent_features}")
        print(f"Inference backend: {self.config['inference_backend']}")
        if self.config["inference_backend"] == "nf":
            print(f"NF config: {self.config['nf_config_name']}")
            print(f"NF samples: {self.config['nf_num_samples']}")
            print(f"NF importance sampling: {not self.config['nf_disable_importance_sampling']}")
        print(f"{'='*80}\n")

        cmd = [
            sys.executable,
            "batch_pipeline.py",
            "--num-experiments",
            str(self.num_experiments) if self.num_experiments else "all",
            "--layer-count",
            str(self.layer_count),
            "--data-directory",
            self.data_directory,
            "--priors-type",
            self.config["priors_type"],
            "--priors-deviation",
            str(prior_deviation),
            "--sld-mode",
            sld_mode,
            "--inference-backend",
            self.config["inference_backend"],
        ]

        if use_prominent_features:
            cmd.append("--use-prominent-features")

        if self.config["inference_backend"] == "nf":
            if self.config["nf_config_name"]:
                cmd.extend(["--config-name", self.config["nf_config_name"]])
            cmd.extend(["--nf-num-samples", str(self.config["nf_num_samples"])])
            if self.config["nf_disable_importance_sampling"]:
                cmd.append("--nf-disable-importance-sampling")

        if self.config["use_sigmas_as_input"]:
            cmd.append("--use-sigmas-as-input")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200,
            )

            execution_time = time.time() - start_time

            run_result = {
                "run_number": self.run_count,
                "prior_deviation": prior_deviation,
                "sld_mode": sld_mode,
                "use_prominent_features": use_prominent_features,
                "success": True,
                "execution_time": execution_time,
                "return_code": result.returncode,
            }

            print(f"Run {self.run_count} completed successfully in {execution_time:.1f}s")

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            run_result = {
                "run_number": self.run_count,
                "prior_deviation": prior_deviation,
                "sld_mode": sld_mode,
                "use_prominent_features": use_prominent_features,
                "success": False,
                "execution_time": execution_time,
                "return_code": "TIMEOUT",
                "error_output": "Process timed out after 7200 seconds",
            }
            print(f"Run {self.run_count} TIMED OUT after {execution_time:.1f}s")

        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            run_result = {
                "run_number": self.run_count,
                "prior_deviation": prior_deviation,
                "sld_mode": sld_mode,
                "use_prominent_features": use_prominent_features,
                "success": False,
                "execution_time": execution_time,
                "return_code": e.returncode,
                "error_output": e.stderr[:500] if e.stderr else "",
            }
            print(f"Run {self.run_count} FAILED with return code {e.returncode}")

        except Exception as e:
            execution_time = time.time() - start_time
            run_result = {
                "run_number": self.run_count,
                "prior_deviation": prior_deviation,
                "sld_mode": sld_mode,
                "use_prominent_features": use_prominent_features,
                "success": False,
                "execution_time": execution_time,
                "return_code": "EXCEPTION",
                "error_output": str(e),
            }
            print(f"Run {self.run_count} FAILED with exception: {e}")

        self.results_summary.append(run_result)
        self.save_summary()

        return run_result

    def save_summary(self):
        """Save current results summary."""
        summary_file = self.sweep_dir / "sweep_summary.json"

        summary_data = {
            "configuration": self.config,
            "total_runs": self.total_runs,
            "completed_runs": len(self.results_summary),
            "timestamp": datetime.now().isoformat(),
            "results": self.results_summary,
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    def run_full_sweep(self):
        """Run complete parameter sweep."""
        sweep_start = time.time()

        for prior_deviation in self.config["prior_deviations"]:
            for sld_mode in self.config["sld_modes"]:
                for use_prominent in self.config["prominent_features"]:
                    self.run_single_batch(prior_deviation, sld_mode, use_prominent)

        total_sweep_time = time.time() - sweep_start

        successful_runs = sum(1 for r in self.results_summary if r["success"])
        failed_runs = len(self.results_summary) - successful_runs

        print(f"\n{'='*80}")
        print("SWEEP COMPLETE")
        print(f"{'='*80}")
        print(f"Total runs: {len(self.results_summary)}/{self.total_runs}")
        print(f"Successful runs: {successful_runs}/{self.total_runs}")
        print(f"Failed runs: {failed_runs}/{self.total_runs}")
        print(f"Success rate: {successful_runs / self.total_runs * 100:.1f}%")
        print(f"Average time per run: {total_sweep_time / self.total_runs:.1f} seconds")
        print(f"Results saved to: {self.sweep_dir}")

        return self.results_summary


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch pipeline parameter sweep with constraint-based priors"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--sweep-name",
        type=str,
        help="Name for this sweep (used in output directory)"
    )
    
    parser.add_argument(
        "--prior-deviations",
        type=int,
        nargs="+",
        help="Prior deviation percentages to test (e.g., 5 30 99)"
    )
    
    parser.add_argument(
        "--sld-modes",
        type=str,
        nargs="+",
        choices=["none", "backing", "all"],
        help="SLD fixing modes to test"
    )
    
    parser.add_argument(
        "--prominent-features",
        type=str,
        nargs="+",
        choices=["true", "false"],
        help="Test with prominent features enabled/disabled"
    )
    
    parser.add_argument(
        "--nf-config",
        type=str,
        help="NF config YAML filename"
    )
    
    parser.add_argument(
        "--use-sigmas-as-input",
        action="store_true",
        help="Use sigmas as input channel (requires 2-channel model)"
    )
    
    parser.add_argument(
        "--num-experiments",
        type=int,
        help="Number of experiments to process (None = all)"
    )
    
    parser.add_argument(
        "--layer-count",
        type=int,
        help="Layer count to filter experiments"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the parameter sweep."""
    args = parse_arguments()
    
    # Load config from file if provided
    config = {}
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config_from_yaml(args.config)
    
    # Override with command line arguments
    if args.sweep_name:
        config["sweep_results_dir"] = f"sweep_results_{args.sweep_name}"
    
    if args.prior_deviations:
        config["prior_deviations"] = args.prior_deviations
    
    if args.sld_modes:
        config["sld_modes"] = args.sld_modes
    
    if args.prominent_features:
        config["prominent_features"] = [
            p.lower() == "true" for p in args.prominent_features
        ]
    
    if args.nf_config:
        config["nf_config_name"] = args.nf_config
    
    if args.use_sigmas_as_input:
        config["use_sigmas_as_input"] = True
    
    if args.num_experiments:
        config["num_experiments"] = args.num_experiments
    
    if args.layer_count:
        config["layer_count"] = args.layer_count
    
    print("Starting automated batch pipeline parameter sweep...")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    sweep = BatchPipelineSweep(config=config)

    try:
        sweep.run_full_sweep()
        print("\nParameter sweep completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\nParameter sweep interrupted by user")
        sweep.save_summary()
        return 1

    except Exception as e:
        print(f"\nParameter sweep failed with error: {e}")
        sweep.save_summary()
        return 1


if __name__ == "__main__":
    sys.exit(main())
