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
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml
from utils import write_json_atomic

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
    "config_name": None,
    "nf_num_samples": 1000,
    "nf_disable_importance_sampling": False,
    "use_sigmas_as_input": False,
    "use_theoretical": False,
    "disable_preprocessing": False,
    "disable_constraints": False,
    "experiment_ids": None,
    # Experiment configuration
    "num_experiments": None,
    "layer_count": 1,
    "data_directory": "dataset/test",
    # Output configuration
    "sweep_results_dir": "sweep_results",
}


# =============================================================================


logger = logging.getLogger(__name__)


class BatchPipelineSweep:
    """Automated parameter sweep for batch pipeline."""

    def __init__(self, config=None):
        """Initialize sweep with configuration."""
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("Sweep configuration must be a mapping")
            self.config.update(config)
        for key, allowed in (
            ("prior_deviations", {5, 30, 60, 99}),
            ("sld_modes", {"none", "backing", "all"}),
            ("prominent_features", {False, True}),
        ):
            values = self.config[key]
            if (
                not isinstance(values, list)
                or not values
                or any(value not in allowed for value in values)
            ):
                raise ValueError(f"{key} must be a nonempty list drawn from {allowed}")

        self.num_experiments = self.config["num_experiments"]
        self.layer_count = self.config["layer_count"]
        self.data_directory = self.config["data_directory"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.sweep_dir = (
            Path(self.config["sweep_results_dir"]).resolve() / f"sweep_{timestamp}"
        )
        self.sweep_dir.mkdir(parents=True, exist_ok=False)

        self.results_summary = []
        self.total_runs = (
            len(self.config["prior_deviations"])
            * len(self.config["sld_modes"])
            * len(self.config["prominent_features"])
        )
        self.run_count = 0
        self.save_summary()

        logger.info(
            f"Initialized sweep with {self.total_runs} total parameter combinations"
        )
        logger.info(f"Results will be saved to: {self.sweep_dir}")

    def run_single_batch(self, prior_deviation, sld_mode, use_prominent_features):
        """Run a single batch pipeline instance."""
        self.run_count += 1
        logger.info(f"\n{'=' * 80}")
        logger.info(f"RUN {self.run_count}/{self.total_runs}")
        logger.info(f"Prior deviation: {prior_deviation}%")
        logger.info(f"SLD mode: {sld_mode}")
        logger.info(f"Prominent features: {use_prominent_features}")
        logger.info(f"Inference backend: {self.config['inference_backend']}")
        if self.config["inference_backend"] == "nf":
            logger.info(f"NF config: {self.config['nf_config_name']}")
            logger.info(f"NF samples: {self.config['nf_num_samples']}")
            logger.info(
                f"NF importance sampling: {not self.config['nf_disable_importance_sampling']}"
            )
        logger.info(f"{'=' * 80}\n")

        batch_output_root = self.sweep_dir / f"run_{self.run_count:03d}"
        batch_output_root.mkdir()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().with_name("batch_pipeline.py")),
            "--output-dir",
            str(batch_output_root),
        ]

        if self.num_experiments is not None:
            cmd.extend(["--num-experiments", str(self.num_experiments)])

        cmd.extend(
            [
                "--layer-count",
                str(self.layer_count),
                "--data-directory",
                self.data_directory,
                "--priors-type",
                self.config["priors_type"],
                "--priors-deviation",
                str(prior_deviation),
                "--fix-sld-mode",
                sld_mode,
                "--inference-backend",
                self.config["inference_backend"],
            ]
        )

        if use_prominent_features:
            cmd.append("--use-prominent-features")

        for key in ("disable_preprocessing", "disable_constraints"):
            if self.config[key]:
                cmd.append("--" + key.replace("_", "-"))
        if self.config["experiment_ids"]:
            cmd.extend(["--experiment-ids", *self.config["experiment_ids"]])

        config_name = self.config["config_name"]
        if config_name is None and self.config["inference_backend"] == "nf":
            config_name = self.config["nf_config_name"]
        if config_name:
            cmd.extend(["--config-name", config_name])

        if self.config["inference_backend"] == "nf":
            cmd.extend(["--nf-num-samples", str(self.config["nf_num_samples"])])
            if self.config["nf_disable_importance_sampling"]:
                cmd.append("--nf-disable-importance-sampling")

        if self.config["use_sigmas_as_input"]:
            cmd.append("--use-sigmas-input")

        if self.config["use_theoretical"]:
            cmd.append("--use-theoretical")

        start_time = time.time()
        run_result = {
            "run_number": self.run_count,
            "prior_deviation": prior_deviation,
            "sld_mode": sld_mode,
            "use_prominent_features": use_prominent_features,
            "success": False,
            "return_code": "EXCEPTION",
            "output_dir": str(batch_output_root),
            "command": cmd,
        }

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=Path(__file__).resolve().parent,
            )
            run_result.update(success=True, return_code=result.returncode)

        except subprocess.TimeoutExpired:
            run_result.update(
                return_code="TIMEOUT",
                error_output="Process timed out after 7200 seconds",
            )

        except subprocess.CalledProcessError as e:
            run_result.update(
                return_code=e.returncode,
                error_output=e.stderr[-2000:] if e.stderr else "",
            )

        except OSError as e:
            run_result["error_output"] = str(e)
            logger.exception(
                "Run %s failed while launching subprocess: %s", self.run_count, e
            )
        except KeyboardInterrupt:
            run_result.update(return_code=130, error_output="Interrupted by user")
            raise
        finally:
            run_result["execution_time"] = time.time() - start_time
            child_directories = sorted(
                path for path in batch_output_root.iterdir() if path.is_dir()
            )
            if len(child_directories) == 1:
                run_result["output_dir"] = str(child_directories[0])
            self.results_summary.append(run_result)
            self.save_summary()
            logger.info(
                "Run %s finished with status %s; results: %s",
                self.run_count,
                run_result["return_code"],
                run_result["output_dir"],
            )

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

        write_json_atomic(summary_file, summary_data)

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

        logger.info(f"\n{'=' * 80}")
        logger.info("SWEEP COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total runs: {len(self.results_summary)}/{self.total_runs}")
        logger.info(f"Successful runs: {successful_runs}/{self.total_runs}")
        logger.info(f"Failed runs: {failed_runs}/{self.total_runs}")
        logger.info(f"Success rate: {successful_runs / self.total_runs * 100:.1f}%")
        logger.info(
            f"Average time per run: {total_sweep_time / self.total_runs:.1f} seconds"
        )
        logger.info(f"Results saved to: {self.sweep_dir}")

        return self.results_summary


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Sweep configuration must be a YAML mapping")
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch pipeline parameter sweep with constraint-based priors"
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    parser.add_argument(
        "--sweep-name", type=str, help="Name for this sweep (used in output directory)"
    )

    parser.add_argument(
        "--num-experiments",
        type=int,
        help="Number of experiments to process (None = all)",
    )

    parser.add_argument(
        "--layer-count",
        type=int,
        choices=[1, 2],
        help="Layer count to filter experiments (1 or 2)",
    )

    parser.add_argument("--data-directory", type=str, help="Data directory path")

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
        "--experiment-ids",
        type=str,
        nargs="+",
        help="Specific experiment IDs to process (e.g., s005156 s004141)",
    )

    parser.add_argument(
        "--prior-deviations",
        type=int,
        nargs="+",
        help="Prior deviation percentages to test (e.g., 5 30 60 99)",
    )

    parser.add_argument(
        "--priors-type",
        type=str,
        choices=["narrow", "constraint_based"],
        help="Prior bounds type: narrow or constraint_based",
    )

    parser.add_argument(
        "--sld-modes",
        type=str,
        nargs="+",
        choices=["none", "backing", "all"],
        help="SLD fixing modes to test",
    )

    parser.add_argument(
        "--prominent-features",
        type=str,
        nargs="+",
        choices=["true", "false"],
        help="Test with prominent features enabled/disabled",
    )

    parser.add_argument(
        "--inference-backend",
        type=str,
        choices=["predict", "nf"],
        help="Inference backend: predict or nf",
    )

    parser.add_argument(
        "--config-name",
        type=str,
        help="Optional reflectorch config name (overrides backend default)",
    )

    parser.add_argument(
        "--nf-num-samples",
        type=int,
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
    """Main function to run the parameter sweep."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_arguments()

    # Load config from file if provided
    config = {}
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        try:
            config = load_config_from_yaml(args.config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            logger.error("Invalid sweep configuration: %s", error)
            return 1

    # Override with command line arguments
    if args.sweep_name:
        config["sweep_results_dir"] = f"sweep_results_{args.sweep_name}"

    if args.num_experiments is not None:
        config["num_experiments"] = args.num_experiments

    if args.layer_count:
        config["layer_count"] = args.layer_count

    if args.data_directory:
        config["data_directory"] = args.data_directory

    if args.disable_preprocessing:
        config["disable_preprocessing"] = True

    if args.disable_constraints:
        config["disable_constraints"] = True

    if args.experiment_ids:
        config["experiment_ids"] = args.experiment_ids

    if args.prior_deviations:
        config["prior_deviations"] = args.prior_deviations

    if args.priors_type:
        config["priors_type"] = args.priors_type

    if args.sld_modes:
        config["sld_modes"] = args.sld_modes

    if args.prominent_features:
        config["prominent_features"] = [
            p.lower() == "true" for p in args.prominent_features
        ]

    if args.inference_backend:
        config["inference_backend"] = args.inference_backend

    if args.config_name:
        config["config_name"] = args.config_name

    if args.nf_num_samples is not None:
        config["nf_num_samples"] = args.nf_num_samples

    if args.nf_disable_importance_sampling:
        config["nf_disable_importance_sampling"] = True

    if args.use_sigmas_input:
        config["use_sigmas_as_input"] = True

    if args.use_theoretical:
        config["use_theoretical"] = True

    logger.info("Starting automated batch pipeline parameter sweep...")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    sweep = None
    try:
        sweep = BatchPipelineSweep(config=config)
        results = sweep.run_full_sweep()
        if any(not result["success"] for result in results):
            logger.error("\nParameter sweep completed with failed runs")
            return 1
        logger.info("\nParameter sweep completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nParameter sweep interrupted by user")
        if sweep is not None:
            sweep.save_summary()
        return 130

    except (OSError, ValueError, KeyError, RuntimeError) as e:
        logger.exception("\nParameter sweep failed with error: %s", e)
        if sweep is not None:
            sweep.save_summary()
        return 1


if __name__ == "__main__":
    sys.exit(main())
