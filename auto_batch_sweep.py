#!/usr/bin/env python3
"""
Automated batch pipeline parameter sweep script.

This script runs the batch pipeline with different parameter combinations:
- Prior deviations: 5%, 30%, 99%
- SLD fixing modes: none, fronting_backing, all
- With and without prominent features
- Results stored in dedicated sweep_results folder
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Parameter combinations to test
PRIOR_DEVIATIONS = [5, 30, 99]
SLD_MODES = ["none", "fronting_backing", "all"]
PROMINENT_FEATURES = [False, True]

# Experiment configuration
NUM_EXPERIMENTS = None  # None means use all available experiments
LAYER_COUNT = 1
DATA_DIRECTORY = "data"

# Output configuration
SWEEP_RESULTS_DIR = "sweep_results"

# =============================================================================


class BatchPipelineSweep:
    """Automated parameter sweep for batch pipeline."""
    
    def __init__(self, num_experiments=NUM_EXPERIMENTS, layer_count=LAYER_COUNT, 
                 data_directory=DATA_DIRECTORY):
        """Initialize the sweep configuration."""
        self.num_experiments = num_experiments
        self.layer_count = layer_count
        self.data_directory = data_directory
        
        # Discover total available experiments for this layer count
        if self.num_experiments is None:
            from parameter_discovery import discover_batch_experiments
            all_experiments = discover_batch_experiments(
                data_directory=self.data_directory,
                layer_count=self.layer_count,
                num_experiments=10000,  # Large number to get all
                experiment_ids=None
            )
            self.num_experiments = len(all_experiments)
            print(f"📊 Discovered {self.num_experiments} experiments with {self.layer_count} layer(s)")
        
        # Create timestamped sweep directory
        timestamp = datetime.now().strftime("%d%B%Y_%H_%M").lower()
        exp_desc = f"ALL{self.num_experiments}" if self.num_experiments > 100 else str(self.num_experiments)
        self.sweep_dir = Path(SWEEP_RESULTS_DIR) / f"sweep_{exp_desc}exps_{self.layer_count}layers_{timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.results_summary = []
        self.total_runs = len(PRIOR_DEVIATIONS) * len(SLD_MODES) * len(PROMINENT_FEATURES)
        self.current_run = 0
        
        print(f"🚀 BATCH PIPELINE PARAMETER SWEEP")
        print(f"=" * 60)
        print(f"Sweep directory: {self.sweep_dir}")
        print(f"Total parameter combinations: {self.total_runs}")
        print(f"Experiments per run: {self.num_experiments}")
        print(f"Layer count: {self.layer_count}")
        print(f"Data directory: {self.data_directory}")
        print(f"=" * 60)
    
    def run_single_configuration(self, prior_deviation, sld_mode, use_prominent):
        """Run batch pipeline with specific parameter configuration."""
        self.current_run += 1
        
        print(f"\n{'='*80}")
        print(f"RUN {self.current_run}/{self.total_runs}")
        print(f"Prior deviation: {prior_deviation}%")
        print(f"SLD mode: {sld_mode}")
        print(f"Prominent features: {'enabled' if use_prominent else 'disabled'}")
        print(f"{'='*80}")
        
        # Build command
        cmd = [
            sys.executable, "batch_pipeline.py",
            "--num-experiments", str(self.num_experiments),
            "--layer-count", str(self.layer_count),
            "--data-directory", self.data_directory,
            "--priors-deviation", str(prior_deviation),
            "--fix-sld-mode", sld_mode
        ]
        
        if use_prominent:
            cmd.append("--use-prominent-features")
        
        print(f"Command: {' '.join(cmd)}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the batch pipeline with extended timeout for all experiments
            timeout_hours = 6 if self.num_experiments > 100 else 1
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_hours*3600)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Parse result
            success = result.returncode == 0
            
            if success:
                print(f"✅ Configuration completed successfully in {execution_time:.1f} seconds")
            else:
                print(f"❌ Configuration failed after {execution_time:.1f} seconds")
                print(f"Error output: {result.stderr[:500]}...")
            
            # Record results
            run_summary = {
                "run_number": self.current_run,
                "prior_deviation": prior_deviation,
                "sld_mode": sld_mode,
                "use_prominent_features": use_prominent,
                "success": success,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "timestamp": datetime.now().isoformat()
            }
            
            if not success:
                run_summary["error_output"] = result.stderr[-1000:]  # Last 1000 chars
                run_summary["stdout_output"] = result.stdout[-1000:]  # Last 1000 chars
            
            self.results_summary.append(run_summary)
            
            # Save intermediate results
            self.save_summary()
            
            return success
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            execution_time = end_time - start_time
            
            timeout_hours = 6 if self.num_experiments > 100 else 1
            print(f"⏰ Configuration timed out after {execution_time:.1f} seconds ({timeout_hours} hour limit)")
            
            # Record timeout
            run_summary = {
                "run_number": self.current_run,
                "prior_deviation": prior_deviation,
                "sld_mode": sld_mode,
                "use_prominent_features": use_prominent,
                "success": False,
                "execution_time": execution_time,
                "return_code": "TIMEOUT",
                "error_output": f"Process timed out after {timeout_hours} hours",
                "timestamp": datetime.now().isoformat()
            }
            
            self.results_summary.append(run_summary)
            self.save_summary()
            
            return False
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"💥 Configuration failed with exception: {e}")
            
            # Record exception
            run_summary = {
                "run_number": self.current_run,
                "prior_deviation": prior_deviation,
                "sld_mode": sld_mode,
                "use_prominent_features": use_prominent,
                "success": False,
                "execution_time": execution_time,
                "return_code": "EXCEPTION",
                "error_output": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.results_summary.append(run_summary)
            self.save_summary()
            
            return False
    
    def save_summary(self):
        """Save current sweep summary to file."""
        summary_file = self.sweep_dir / "sweep_summary.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results_summary, f, indent=2)
    
    def run_full_sweep(self):
        """Run the complete parameter sweep."""
        print(f"\n🎯 STARTING FULL PARAMETER SWEEP")
        
        # Estimate duration based on number of experiments
        if self.num_experiments <= 10:
            estimated_min = self.total_runs * 5
            estimated_max = self.total_runs * 15
        elif self.num_experiments <= 100:
            estimated_min = self.total_runs * 15
            estimated_max = self.total_runs * 45
        else:
            estimated_min = self.total_runs * 60
            estimated_max = self.total_runs * 180
            
        print(f"Estimated duration: {estimated_min} - {estimated_max} minutes ({estimated_min/60:.1f} - {estimated_max/60:.1f} hours)")
        
        sweep_start_time = time.time()
        successful_runs = 0
        failed_runs = 0
        
        # Iterate through all parameter combinations
        for prior_deviation in PRIOR_DEVIATIONS:
            for sld_mode in SLD_MODES:
                for use_prominent in PROMINENT_FEATURES:
                    
                    success = self.run_single_configuration(
                        prior_deviation=prior_deviation,
                        sld_mode=sld_mode,
                        use_prominent=use_prominent
                    )
                    
                    if success:
                        successful_runs += 1
                    else:
                        failed_runs += 1
                    
                    # Brief pause between runs
                    time.sleep(2)
        
        sweep_end_time = time.time()
        total_sweep_time = sweep_end_time - sweep_start_time
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {total_sweep_time:.1f} seconds ({total_sweep_time/60:.1f} minutes)")
        print(f"Successful runs: {successful_runs}/{self.total_runs}")
        print(f"Failed runs: {failed_runs}/{self.total_runs}")
        print(f"Success rate: {successful_runs/self.total_runs*100:.1f}%")
        print(f"Average time per run: {total_sweep_time/self.total_runs:.1f} seconds")
        print(f"Results saved to: {self.sweep_dir}")
        
        # Create detailed summary report
        self.create_final_report(total_sweep_time, successful_runs, failed_runs)
        
        return self.results_summary
    
    def create_final_report(self, total_time, successful_runs, failed_runs):
        """Create a comprehensive final report."""
        report_file = self.sweep_dir / "sweep_report.md"
        
        report_content = f"""# Batch Pipeline Parameter Sweep Report

**Sweep Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Sweep Directory:** `{self.sweep_dir}`

## Configuration
- **Experiments per run:** {self.num_experiments}
- **Layer count:** {self.layer_count}
- **Data directory:** {self.data_directory}

## Parameter Combinations Tested
- **Prior deviations:** {PRIOR_DEVIATIONS}% 
- **SLD fixing modes:** {SLD_MODES}
- **Prominent features:** {PROMINENT_FEATURES}

## Overall Results
- **Total runs:** {self.total_runs}
- **Successful runs:** {successful_runs}
- **Failed runs:** {failed_runs}
- **Success rate:** {successful_runs/self.total_runs*100:.1f}%
- **Total execution time:** {total_time:.1f} seconds ({total_time/60:.1f} minutes)
- **Average time per run:** {total_time/self.total_runs:.1f} seconds

## Detailed Results

| Run | Prior % | SLD Mode | Prominent | Success | Time (s) | Notes |
|-----|---------|----------|-----------|---------|----------|-------|
"""
        
        for result in self.results_summary:
            status = "✅" if result["success"] else "❌"
            prominent = "Yes" if result["use_prominent_features"] else "No"
            time_str = f"{result['execution_time']:.1f}"
            notes = ""
            
            if not result["success"]:
                if result["return_code"] == "TIMEOUT":
                    notes = "Timeout"
                elif result["return_code"] == "EXCEPTION":
                    notes = "Exception"
                else:
                    notes = f"RC:{result['return_code']}"
            
            report_content += f"| {result['run_number']} | {result['prior_deviation']}% | {result['sld_mode']} | {prominent} | {status} | {time_str} | {notes} |\n"
        
        report_content += f"""

## Failed Runs Analysis
"""
        
        failed_results = [r for r in self.results_summary if not r["success"]]
        if failed_results:
            report_content += f"**Total failed runs:** {len(failed_results)}\n\n"
            
            for result in failed_results:
                report_content += f"""
### Run {result['run_number']} - {result['prior_deviation']}% / {result['sld_mode']} / {'Prominent' if result['use_prominent_features'] else 'Standard'}
- **Return code:** {result['return_code']}
- **Error:** {result.get('error_output', 'No error output')[:200]}...
"""
        else:
            report_content += "No failed runs! 🎉\n"
        
        report_content += f"""

## Next Steps
1. Review individual batch results in `batch_inference_results/` directory
2. Compare performance across different parameter combinations
3. Analyze successful vs failed configurations
4. Consider running additional experiments with optimal parameters

---
*Generated by auto_batch_sweep.py on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📊 Detailed report saved to: {report_file}")


def main():
    """Main function to run the parameter sweep."""
    print("Starting automated batch pipeline parameter sweep...")
    
    # Create and run the sweep
    sweep = BatchPipelineSweep()
    
    try:
        results = sweep.run_full_sweep()
        print("\n🎉 Parameter sweep completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Parameter sweep interrupted by user")
        sweep.save_summary()
        return 1
        
    except Exception as e:
        print(f"\n💥 Parameter sweep failed with error: {e}")
        sweep.save_summary()
        return 1


if __name__ == "__main__":
    sys.exit(main())