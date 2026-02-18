"""
Centralized configuration for reflectorch API playground.

This module contains all common paths, default parameters, and configuration
settings used across the project.
"""

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

DATA_DIRECTORY = "dataset/test"
BATCH_RESULTS_DIR = "batch_inference_results"
PAPER_BATCHES_DIR = "paper_batches"
SWEEP_RESULTS_DIR = "sweep_results"

# =============================================================================
# NF (NORMALIZING FLOW) DEFAULTS
# =============================================================================

DEFAULT_NF_CONFIG = "example_nf_config_reflectorch.yaml"
DEFAULT_NF_SAMPLES = 1000
DEFAULT_NF_DISABLE_IMPORTANCE_SAMPLING = False

# =============================================================================
# INFERENCE DEFAULTS
# =============================================================================

DEFAULT_PRIORS_TYPE = "constraint_based"
DEFAULT_PRIORS_DEVIATION = 0.30  # 30% constraint span
DEFAULT_SLD_MODE = "none"
DEFAULT_INFERENCE_BACKEND = "nf"

# =============================================================================
# BATCH PROCESSING DEFAULTS
# =============================================================================

DEFAULT_LAYER_COUNT = 1
DEFAULT_NUM_EXPERIMENTS = None  # None means all available

# =============================================================================
# PLOTTING DEFAULTS
# =============================================================================

DEFAULT_PLOT_STYLE = "paper.mplstyle"
DEFAULT_DPI = 300
DEFAULT_FIGURE_SIZE = (10, 6)

# =============================================================================
# TIMEOUT SETTINGS
# =============================================================================

BATCH_TIMEOUT_SECONDS = 7200  # 2 hours per batch
EXPERIMENT_TIMEOUT_SECONDS = 300  # 5 minutes per experiment

# =============================================================================
# PROMINENT FEATURES SETTINGS
# =============================================================================

PROMINENT_PEAK_THRESHOLD = 0.1  # Minimum prominence for peak detection
PROMINENT_MIN_PEAKS = 1  # Minimum number of peaks required

# =============================================================================
# EDGE CASE DETECTION THRESHOLDS
# =============================================================================

EDGE_CASE_MAPE_THRESHOLD = 50.0  # MAPE threshold for edge case detection
HIGH_ERROR_THICKNESS_THRESHOLD = 100.0  # Thickness MAPE threshold
HIGH_ERROR_ROUGHNESS_THRESHOLD = 100.0  # Roughness MAPE threshold
HIGH_ERROR_SLD_THRESHOLD = 100.0  # SLD MAPE threshold

# =============================================================================
# MODEL CONSTRAINTS FILE
# =============================================================================

MODEL_CONSTRAINTS_FILE = "model_constraints.json"
