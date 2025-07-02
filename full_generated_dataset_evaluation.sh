#!/bin/bash

# Full Generated Dataset Evaluation Script
# This script runs the batch inference pipeline for both 1-layer and 2-layer experiments
# from the MARIA dataset to perform a complete evaluation.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "batch_inference_pipeline.py" ]]; then
    print_error "batch_inference_pipeline.py not found in current directory!"
    print_error "Please run this script from the reflectorch api playground directory."
    exit 1
fi

# Check if Pipenv is available
if ! command -v pipenv &> /dev/null; then
    print_error "Pipenv is not installed or not in PATH!"
    print_error "Please install pipenv: pip install pipenv"
    exit 1
fi

# Activate the pipenv environment and check if it's working
print_status "Activating Pipenv environment..."
if ! pipenv --version &> /dev/null; then
    print_error "Failed to activate Pipenv environment!"
    exit 1
fi

print_success "Pipenv environment activated successfully"

# Start the full evaluation
print_status "Starting Full Generated Dataset Evaluation"
print_status "=========================================="

# Log file for the entire evaluation
LOG_FILE="full_evaluation_$(date '+%Y%m%d_%H%M%S').log"
print_status "Logging to: $LOG_FILE"

# Function to run batch inference with error handling
run_batch_inference() {
    local layer_count=$1
    local num_experiments=$2
    local description=$3
    
    print_status "$description"
    print_status "Running batch inference for $num_experiments experiments with $layer_count layer(s)"
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run the pipeline
    if pipenv run python batch_inference_pipeline.py \
        --num-experiments "$num_experiments" \
        --layer-count "$layer_count" \
        --data-directory "data" 2>&1 | tee -a "$LOG_FILE"; then
        
        # Calculate duration
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        print_success "$description completed successfully!"
        print_success "Duration: ${hours}h ${minutes}m ${seconds}s"
        echo "----------------------------------------" | tee -a "$LOG_FILE"
        
        return 0
    else
        print_error "$description failed!"
        echo "----------------------------------------" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Overall start time
OVERALL_START=$(date +%s)

# Variables to track individual phase timings
PHASE1_SUCCESS=false
PHASE2_SUCCESS=false
PHASE1_DURATION=0
PHASE2_DURATION=0

# Run 1-layer experiments (3169 experiments)
print_status ""
print_status "Phase 1: Starting 1-Layer Experiments..."
PHASE1_START=$(date +%s)

if run_batch_inference 1 3169 "Phase 1: 1-Layer Experiments Evaluation"; then
    PHASE1_SUCCESS=true
    print_success "Phase 1: 1-Layer experiments completed successfully!"
else
    print_error "Phase 1: 1-layer experiments failed. Continuing with 2-layer experiments..."
    echo "1-layer experiments FAILED at $(date)" >> "$LOG_FILE"
fi

PHASE1_END=$(date +%s)
PHASE1_DURATION=$((PHASE1_END - PHASE1_START))
PHASE1_HOURS=$((PHASE1_DURATION / 3600))
PHASE1_MINUTES=$(((PHASE1_DURATION % 3600) / 60))
PHASE1_SECONDS=$((PHASE1_DURATION % 60))

print_status "Phase 1 Total Duration: ${PHASE1_HOURS}h ${PHASE1_MINUTES}m ${PHASE1_SECONDS}s"
echo "Phase 1 (1-layer) Duration: ${PHASE1_HOURS}h ${PHASE1_MINUTES}m ${PHASE1_SECONDS}s" >> "$LOG_FILE"

# Small delay between phases
sleep 5

# Run 2-layer experiments (3340 experiments)
print_status ""
print_status "Phase 2: Starting 2-Layer Experiments..."
PHASE2_START=$(date +%s)

if run_batch_inference 2 3340 "Phase 2: 2-Layer Experiments Evaluation"; then
    PHASE2_SUCCESS=true
    print_success "Phase 2: 2-Layer experiments completed successfully!"
else
    print_error "Phase 2: 2-layer experiments failed."
    echo "2-layer experiments FAILED at $(date)" >> "$LOG_FILE"
fi

PHASE2_END=$(date +%s)
PHASE2_DURATION=$((PHASE2_END - PHASE2_START))
PHASE2_HOURS=$((PHASE2_DURATION / 3600))
PHASE2_MINUTES=$(((PHASE2_DURATION % 3600) / 60))
PHASE2_SECONDS=$((PHASE2_DURATION % 60))

print_status "Phase 2 Total Duration: ${PHASE2_HOURS}h ${PHASE2_MINUTES}m ${PHASE2_SECONDS}s"
echo "Phase 2 (2-layer) Duration: ${PHASE2_HOURS}h ${PHASE2_MINUTES}m ${PHASE2_SECONDS}s" >> "$LOG_FILE"

# Calculate total duration
OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Final summary
print_status ""
print_status "=========================================="
print_status "Full Generated Dataset Evaluation Complete"
print_status "=========================================="

# Individual phase results
print_status ""
print_status "PHASE TIMING SUMMARY:"
print_status "--------------------"
if $PHASE1_SUCCESS; then
    print_success "Phase 1 (1-layer, 3169 experiments): SUCCESS - ${PHASE1_HOURS}h ${PHASE1_MINUTES}m ${PHASE1_SECONDS}s"
else
    print_error "Phase 1 (1-layer, 3169 experiments): FAILED - ${PHASE1_HOURS}h ${PHASE1_MINUTES}m ${PHASE1_SECONDS}s"
fi

if $PHASE2_SUCCESS; then
    print_success "Phase 2 (2-layer, 3340 experiments): SUCCESS - ${PHASE2_HOURS}h ${PHASE2_MINUTES}m ${PHASE2_SECONDS}s"
else
    print_error "Phase 2 (2-layer, 3340 experiments): FAILED - ${PHASE2_HOURS}h ${PHASE2_MINUTES}m ${PHASE2_SECONDS}s"
fi

print_status ""
print_success "Total Overall Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"

# Log the summary
echo "" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"
echo "FINAL TIMING SUMMARY" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"
echo "Phase 1 (1-layer): ${PHASE1_HOURS}h ${PHASE1_MINUTES}m ${PHASE1_SECONDS}s - $(if $PHASE1_SUCCESS; then echo 'SUCCESS'; else echo 'FAILED'; fi)" >> "$LOG_FILE"
echo "Phase 2 (2-layer): ${PHASE2_HOURS}h ${PHASE2_MINUTES}m ${PHASE2_SECONDS}s - $(if $PHASE2_SUCCESS; then echo 'SUCCESS'; else echo 'FAILED'; fi)" >> "$LOG_FILE"
echo "Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" >> "$LOG_FILE"

print_status "Results saved in: batch_inference_results/"
print_status "Full log saved in: $LOG_FILE"

# Show summary of result files
if [[ -d "batch_inference_results" ]]; then
    print_status ""
    print_status "Generated result files:"
    ls -la batch_inference_results/*.png 2>/dev/null | tail -10 || print_warning "No new result files found"
fi

print_status ""
print_success "Full evaluation completed! Check the log file and results directory for details."

# Optional: Display system resources at the end
print_status ""
print_status "System status at completion:"
free -h | head -2
df -h . | tail -1