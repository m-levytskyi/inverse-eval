# s000004 Configuration File - Analysis Summary

## Overview
Successfully created configuration files for sample s000004 from the MARIA_VIPR_dataset for use in the ReflecTorch inference pipeline.

## Sample s000004 Characteristics
- **Structure**: Single layer (deposited_layer) on substrate
- **Data Points**: 76 experimental points after trimming
- **Q Range**: 0.0070 - 0.2835 Å⁻¹
- **Data Format**: 4-column (Q, R, dR, dQ)

## Experimental Layer Parameters (from MARIA dataset)
- **Fronting SLD**: 3.5e-06 Å⁻²
- **Fronting Roughness**: 8.04 Å
- **Layer1 SLD**: 3.56709e-06 Å⁻²
- **Layer1 Thickness**: 28.63 Å
- **Layer1 Roughness**: 7.04 Å
- **Backing SLD**: 9.79867e-06 Å⁻²

## Configuration Files Created
1. **configs/s000004_config.json** - JSON format for inference pipeline
2. **configs/s000004_config.yaml** - YAML format for training/analysis

## Model Performance Results
The inference pipeline was successfully tested with 3 models:

| Model | R² | MSE | L1 Loss | Description |
|-------|----|----|---------|-------------|
| neutron_L2_InputQDq | 0.8355 | 0.020041 | 0.039735 | **Best performing** - 2-layer model |
| neutron_L1_InputQDq | 0.8017 | 0.024156 | 0.047493 | 1-layer model with Q,dQ input |
| neutron_L1_comp | 0.8004 | 0.024313 | 0.046994 | Basic 1-layer model |

## Key Insights
1. **Best Model**: `neutron_L2_InputQDq` achieved the lowest MSE (0.020041) despite s000004 being a single-layer structure
2. **Clean Configuration**: Removed duplicate models for cleaner results
3. **Parameter Bounds**: Tuned to experimental values:
   - Thickness: [20.0, 50.0] Å (centered around 28.63 Å)
   - Roughness: [5.0, 15.0] Å (around 7-8 Å experimental values)
   - SLD: [3.0, 11.0] ×10⁻⁶ Å⁻² (covering all layer SLDs)

## Usage
```bash
# Run inference on s000004 data
python inference_pipeline.py configs/s000004_config.json

# Alternative with YAML (for training)
# Use configs/s000004_config.yaml
```

## Files Structure
```
configs/
├── s000004_config.json     # Inference configuration
└── s000004_config.yaml     # Training/analysis configuration

data/s000004/
├── s000004_experimental_curve.dat
├── s000004_model.txt
├── s000004_reflectivity_plot.pdf
├── s000004_sld_profile.dat
├── s000004_sld_profile.pdf
└── s000004_theoretical_curve.dat
```

## Integration
The configuration has been integrated into the inference pipeline with updated help messages showing s000004 as an available option alongside membrane_config.json and s000000_config.json.
