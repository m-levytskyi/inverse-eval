#!/usr/bin/env python3
"""
Comprehensive analysis of the MARIA_VIPR_dataset
Analyzes 10,001 simulated neutron reflectivity experiments with 0-2 layers
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict, Counter
import json

def parse_model_file(filepath):
    """Parse a model.txt file to extract layer information"""
    layers = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # Skip header
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            if len(parts) >= 4:
                layer_name = parts[0]
                sld = float(parts[1]) if parts[1] != 'inf' else np.inf
                thickness = float(parts[2]) if parts[2] != 'inf' else np.inf
                roughness = float(parts[3]) if parts[3] != 'none' else 0.0
                
                layers.append({
                    'name': layer_name,
                    'sld': sld,
                    'thickness': thickness,
                    'roughness': roughness
                })
    
    return layers

def count_active_layers(layers):
    """Count the number of active layers (excluding fronting and backing)"""
    active_layers = [layer for layer in layers 
                    if layer['name'] not in ['fronting', 'backing'] 
                    and layer['thickness'] != np.inf]
    return len(active_layers)

def get_fronting_backing_info(layers):
    """Extract fronting and backing material information"""
    fronting = None
    backing = None
    
    for layer in layers:
        if layer['name'] == 'fronting':
            fronting = layer
        elif layer['name'] == 'backing':
            backing = layer
    
    return fronting, backing

def load_reflectivity_curve(filepath):
    """Load reflectivity curve data"""
    try:
        data = np.loadtxt(filepath)
        if data.shape[1] >= 2:
            q = data[:, 0]
            r = data[:, 1]
            return q, r
        else:
            return None, None
    except:
        return None, None

def analyze_dataset(dataset_path):
    """Comprehensive analysis of the MARIA_VIPR_dataset"""
    
    print("🔍 Analyzing MARIA_VIPR_dataset...")
    print("=" * 60)
    
    # Initialize storage for analysis
    results = {
        'total_experiments': 0,
        'layer_distribution': Counter(),
        'sld_statistics': defaultdict(list),
        'thickness_statistics': defaultdict(list),
        'roughness_statistics': defaultdict(list),
        'q_range_info': {},
        'errors': []
    }
    
    # Get all model files
    model_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('_model.txt')])
    results['total_experiments'] = len(model_files)
    
    print(f"📊 Total experiments found: {results['total_experiments']}")
    
    # Analyze each experiment
    layer_details = []
    q_ranges = []
    
    for i, model_file in enumerate(model_files):
        if i % 1000 == 0:
            print(f"   Processing experiment {i}...")
        
        experiment_id = model_file.replace('_model.txt', '')
        model_path = os.path.join(dataset_path, model_file)
        
        try:
            # Parse model file
            layers = parse_model_file(model_path)
            num_layers = count_active_layers(layers)
            results['layer_distribution'][num_layers] += 1
            
            # Extract layer parameters
            active_layers = [layer for layer in layers 
                           if layer['name'] not in ['fronting', 'backing'] 
                           and layer['thickness'] != np.inf]
            
            # Get fronting and backing information
            fronting, backing = get_fronting_backing_info(layers)
            
            exp_data = {
                'experiment_id': experiment_id,
                'num_layers': num_layers,
                'layers': active_layers,
                'fronting': fronting,
                'backing': backing
            }
            layer_details.append(exp_data)
            
            # Collect statistics for active layers
            for j, layer in enumerate(active_layers):
                layer_pos = f"layer_{j+1}"
                results['sld_statistics'][layer_pos].append(layer['sld'])
                results['thickness_statistics'][layer_pos].append(layer['thickness'])
                results['roughness_statistics'][layer_pos].append(layer['roughness'])
            
            # Check reflectivity curve
            curve_file = os.path.join(dataset_path, f"{experiment_id}_experimental_curve.dat")
            if os.path.exists(curve_file):
                q, r = load_reflectivity_curve(curve_file)
                if q is not None:
                    q_ranges.append({
                        'q_min': np.min(q),
                        'q_max': np.max(q),
                        'n_points': len(q),
                        'experiment_id': experiment_id
                    })
                    
        except Exception as e:
            results['errors'].append(f"Error processing {model_file}: {str(e)}")
    
    # Remove file type counting as it's not needed
    
    # Q-range analysis
    if q_ranges:
        q_df = pd.DataFrame(q_ranges)
        results['q_range_info'] = {
            'q_min_range': [q_df['q_min'].min(), q_df['q_min'].max()],
            'q_max_range': [q_df['q_max'].min(), q_df['q_max'].max()],
            'n_points_range': [q_df['n_points'].min(), q_df['n_points'].max()],
            'avg_n_points': q_df['n_points'].mean()
        }
    
    return results, layer_details, q_ranges

def create_visualizations(results, layer_details, q_ranges):
    """Create comprehensive visualizations of the dataset"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MARIA_VIPR_dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Layer distribution (0-2 layers)
    layer_counts = dict(results['layer_distribution'])
    # Ensure we show layers 0-2 even if some are missing
    display_counts = {i: layer_counts.get(i, 0) for i in range(0, 3)}
    
    axes[0, 0].bar(display_counts.keys(), display_counts.values(), 
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_xlabel('Number of Layers')
    axes[0, 0].set_ylabel('Number of Experiments')
    axes[0, 0].set_title('Distribution of Layer Counts')
    axes[0, 0].set_xticks([0, 1, 2])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add percentage labels
    total = sum(display_counts.values())
    for k, v in display_counts.items():
        if v > 0:  # Only show labels for non-zero counts
            axes[0, 0].text(k, v + total*0.01, f'{v}\n({v/total*100:.1f}%)', 
                           ha='center', va='bottom')
    
    # 2. SLD distribution for different layer positions
    sld_data = []
    for layer_pos, slds in results['sld_statistics'].items():
        for sld in slds:
            sld_data.append({'Layer Position': layer_pos, 'SLD (Å⁻²)': sld})
    
    if sld_data:
        sld_df = pd.DataFrame(sld_data)
        sns.boxplot(data=sld_df, x='Layer Position', y='SLD (Å⁻²)', ax=axes[0, 1])
        axes[0, 1].set_title('SLD Distribution by Layer Position')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Thickness distribution
    thickness_data = []
    for layer_pos, thicknesses in results['thickness_statistics'].items():
        for thickness in thicknesses:
            thickness_data.append({'Layer Position': layer_pos, 'Thickness (Å)': thickness})
    
    if thickness_data:
        thickness_df = pd.DataFrame(thickness_data)
        sns.boxplot(data=thickness_df, x='Layer Position', y='Thickness (Å)', ax=axes[1, 0])
        axes[1, 0].set_title('Thickness Distribution by Layer Position')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Roughness distribution
    roughness_data = []
    for layer_pos, roughnesses in results['roughness_statistics'].items():
        for roughness in roughnesses:
            roughness_data.append({'Layer Position': layer_pos, 'Roughness (Å)': roughness})
    
    if roughness_data:
        roughness_df = pd.DataFrame(roughness_data)
        sns.boxplot(data=roughness_df, x='Layer Position', y='Roughness (Å)', ax=axes[1, 1])
        axes[1, 1].set_title('Roughness Distribution by Layer Position')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/levytskyi/Documents/reflectorch api playground/maria_dataset_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_report(results, layer_details, q_ranges):
    """Print a comprehensive summary report"""
    
    print("\n" + "="*80)
    print("📋 MARIA_VIPR_DATASET ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n🔢 DATASET OVERVIEW:")
    print(f"   • Total experiments: {results['total_experiments']:,}")
    print(f"   • Layer distribution:")
    for layers, count in sorted(results['layer_distribution'].items()):
        percentage = count / results['total_experiments'] * 100
        print(f"     - {layers} layer(s): {count:,} experiments ({percentage:.1f}%)")
    
    print(f"\n📏 PARAMETER RANGES:")
    
    # SLD ranges
    print(f"   • SLD ranges (Å⁻²):")
    for layer_pos, slds in results['sld_statistics'].items():
        if slds:
            print(f"     - {layer_pos}: {np.min(slds):.2e} to {np.max(slds):.2e} "
                  f"(mean: {np.mean(slds):.2e})")
    
    # Thickness ranges
    print(f"   • Thickness ranges (Å):")
    for layer_pos, thicknesses in results['thickness_statistics'].items():
        if thicknesses:
            print(f"     - {layer_pos}: {np.min(thicknesses):.1f} to {np.max(thicknesses):.1f} "
                  f"(mean: {np.mean(thicknesses):.1f})")
    
    # Roughness ranges
    print(f"   • Roughness ranges (Å):")
    for layer_pos, roughnesses in results['roughness_statistics'].items():
        if roughnesses:
            print(f"     - {layer_pos}: {np.min(roughnesses):.1f} to {np.max(roughnesses):.1f} "
                  f"(mean: {np.mean(roughnesses):.1f})")
    
    print(f"\n📈 Q-SPACE INFORMATION:")
    if results['q_range_info']:
        qi = results['q_range_info']
        print(f"   • Q_min range: {qi['q_min_range'][0]:.4f} to {qi['q_min_range'][1]:.4f}")
        print(f"   • Q_max range: {qi['q_max_range'][0]:.4f} to {qi['q_max_range'][1]:.4f}")
        print(f"   • Number of Q points: {qi['n_points_range'][0]} to {qi['n_points_range'][1]} "
              f"(avg: {qi['avg_n_points']:.1f})")
    
    if results['errors']:
        print(f"\n⚠️  ERRORS ENCOUNTERED:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"   • {error}")
        if len(results['errors']) > 5:
            print(f"   ... and {len(results['errors']) - 5} more errors")
    
    print(f"\n✅ Analysis completed successfully!")
    print("="*80)

def save_analysis_results(results, layer_details, output_path):
    """Save detailed analysis results to files"""
    
    # Save summary statistics
    summary = {
        'total_experiments': int(results['total_experiments']),
        'layer_distribution': {str(k): int(v) for k, v in results['layer_distribution'].items()},
        'q_range_info': {k: [float(x) for x in v] if isinstance(v, list) else float(v) 
                        for k, v in results['q_range_info'].items()},
        'parameter_statistics': {
            'sld': {k: {'min': float(np.min(v)), 'max': float(np.max(v)), 
                       'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                   for k, v in results['sld_statistics'].items()},
            'thickness': {k: {'min': float(np.min(v)), 'max': float(np.max(v)), 
                             'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                         for k, v in results['thickness_statistics'].items()},
            'roughness': {k: {'min': float(np.min(v)), 'max': float(np.max(v)), 
                             'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                         for k, v in results['roughness_statistics'].items()}
        }
    }
    
    with open(f"{output_path}/maria_dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed layer information
    layer_df = []
    for exp in layer_details:
        if exp['num_layers'] == 0:
            # For 0-layer experiments, add a row with fronting and backing info
            layer_df.append({
                'experiment_id': exp['experiment_id'],
                'num_layers': exp['num_layers'],
                'layer_position': 0,  # Special case for 0 layers
                'layer_type': 'substrate_interface',
                'fronting_sld': exp['fronting']['sld'] if exp['fronting'] else None,
                'fronting_roughness': exp['fronting']['roughness'] if exp['fronting'] else None,
                'backing_sld': exp['backing']['sld'] if exp['backing'] else None,
                'backing_roughness': exp['backing']['roughness'] if exp['backing'] else None,
                'sld': None,  # No deposited layer
                'thickness': None,  # No deposited layer
                'roughness': None  # No deposited layer
            })
        else:
            # For experiments with layers, add one row per layer
            for i, layer in enumerate(exp['layers']):
                layer_df.append({
                    'experiment_id': exp['experiment_id'],
                    'num_layers': exp['num_layers'],
                    'layer_position': i + 1,
                    'layer_type': 'deposited_layer',
                    'fronting_sld': exp['fronting']['sld'] if exp['fronting'] else None,
                    'fronting_roughness': exp['fronting']['roughness'] if exp['fronting'] else None,
                    'backing_sld': exp['backing']['sld'] if exp['backing'] else None,
                    'backing_roughness': exp['backing']['roughness'] if exp['backing'] else None,
                    'sld': layer['sld'],
                    'thickness': layer['thickness'],
                    'roughness': layer['roughness']
                })
    
    if layer_df:
        pd.DataFrame(layer_df).to_csv(f"{output_path}/maria_dataset_layers.csv", index=False)
    
    print(f"\n💾 Analysis results saved to:")
    print(f"   • {output_path}/maria_dataset_summary.json")
    print(f"   • {output_path}/maria_dataset_layers.csv")
    print(f"   • {output_path}/maria_dataset_analysis.png")

def main():
    """Main analysis function"""
    dataset_path = "/home/levytskyi/Documents/reflectorch api playground/data/MARIA_VIPR_dataset"
    output_path = "/home/levytskyi/Documents/reflectorch api playground"
    
    # Run comprehensive analysis
    results, layer_details, q_ranges = analyze_dataset(dataset_path)
    
    # Generate visualizations
    create_visualizations(results, layer_details, q_ranges)
    
    # Print summary report
    print_summary_report(results, layer_details, q_ranges)
    
    # Save results
    save_analysis_results(results, layer_details, output_path)

if __name__ == "__main__":
    main()
