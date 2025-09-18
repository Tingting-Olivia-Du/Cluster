# -*- coding: utf-8 -*-
"""
Mistral Datasets Cluster Cohesion Comparison
比较三个Mistral数据集的聚类内聚度特征：BB, BBH, BBEH
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os

# Set English font and style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION
# ============================================================================

# 三个Mistral数据集配置
# NAME = "lies_mistral-nemo-12b_batch_7"
# NAME = "lies_ministral-8b_batch_5"
NAME = "lies_mistral_batch_4"
DATASETS = {
    'bb': {
        'path': f'./output/bb_{NAME}/cluster_cohesion_analysis.json',
        'display_name': 'BB Lies',
        'color': '#1f77b4'
    },
    'bbh': {
        'path': f'./output/bbh_{NAME}/cluster_cohesion_analysis.json',
        'display_name': 'BBH Lies',
        'color': '#ff7f0e'
    },
    'bbeh': {
        'path': f'./output/bbeh_{NAME}/cluster_cohesion_analysis.json',
        'display_name': 'BBEH Lies',
        'color': '#2ca02c'
    }
}

# 输出目录
OUTPUT_DIR = f'./output/{NAME}_comparison'
VISUALIZATION_DIR = f'{OUTPUT_DIR}/visualizations'

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

print("🚀 Starting Mistral Datasets Cluster Cohesion Comparison...")

# ============================================================================
# DATA LOADING AND EXTRACTION
# ============================================================================

def load_data(file_path):
    """Load JSON data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from {file_path}: {e}")
        return None

def extract_cohesion_data(data, dataset_name):
    """Extract cohesion data from the dataset"""
    if not data or 'pair_comparisons' not in data:
        return None
    
    cohesion_data = {
        'positive': [],
        'negative': [],
        'differences': [],
        'qids': [],
        'valid_pairs': 0,
        'total_pairs': len(data['pair_comparisons'])
    }
    
    for pair in data['pair_comparisons']:
        # 只包含有效的样本对（两个样本都有足够的tokens）
        pos_analysis = pair.get('positive_analysis', {})
        neg_analysis = pair.get('negative_analysis', {})
        
        # Check if both analyses are valid
        if (pos_analysis.get('n_tokens', 0) >= 2 and 
            neg_analysis.get('n_tokens', 0) >= 2 and
            'error' not in pos_analysis and 'error' not in neg_analysis and
            'cluster_cohesion' in pos_analysis and 'cluster_cohesion' in neg_analysis):
            
            cohesion_data['positive'].append(pos_analysis['cluster_cohesion'])
            cohesion_data['negative'].append(neg_analysis['cluster_cohesion'])
            
            # Handle cohesion difference - check both possible locations
            if 'cohesion_difference' in pair:
                cohesion_data['differences'].append(pair['cohesion_difference'])
            elif 'comparison' in pair and 'cohesion_difference' in pair['comparison']:
                cohesion_data['differences'].append(pair['comparison']['cohesion_difference'])
            else:
                # Calculate difference manually if not provided
                diff = pos_analysis['cluster_cohesion'] - neg_analysis['cluster_cohesion']
                cohesion_data['differences'].append(diff)
            
            cohesion_data['qids'].append(pair.get('qid', f'pair_{cohesion_data["valid_pairs"]}'))
            cohesion_data['valid_pairs'] += 1
    
    print(f"  📊 {dataset_name}: Found {cohesion_data['valid_pairs']} valid pairs out of {cohesion_data['total_pairs']} total")
    
    return cohesion_data

def load_all_datasets():
    """Load and extract data from all three datasets"""
    all_data = {}
    
    print("📂 Loading datasets...")
    for name, config in DATASETS.items():
        print(f"🔄 Loading {config['display_name']} from {config['path']}")
        
        if not Path(config['path']).exists():
            print(f"❌ File does not exist: {config['path']}")
            continue
            
        data = load_data(config['path'])
        if data:
            cohesion_data = extract_cohesion_data(data, config['display_name'])
            if cohesion_data and cohesion_data['valid_pairs'] > 0:
                all_data[name] = cohesion_data
                print(f"✅ Loaded {config['display_name']}: {cohesion_data['valid_pairs']}/{cohesion_data['total_pairs']} valid pairs")
            else:
                print(f"⚠️ No valid data found in {config['display_name']}")
        else:
            print(f"❌ Failed to load {config['display_name']}")
    
    return all_data

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(all_data):
    """Compute detailed statistics for all datasets"""
    stats_summary = {}
    
    for name, data in all_data.items():
        if not data['positive'] or not data['negative']:
            continue
            
        pos_cohesions = np.array(data['positive'])
        neg_cohesions = np.array(data['negative'])
        differences = np.array(data['differences'])
        
        stats = {
            'positive': {
                'mean': float(np.mean(pos_cohesions)),
                'std': float(np.std(pos_cohesions)),
                'median': float(np.median(pos_cohesions)),
                'min': float(np.min(pos_cohesions)),
                'max': float(np.max(pos_cohesions)),
                'q25': float(np.percentile(pos_cohesions, 25)),
                'q75': float(np.percentile(pos_cohesions, 75))
            },
            'negative': {
                'mean': float(np.mean(neg_cohesions)),
                'std': float(np.std(neg_cohesions)),
                'median': float(np.median(neg_cohesions)),
                'min': float(np.min(neg_cohesions)),
                'max': float(np.max(neg_cohesions)),
                'q25': float(np.percentile(neg_cohesions, 25)),
                'q75': float(np.percentile(neg_cohesions, 75))
            },
            'differences': {
                'mean': float(np.mean(differences)),
                'std': float(np.std(differences)),
                'median': float(np.median(differences)),
                'positive_count': int(np.sum(differences > 0)),
                'negative_count': int(np.sum(differences < 0)),
                'zero_count': int(np.sum(np.abs(differences) < 1e-10))  # Consider very small values as zero
            },
            'sample_info': {
                'valid_pairs': data['valid_pairs'],
                'total_pairs': data['total_pairs'],
                'validity_rate': data['valid_pairs'] / data['total_pairs'] if data['total_pairs'] > 0 else 0
            }
        }
        
        # Effect size (Cohen's d)
        if len(pos_cohesions) > 1 and len(neg_cohesions) > 1:
            pooled_std = np.sqrt((np.var(pos_cohesions) + np.var(neg_cohesions)) / 2)
            if pooled_std > 0:
                stats['effect_size'] = float((np.mean(pos_cohesions) - np.mean(neg_cohesions)) / pooled_std)
            else:
                stats['effect_size'] = 0.0
        else:
            stats['effect_size'] = 0.0
        
        stats_summary[name] = stats
    
    return stats_summary

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comprehensive_visualizations(all_data):
    """Create comprehensive visualization charts for Mistral datasets comparison"""
    
    if not all_data:
        print("❌ No data available for visualization")
        return
    
    # Create charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mistral Datasets Cluster Cohesion Analysis Comparison\n(BB vs BBH vs BBEH)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Box plot comparison
    ax1 = axes[0, 0]
    box_data = []
    box_labels = []
    box_colors = []
    
    for name, data in all_data.items():
        if data['positive'] and data['negative']:
            box_data.extend([data['positive'], data['negative']])
            display_name = DATASETS[name]['display_name']
            box_labels.extend([f'{display_name}\nPositive', f'{display_name}\nNegative'])
            box_colors.extend([DATASETS[name]['color'], DATASETS[name]['color']])
    
    if box_data:
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Cluster Cohesion Distribution Comparison')
        ax1.set_ylabel('Cluster Cohesion Value')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: positive vs negative samples
    ax2 = axes[0, 1]
    for name, data in all_data.items():
        if data['positive'] and data['negative']:
            display_name = DATASETS[name]['display_name']
            color = DATASETS[name]['color']
            ax2.scatter(data['positive'], data['negative'], 
                       label=display_name, color=color, alpha=0.7, s=60)
    
    # Add diagonal line
    all_values = []
    for data in all_data.values():
        all_values.extend(data['positive'] + data['negative'])
    
    if all_values:
        max_val = max(all_values)
        min_val = min(all_values)
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax2.set_xlabel('Positive Sample Cohesion')
    ax2.set_ylabel('Negative Sample Cohesion')
    ax2.set_title('Positive vs Negative Sample Cohesion')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Difference distribution histogram
    ax3 = axes[0, 2]
    for name, data in all_data.items():
        if data['differences']:
            display_name = DATASETS[name]['display_name']
            color = DATASETS[name]['color']
            ax3.hist(data['differences'], alpha=0.6, label=display_name, 
                    color=color, bins=max(5, len(data['differences'])//2), edgecolor='black')
    
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No Difference')
    ax3.set_xlabel('Cohesion Difference (Positive - Negative)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Cohesion Difference Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bar chart: average cohesion comparison
    ax4 = axes[1, 0]
    datasets_names = []
    means_pos = []
    means_neg = []
    colors = []
    
    for name, data in all_data.items():
        if data['positive'] and data['negative']:
            datasets_names.append(DATASETS[name]['display_name'])
            means_pos.append(np.mean(data['positive']))
            means_neg.append(np.mean(data['negative']))
            colors.append(DATASETS[name]['color'])
    
    if means_pos and means_neg:
        x = np.arange(len(datasets_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, means_pos, width, label='Positive Samples', 
                       color=colors, alpha=0.8, edgecolor='black')
        bars2 = ax4.bar(x + width/2, means_neg, width, label='Negative Samples', 
                       color=colors, alpha=0.5, edgecolor='black')
        
        # Add value labels
        max_val = max(max(means_pos), max(means_neg))
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax4.text(bar1.get_x() + bar1.get_width()/2, height1 + max_val*0.01,
                    f'{means_pos[i]:.5f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax4.text(bar2.get_x() + bar2.get_width()/2, height2 + max_val*0.01,
                    f'{means_neg[i]:.5f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Average Cohesion')
        ax4.set_title('Average Cohesion Comparison Across Mistral Datasets')
        ax4.set_xticks(x)
        ax4.set_xticklabels(datasets_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Violin plot: distribution shape comparison
    ax5 = axes[1, 1]
    violin_data = []
    violin_labels = []
    violin_colors = []
    
    for name, data in all_data.items():
        if data['positive']:
            violin_data.append(data['positive'])
            violin_labels.append(f'{DATASETS[name]["display_name"]}\nPositive')
            violin_colors.append('lightblue')
        if data['negative']:
            violin_data.append(data['negative'])
            violin_labels.append(f'{DATASETS[name]["display_name"]}\nNegative')
            violin_colors.append('lightcoral')
    
    if violin_data:
        parts = ax5.violinplot(violin_data, showmeans=True, showmedians=True)
        ax5.set_xticks(range(1, len(violin_labels) + 1))
        ax5.set_xticklabels(violin_labels, rotation=45)
        ax5.set_ylabel('Cluster Cohesion Value')
        ax5.set_title('Cohesion Distribution Shape Comparison')
        
        # Set colors
        for i, (pc, color) in enumerate(zip(parts['bodies'], violin_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax5.grid(True, alpha=0.3)
    
    # 6. Effect size comparison
    ax6 = axes[1, 2]
    effect_sizes = []
    dataset_labels = []
    bar_colors = []
    
    stats_summary = compute_statistics(all_data)
    
    for name, stats in stats_summary.items():
        effect_sizes.append(stats['effect_size'])
        dataset_labels.append(DATASETS[name]['display_name'])
        bar_colors.append(DATASETS[name]['color'])
    
    if effect_sizes:
        bars = ax6.bar(dataset_labels, effect_sizes, color=bar_colors, alpha=0.8, edgecolor='black')
        ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax6.set_ylabel('Effect Size (Cohen\'s d)')
        ax6.set_title('Effect Size Comparison\n(Positive vs Negative Cohesion)')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, effect_size in zip(bars, effect_sizes):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, height + 0.01 if height >= 0 else height - 0.03,
                    f'{effect_size:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        # Add effect size interpretation
        ax6.text(0.02, 0.98, 'Effect Size:\n|d| < 0.2: negligible\n|d| < 0.5: small\n|d| < 0.8: medium\n|d| ≥ 0.8: large',
                transform=ax6.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save image
    output_path = f"{VISUALIZATION_DIR}/mistral_cluster_cohesion_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive visualization saved to: {output_path}")
    plt.show()
    plt.close()

def print_detailed_statistics(all_data):
    """Print detailed statistical analysis"""
    stats_summary = compute_statistics(all_data)
    
    print("\n" + "="*90)
    print("=== DETAILED MISTRAL DATASETS COMPARISON STATISTICS ===")
    print("="*90)
    
    # Dataset overview
    print(f"\n{'Dataset':<12} {'Valid Pairs':<12} {'Total Pairs':<12} {'Validity Rate':<15}")
    print("-" * 60)
    for name, data in all_data.items():
        display_name = DATASETS[name]['display_name']
        validity_rate = data['valid_pairs'] / data['total_pairs'] if data['total_pairs'] > 0 else 0
        print(f"{display_name:<12} {data['valid_pairs']:<12} {data['total_pairs']:<12} {validity_rate:<15.2%}")
    
    # Cohesion statistics
    print(f"\n{'Dataset':<12} {'Type':<10} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    print("-" * 85)
    
    for name, stats in stats_summary.items():
        display_name = DATASETS[name]['display_name']
        print(f"{display_name:<12} {'Positive':<10} {stats['positive']['mean']:<12.6f} {stats['positive']['std']:<12.6f} "
              f"{stats['positive']['median']:<12.6f} {stats['positive']['min']:<12.6f} {stats['positive']['max']:<12.6f}")
        print(f"{'':<12} {'Negative':<10} {stats['negative']['mean']:<12.6f} {stats['negative']['std']:<12.6f} "
              f"{stats['negative']['median']:<12.6f} {stats['negative']['min']:<12.6f} {stats['negative']['max']:<12.6f}")
        print("-" * 85)
    
    # Effect size and differences
    print(f"\n{'Dataset':<12} {'Mean Diff':<15} {'Effect Size':<12} {'Pos>Neg':<10} {'Pos<Neg':<10} {'Pos≈Neg':<10}")
    print("-" * 80)
    
    for name, stats in stats_summary.items():
        display_name = DATASETS[name]['display_name']
        diff_stats = stats['differences']
        print(f"{display_name:<12} {diff_stats['mean']:<15.8f} {stats['effect_size']:<12.3f} "
              f"{diff_stats['positive_count']:<10} {diff_stats['negative_count']:<10} {diff_stats['zero_count']:<10}")
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY INSIGHTS:")
    print("="*50)
    
    if stats_summary:
        best_pos_cohesion = max(stats_summary.items(), key=lambda x: x[1]['positive']['mean'])
        best_neg_cohesion = max(stats_summary.items(), key=lambda x: x[1]['negative']['mean'])
        largest_effect = max(stats_summary.items(), key=lambda x: abs(x[1]['effect_size']))
        
        print(f"🏆 Highest positive cohesion: {DATASETS[best_pos_cohesion[0]]['display_name']} "
              f"({best_pos_cohesion[1]['positive']['mean']:.6f})")
        print(f"🏆 Highest negative cohesion: {DATASETS[best_neg_cohesion[0]]['display_name']} "
              f"({best_neg_cohesion[1]['negative']['mean']:.6f})")
        print(f"🎯 Largest effect size: {DATASETS[largest_effect[0]]['display_name']} "
              f"(d = {largest_effect[1]['effect_size']:.3f})")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("MISTRAL DATASETS CLUSTER COHESION COMPARISON")
    print("BB Lies vs BBH Lies vs BBEH Lies")
    print("="*80)
    
    # Load all datasets
    all_data = load_all_datasets()
    
    if not all_data:
        print("❌ No valid datasets found. Please check file paths and data format.")
        return
    
    print(f"\n✅ Successfully loaded {len(all_data)} datasets for comparison")
    
    # Save comparison data
    comparison_data = {
        'metadata': {
            'analysis_type': 'mistral_cluster_cohesion_comparison',
            'datasets': [DATASETS[name]['display_name'] for name in all_data.keys()],
            'timestamp': str(np.datetime64('now'))
        },
        'raw_data': all_data,
        'statistics': compute_statistics(all_data)
    }
    
    output_file = f"{OUTPUT_DIR}/mistral_comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Comparison results saved to: {output_file}")
    
    # Create visualizations
    print("\n🎨 Creating comprehensive visualizations...")
    create_comprehensive_visualizations(all_data)
    
    # Print detailed statistics
    print_detailed_statistics(all_data)
    
    print(f"\n📁 All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("✅ Mistral datasets comparison analysis complete!")

if __name__ == "__main__":
    main()