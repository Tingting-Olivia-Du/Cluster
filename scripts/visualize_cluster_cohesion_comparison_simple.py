import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def load_data(file_path):
    """Load JSON data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_cohesion_data(data, dataset_name):
    """Extract cohesion data from the data"""
    cohesion_data = {
        'positive': [],
        'negative': [],
        'differences': [],
        'qids': []
    }
    
    if dataset_name == 'bb':
        # BB dataset structure
        for pair in data['pair_comparisons']:
            cohesion_data['positive'].append(pair['positive_analysis']['cluster_cohesion'])
            cohesion_data['negative'].append(pair['negative_analysis']['cluster_cohesion'])
            cohesion_data['differences'].append(pair['comparison']['cohesion_difference'])
            cohesion_data['qids'].append(pair['qid'])
    else:
        # BBH and BBEH dataset structure
        for pair in data['pairs']:
            cohesion_data['positive'].append(pair['pos']['cluster_cohesion'])
            cohesion_data['negative'].append(pair['neg']['cluster_cohesion'])
            cohesion_data['differences'].append(pair['diff'])
            cohesion_data['qids'].append(pair['qid'])
    
    return cohesion_data

def create_visualizations():
    """Create visualization charts"""
    
    # Load data
    datasets = {
        'bb': 'output/bb_causal_judgment_analysis/cluster_cohesion_analysis.json',
        'bbh': 'output/bbh_causal_judgment_analysis/cluster_cohesion_analysis.json',
        'bbeh': 'output/bbeh_causal_judgment_analysis/cluster_cohesion_analysis.json'
    }
    
    all_data = {}
    for name, path in datasets.items():
        if Path(path).exists():
            data = load_data(path)
            all_data[name] = extract_cohesion_data(data, name)
        else:
            print(f"Warning: File {path} does not exist")
    
    if not all_data:
        print("Error: No data files found")
        return
    
    # Create charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cluster Cohesion Analysis Comparison (BB vs BBH vs BBEH)', fontsize=16, fontweight='bold')
    
    # Color mapping
    colors = {'bb': '#1f77b4', 'bbh': '#ff7f0e', 'bbeh': '#2ca02c'}
    
    # 1. Box plot comparison
    ax1 = axes[0, 0]
    positive_data = []
    negative_data = []
    labels = []
    
    for name, data in all_data.items():
        positive_data.extend(data['positive'])
        negative_data.extend(data['negative'])
        labels.extend([f'{name}_pos'] * len(data['positive']))
        labels.extend([f'{name}_neg'] * len(data['negative']))
    
    # Create box plot data
    box_data = positive_data + negative_data
    box_labels = labels
    
    bp = ax1.boxplot([box_data[i::6] for i in range(6)], labels=['BB_pos', 'BB_neg', 'BBH_pos', 'BBH_neg', 'BBEH_pos', 'BBEH_neg'])
    ax1.set_title('Cluster Cohesion Distribution Comparison')
    ax1.set_ylabel('Cluster Cohesion Value')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter plot: positive vs negative samples
    ax2 = axes[0, 1]
    for name, data in all_data.items():
        ax2.scatter(data['positive'], data['negative'], 
                   label=name.upper(), color=colors[name], alpha=0.7, s=60)
    
    # Add diagonal line
    max_val = max([max(data['positive'] + data['negative']) for data in all_data.values()])
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    ax2.set_xlabel('Positive Sample Cohesion')
    ax2.set_ylabel('Negative Sample Cohesion')
    ax2.set_title('Positive vs Negative Sample Cohesion')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Difference distribution histogram
    ax3 = axes[0, 2]
    for name, data in all_data.items():
        ax3.hist(data['differences'], alpha=0.6, label=name.upper(), 
                color=colors[name], bins=10, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Difference Line')
    ax3.set_xlabel('Cohesion Difference (Positive - Negative)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Cohesion Difference Distribution')
    ax3.legend()
    
    # 4. Bar chart: average cohesion comparison
    ax4 = axes[1, 0]
    means_pos = [np.mean(data['positive']) for data in all_data.values()]
    means_neg = [np.mean(data['negative']) for data in all_data.values()]
    datasets_names = list(all_data.keys())
    
    x = np.arange(len(datasets_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, means_pos, width, label='Positive Samples', alpha=0.8)
    bars2 = ax4.bar(x + width/2, means_neg, width, label='Negative Samples', alpha=0.8)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax4.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max(means_pos)*0.01,
                f'{means_pos[i]:.2e}', ha='center', va='bottom', fontsize=8)
        ax4.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max(means_neg)*0.01,
                f'{means_neg[i]:.2e}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Average Cohesion')
    ax4.set_title('Average Cohesion Comparison Across Datasets')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.upper() for name in datasets_names])
    ax4.legend()
    
    # 5. Heatmap: difference matrix
    ax5 = axes[1, 1]
    # Calculate statistics for each dataset
    stats_data = []
    stat_labels = []
    
    for name, data in all_data.items():
        stats_data.append([
            np.mean(data['positive']),
            np.mean(data['negative']),
            np.std(data['positive']),
            np.std(data['negative']),
            np.mean(data['differences']),
            len(data['positive'])
        ])
        stat_labels.append(name.upper())
    
    # Normalize data for heatmap
    stats_array = np.array(stats_data)
    stats_normalized = (stats_array - stats_array.min(axis=0)) / (stats_array.max(axis=0) - stats_array.min(axis=0))
    
    im = ax5.imshow(stats_normalized.T, cmap='YlOrRd', aspect='auto')
    ax5.set_xticks(range(len(stat_labels)))
    ax5.set_xticklabels(stat_labels)
    ax5.set_yticks(range(6))
    ax5.set_yticklabels(['Positive Mean', 'Negative Mean', 'Positive Std', 'Negative Std', 'Difference Mean', 'Sample Count'])
    
    # Add value labels
    for i in range(len(stat_labels)):
        for j in range(6):
            text = ax5.text(i, j, f'{stats_array[i, j]:.2e}', ha="center", va="center", color="black", fontsize=8)
    
    ax5.set_title('Dataset Statistics Heatmap')
    
    # 6. Violin plot: distribution shape comparison
    ax6 = axes[1, 2]
    violin_data = []
    violin_labels = []
    
    for name, data in all_data.items():
        violin_data.append(data['positive'])
        violin_data.append(data['negative'])
        violin_labels.append(f'{name.upper()}_pos')
        violin_labels.append(f'{name.upper()}_neg')
    
    parts = ax6.violinplot(violin_data, showmeans=True)
    ax6.set_xticks(range(1, len(violin_labels) + 1))
    ax6.set_xticklabels(violin_labels, rotation=45)
    ax6.set_ylabel('Cluster Cohesion Value')
    ax6.set_title('Cohesion Distribution Shape Comparison')
    
    # Set colors
    for i, pc in enumerate(parts['bodies']):
        if i % 2 == 0:  # Positive samples
            pc.set_facecolor('lightblue')
        else:  # Negative samples
            pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    
    plt.tight_layout()
    
    # Save image
    output_dir = Path('output/visualizations')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'cluster_cohesion_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistical summary
    print("\n=== Cluster Cohesion Analysis Summary ===")
    print(f"{'Dataset':<8} {'Pos Mean':<15} {'Neg Mean':<15} {'Diff Mean':<15} {'Samples':<8}")
    print("-" * 70)
    
    for name, data in all_data.items():
        pos_mean = np.mean(data['positive'])
        neg_mean = np.mean(data['negative'])
        diff_mean = np.mean(data['differences'])
        sample_count = len(data['positive'])
        
        print(f"{name.upper():<8} {pos_mean:<15.2e} {neg_mean:<15.2e} {diff_mean:<15.2e} {sample_count:<8}")

if __name__ == "__main__":
    create_visualizations() 