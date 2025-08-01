# -*- coding: utf-8 -*-
"""
Causal Judgment Cluster Cohesion Analysis (Sample-wise)
分析因果判断任务中正负样本的聚类内聚度特征 - 样本级别分析
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# 本地相对路径配置
INPUT_PATH = "./logits/bb_causal_judgment_experiment_26-50.json"
OUTPUT_DIR = "./output/bb_26-50_causal_judgment_analysis"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualizations"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

print("🚀 Starting Causal Judgment Cluster Cohesion Analysis (Sample-wise)...")
print(f"📂 Input path: {INPUT_PATH}")
print(f"📁 Output directory: {OUTPUT_DIR}")

# ============================================================================
# DATA LOADING AND HELPER FUNCTIONS
# ============================================================================

def load_causal_judgment_data(file_path):
    """加载因果判断实验数据"""
    print(f"🔄 Attempting to load data from: {os.path.abspath(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("📁 Current working directory:", os.getcwd())
        print("📂 Available files in current directory:")
        try:
            for item in os.listdir("."):
                if os.path.isfile(item):
                    print(f"   📄 {item}")
                elif os.path.isdir(item):
                    print(f"   📁 {item}/")
        except:
            pass
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded data with {len(data.get('results', {}))} questions")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def compute_cluster_cohesion(vectors):
    """
    计算向量集合的聚类内聚度
    
    Args:
        vectors: list of hidden vectors
        
    Returns:
        dict: cluster cohesion metrics
    """
    if not vectors or len(vectors) < 2:
        return {
            'cluster_cohesion': 0.0,
            'optimal_clusters': 1,
            'min_inertia': 0.0,
            'best_silhouette_score': 0.0,
            'n_tokens': len(vectors) if vectors else 0
        }
    
    vectors = np.array(vectors)
    n_tokens = len(vectors)
    
    # 计算质心
    centroid = np.mean(vectors, axis=0)
    
    # 基础度量
    centroid_distances = [np.linalg.norm(vec - centroid) for vec in vectors]
    
    # K-means聚类分析
    max_clusters = min(5, n_tokens)
    inertias = []
    silhouette_scores = []
    
    for k in range(1, max_clusters + 1):
        if k == 1:
            # 单个聚类的惯性
            inertia = np.sum([np.linalg.norm(vec - centroid) ** 2 for vec in vectors])
            inertias.append(inertia)
            silhouette_scores.append(0.0)
        else:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors)
                inertias.append(kmeans.inertia_)
                
                # 计算轮廓系数
                if len(set(labels)) > 1:
                    sil_score = silhouette_score(vectors, labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0.0)
            except:
                inertias.append(inertias[-1] if inertias else 0.0)
                silhouette_scores.append(0.0)
    
    # 最佳指标
    best_silhouette_score = float(np.max(silhouette_scores))
    optimal_clusters = int(np.argmax(silhouette_scores) + 1)
    min_inertia = float(np.min(inertias))
    
    # 聚类内聚度 = 1 / (1 + 最小惯性)
    cluster_cohesion = float(1 / (1 + min_inertia))
    
    return {
        'cluster_cohesion': cluster_cohesion,
        'optimal_clusters': optimal_clusters,
        'min_inertia': min_inertia,
        'best_silhouette_score': best_silhouette_score,
        'n_tokens': n_tokens,
        'centroid_distance_mean': float(np.mean(centroid_distances)),
        'centroid_distance_std': float(np.std(centroid_distances))
    }

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def extract_positive_negative_pairs(data):
    """提取正负样本对"""
    results = data.get('results', {})
    pairs = []
    
    for qid, question_data in results.items():
        samplings = question_data.get('samplings', {})
        true_answer = question_data.get('true_final_result', '')
        
        # 找到正样本和负样本
        positive_sample = None
        negative_sample = None
        
        for sid, sample in samplings.items():
            extracted_answer = sample.get('extracted_answer', '')
            is_correct = sample.get('is_correct', False)
            
            # 检查是否有token_details
            if 'token_details' not in sample or not sample['token_details']:
                continue
            
            if is_correct and positive_sample is None:
                positive_sample = (sid, sample)
            elif not is_correct and negative_sample is None:
                negative_sample = (sid, sample)
        
        # 如果找到了配对的正负样本
        if positive_sample and negative_sample:
            pairs.append({
                'qid': qid,
                'true_answer': true_answer,
                'positive': {
                    'sid': positive_sample[0],
                    'sample': positive_sample[1]
                },
                'negative': {
                    'sid': negative_sample[0],
                    'sample': negative_sample[1]
                }
            })
    
    print(f"✅ Found {len(pairs)} positive-negative pairs")
    return pairs

def analyze_sample_cluster_cohesion(sample, sample_type):
    """分析单个样本的整体聚类内聚度 (sample-wise)"""
    
    token_details = sample.get('token_details', [])
    if not token_details:
        return {
            'cluster_cohesion': 0.0,
            'optimal_clusters': 1,
            'min_inertia': 0.0,
            'best_silhouette_score': 0.0,
            'n_tokens': 0,
            'sample_type': sample_type
        }
    
    # 提取所有tokens的向量
    sample_vectors = []
    for token in token_details:
        if 'hidden_vector' in token:
            sample_vectors.append(token['hidden_vector'])
    
    if len(sample_vectors) < 2:
        return {
            'cluster_cohesion': 0.0,
            'optimal_clusters': 1,
            'min_inertia': 0.0,
            'best_silhouette_score': 0.0,
            'n_tokens': len(sample_vectors),
            'sample_type': sample_type
        }
    
    # 计算整个样本的聚类内聚度
    cohesion_metrics = compute_cluster_cohesion(sample_vectors)
    cohesion_metrics['sample_type'] = sample_type
    
    return cohesion_metrics

def analyze_cluster_cohesion_for_pairs(pairs):
    """分析正负样本对的聚类内聚度 (sample-wise)"""
    
    analysis_results = {
        'pair_comparisons': [],
        'summary_statistics': {
            'total_pairs': len(pairs),
            'positive_stats': {
                'cluster_cohesion': [],
                'optimal_clusters': [],
                'best_silhouette_score': [],
                'n_tokens': []
            },
            'negative_stats': {
                'cluster_cohesion': [],
                'optimal_clusters': [],
                'best_silhouette_score': [],
                'n_tokens': []
            }
        }
    }
    
    for pair_idx, pair in enumerate(pairs):
        print(f"🔄 Processing pair {pair_idx + 1}/{len(pairs)}: {pair['qid']}")
        
        # 分析正样本和负样本
        positive_analysis = analyze_sample_cluster_cohesion(pair['positive']['sample'], 'positive')
        negative_analysis = analyze_sample_cluster_cohesion(pair['negative']['sample'], 'negative')
        
        pair_result = {
            'qid': pair['qid'],
            'true_answer': pair['true_answer'],
            'positive_analysis': positive_analysis,
            'negative_analysis': negative_analysis,
            'comparison': {
                'cohesion_difference': positive_analysis['cluster_cohesion'] - negative_analysis['cluster_cohesion'],
                'positive_cohesion': positive_analysis['cluster_cohesion'],
                'negative_cohesion': negative_analysis['cluster_cohesion'],
                'positive_tokens': positive_analysis['n_tokens'],
                'negative_tokens': negative_analysis['n_tokens']
            }
        }
        
        # 添加到统计数据
        if positive_analysis['n_tokens'] >= 2:
            analysis_results['summary_statistics']['positive_stats']['cluster_cohesion'].append(positive_analysis['cluster_cohesion'])
            analysis_results['summary_statistics']['positive_stats']['optimal_clusters'].append(positive_analysis['optimal_clusters'])
            analysis_results['summary_statistics']['positive_stats']['best_silhouette_score'].append(positive_analysis['best_silhouette_score'])
            analysis_results['summary_statistics']['positive_stats']['n_tokens'].append(positive_analysis['n_tokens'])
        
        if negative_analysis['n_tokens'] >= 2:
            analysis_results['summary_statistics']['negative_stats']['cluster_cohesion'].append(negative_analysis['cluster_cohesion'])
            analysis_results['summary_statistics']['negative_stats']['optimal_clusters'].append(negative_analysis['optimal_clusters'])
            analysis_results['summary_statistics']['negative_stats']['best_silhouette_score'].append(negative_analysis['best_silhouette_score'])
            analysis_results['summary_statistics']['negative_stats']['n_tokens'].append(negative_analysis['n_tokens'])
        
        analysis_results['pair_comparisons'].append(pair_result)
    
    # 计算总体统计
    pos_cohesions = analysis_results['summary_statistics']['positive_stats']['cluster_cohesion']
    neg_cohesions = analysis_results['summary_statistics']['negative_stats']['cluster_cohesion']
    
    if pos_cohesions and neg_cohesions:
        analysis_results['overall_comparison'] = {
            'positive_mean': float(np.mean(pos_cohesions)),
            'positive_std': float(np.std(pos_cohesions)),
            'negative_mean': float(np.mean(neg_cohesions)),
            'negative_std': float(np.std(neg_cohesions)),
            'mean_difference': float(np.mean(pos_cohesions) - np.mean(neg_cohesions)),
            'effect_size': float((np.mean(pos_cohesions) - np.mean(neg_cohesions)) / 
                                np.sqrt((np.var(pos_cohesions) + np.var(neg_cohesions)) / 2))
        }
    
    return analysis_results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(analysis_results):
    """创建可视化图表"""
    
    print("📊 Creating visualizations...")
    
    # 1. 总体对比箱型图
    create_overall_comparison_plot(analysis_results)
    
    # 2. 逐对比较图
    create_pairwise_comparison_plot(analysis_results)
    
    # 3. 分布对比图
    create_distribution_comparison_plot(analysis_results)
    
    # 4. 样本级别分析图
    create_sample_level_analysis_plot(analysis_results)

def create_overall_comparison_plot(analysis_results):
    """创建总体对比箱型图"""
    
    pos_cohesions = analysis_results['summary_statistics']['positive_stats']['cluster_cohesion']
    neg_cohesions = analysis_results['summary_statistics']['negative_stats']['cluster_cohesion']
    
    if not pos_cohesions or not neg_cohesions:
        return
    
    plt.figure(figsize=(10, 6))
    
    data_to_plot = [pos_cohesions, neg_cohesions]
    labels = ['Positive Samples\n(Correct Answers)', 'Negative Samples\n(Incorrect Answers)']
    colors = ['lightgreen', 'lightcoral']
    
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Sample-wise Cluster Cohesion: Positive vs Negative Samples', fontsize=14, fontweight='bold')
    plt.ylabel('Cluster Cohesion Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    if 'overall_comparison' in analysis_results:
        stats = analysis_results['overall_comparison']
        plt.figtext(0.02, 0.02, 
                   f"Positive: μ={stats['positive_mean']:.3f}±{stats['positive_std']:.3f}\n"
                   f"Negative: μ={stats['negative_mean']:.3f}±{stats['negative_std']:.3f}\n"
                   f"Difference: {stats['mean_difference']:.3f}\n"
                   f"Effect Size: {stats['effect_size']:.3f}",
                   fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/overall_cluster_cohesion_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.show()  # 显示图表
    plt.close()
    
    print(f"✅ Overall comparison plot saved to: {os.path.abspath(VISUALIZATION_DIR)}/overall_cluster_cohesion_comparison.png")

def create_pairwise_comparison_plot(analysis_results):
    """创建逐对比较图 (sample-wise)"""
    
    pairs = analysis_results['pair_comparisons']
    if not pairs:
        return
    
    # 提取每对样本的聚类内聚度
    positive_cohesions = []
    negative_cohesions = []
    qids = []
    
    for pair in pairs:
        pos_analysis = pair['positive_analysis']
        neg_analysis = pair['negative_analysis']
        
        if pos_analysis['n_tokens'] >= 2 and neg_analysis['n_tokens'] >= 2:
            positive_cohesions.append(pos_analysis['cluster_cohesion'])
            negative_cohesions.append(neg_analysis['cluster_cohesion'])
            qids.append(pair['qid'])
    
    if not positive_cohesions:
        return
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(qids))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, positive_cohesions, width, label='Positive (Correct)', 
                    color='lightgreen', alpha=0.8)
    bars2 = plt.bar(x + width/2, negative_cohesions, width, label='Negative (Incorrect)', 
                    color='lightcoral', alpha=0.8)
    
    plt.xlabel('Question ID')
    plt.ylabel('Sample Cluster Cohesion')
    plt.title('Sample-wise Cluster Cohesion Comparison')
    plt.xticks(x, [f"Q{i+1}" for i in range(len(qids))], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/pairwise_cluster_cohesion_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.show()  # 显示图表
    plt.close()
    
    print(f"✅ Pairwise comparison plot saved to: {os.path.abspath(VISUALIZATION_DIR)}/pairwise_cluster_cohesion_comparison.png")

def create_distribution_comparison_plot(analysis_results):
    """创建分布对比图"""
    
    pos_cohesions = analysis_results['summary_statistics']['positive_stats']['cluster_cohesion']
    neg_cohesions = analysis_results['summary_statistics']['negative_stats']['cluster_cohesion']
    
    if not pos_cohesions or not neg_cohesions:
        return
    
    plt.figure(figsize=(12, 6))
    
    # 子图1: 直方图
    plt.subplot(1, 2, 1)
    plt.hist(pos_cohesions, bins=20, alpha=0.7, label='Positive', color='lightgreen', density=True)
    plt.hist(neg_cohesions, bins=20, alpha=0.7, label='Negative', color='lightcoral', density=True)
    plt.xlabel('Cluster Cohesion Score')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 累积分布
    plt.subplot(1, 2, 2)
    
    pos_sorted = np.sort(pos_cohesions)
    neg_sorted = np.sort(neg_cohesions)
    
    pos_y = np.arange(1, len(pos_sorted) + 1) / len(pos_sorted)
    neg_y = np.arange(1, len(neg_sorted) + 1) / len(neg_sorted)
    
    plt.plot(pos_sorted, pos_y, label='Positive', color='green', linewidth=2)
    plt.plot(neg_sorted, neg_y, label='Negative', color='red', linewidth=2)
    
    plt.xlabel('Cluster Cohesion Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/distribution_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.show()  # 显示图表
    plt.close()
    
    print(f"✅ Distribution comparison plot saved to: {os.path.abspath(VISUALIZATION_DIR)}/distribution_comparison.png")

def create_sample_level_analysis_plot(analysis_results):
    """创建样本级别详细分析图 (sample-wise)"""
    
    pairs = analysis_results['pair_comparisons']
    if not pairs or len(pairs) < 3:
        return
    
    # 选择前几个对进行详细分析
    selected_pairs = pairs[:min(3, len(pairs))]
    
    fig, axes = plt.subplots(len(selected_pairs), 1, figsize=(12, 4 * len(selected_pairs)))
    if len(selected_pairs) == 1:
        axes = [axes]
    
    for idx, pair in enumerate(selected_pairs):
        ax = axes[idx]
        
        pos_analysis = pair['positive_analysis']
        neg_analysis = pair['negative_analysis']
        
        # 显示样本级别的聚类信息
        categories = ['Cluster\nCohesion', 'Optimal\nClusters', 'Best Silhouette\nScore']
        
        pos_values = [
            pos_analysis['cluster_cohesion'],
            pos_analysis['optimal_clusters'] / 10.0,  # 标准化显示
            pos_analysis['best_silhouette_score']
        ]
        
        neg_values = [
            neg_analysis['cluster_cohesion'],
            neg_analysis['optimal_clusters'] / 10.0,  # 标准化显示
            neg_analysis['best_silhouette_score']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pos_values, width, label='Positive', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, neg_values, width, label='Negative', color='red', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Normalized Values')
        ax.set_title(f'Sample Analysis: {pair["qid"]} (Answer: {pair["true_answer"]})\n'
                    f'Tokens: Pos={pos_analysis["n_tokens"]}, Neg={neg_analysis["n_tokens"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            if i == 1:  # optimal_clusters，显示原始值
                ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
                       f'{int(pos_analysis["optimal_clusters"])}', ha='center', va='bottom', fontsize=8)
                ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
                       f'{int(neg_analysis["optimal_clusters"])}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
                       f'{height1:.3f}', ha='center', va='bottom', fontsize=8)
                ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
                       f'{height2:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/sample_level_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.show()  # 显示图表
    plt.close()
    
    print(f"✅ Sample-level analysis plot saved to: {os.path.abspath(VISUALIZATION_DIR)}/sample_level_analysis.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("CAUSAL JUDGMENT CLUSTER COHESION ANALYSIS (SAMPLE-WISE)")
    print("="*60)
    
    # 1. 加载数据
    data = load_causal_judgment_data(INPUT_PATH)
    if not data:
        print("❌ Failed to load data. Exiting.")
        return
    
    # 2. 提取正负样本对
    pairs = extract_positive_negative_pairs(data)
    if not pairs:
        print("❌ No valid positive-negative pairs found. Exiting.")
        return
    
    # 3. 分析聚类内聚度
    print("\n🔄 Analyzing sample-wise cluster cohesion...")
    analysis_results = analyze_cluster_cohesion_for_pairs(pairs)
    
    # 4. 保存结果
    output_file = f"{OUTPUT_DIR}/cluster_cohesion_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis results saved to: {os.path.abspath(output_file)}")
    
    # 5. 创建可视化
    create_visualizations(analysis_results)
    
    # 6. 打印总结
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    if 'overall_comparison' in analysis_results:
        stats = analysis_results['overall_comparison']
        print(f"📊 Total pairs analyzed: {len(pairs)}")
        print(f"📈 Positive samples mean cohesion: {stats['positive_mean']:.4f} ± {stats['positive_std']:.4f}")
        print(f"📉 Negative samples mean cohesion: {stats['negative_mean']:.4f} ± {stats['negative_std']:.4f}")
        print(f"🔍 Mean difference: {stats['mean_difference']:.4f}")
        print(f"📏 Effect size (Cohen's d): {stats['effect_size']:.4f}")
        
        # 解释效应量
        effect_size = abs(stats['effect_size'])
        if effect_size < 0.2:
            effect_desc = "negligible"
        elif effect_size < 0.5:
            effect_desc = "small"
        elif effect_size < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        print(f"📋 Effect size interpretation: {effect_desc}")
        
        direction = "higher" if stats['mean_difference'] > 0 else "lower"
        print(f"🎯 Conclusion: Positive samples show {direction} cluster cohesion than negative samples")
    
    print(f"\n📁 Output files:")
    print(f"  - Analysis data: {os.path.abspath(output_file)}")
    print(f"  - Visualizations: {os.path.abspath(VISUALIZATION_DIR)}/")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()