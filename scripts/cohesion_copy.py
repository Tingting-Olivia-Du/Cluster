# -*- coding: utf-8 -*-
"""
Causal Judgment Cluster Cohesion Analysis (Modified for actual data structure)
分析因果判断任务中正负样本的聚类内聚度特征 - 修改版本适配实际数据
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# 本地相对路径配置
INPUT_PATH = "./logits/bb_lies_mistral-nemo-12b_batch_6.json"
OUTPUT_DIR = "./output/bb_lies_mistral-nemo-12b_batch_6"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualizations"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

print("🚀 Starting Modified Cluster Cohesion Analysis...")
print(f"📂 Input path: {INPUT_PATH}")
print(f"📁 Output directory: {OUTPUT_DIR}")

# ============================================================================
# DATA LOADING AND HELPER FUNCTIONS
# ============================================================================

def load_data(file_path):
    """加载实验数据"""
    print(f"🔄 Attempting to load data from: {os.path.abspath(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded data")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def extract_token_features(token_details):
    """
    从token_details中提取可用于聚类分析的特征
    由于hidden_vector为空，我们使用其他可用的数值特征
    """
    features = []
    
    for token in token_details:
        # 提取可用的数值特征
        feature_vector = []
        
        # 1. 概率相关特征
        if 'chosen_prob' in token and token['chosen_prob'] is not None:
            feature_vector.append(token['chosen_prob'])
        else:
            feature_vector.append(0.0)
        
        # 2. 信息内容
        if 'information_content' in token and token['information_content'] is not None:
            # 处理可能的无限值
            ic = token['information_content']
            if np.isfinite(ic):
                feature_vector.append(ic)
            else:
                feature_vector.append(10.0)  # 用一个较大的有限值替代无限值
        else:
            feature_vector.append(0.0)
        
        # 3. topk信息中的特征
        if 'topk_info' in token and token['topk_info']:
            topk = token['topk_info']
            
            # 3.1 熵值
            if 'entropy' in topk and topk['entropy'] is not None:
                entropy = topk['entropy']
                if np.isfinite(entropy):
                    feature_vector.append(entropy)
                else:
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
            
            # 3.2 softmax分布的方差（反映不确定性）
            if 'softmax' in topk and topk['softmax']:
                softmax_vals = [s for s in topk['softmax'] if s is not None and np.isfinite(s)]
                if softmax_vals:
                    softmax_var = np.var(softmax_vals)
                    feature_vector.append(softmax_var)
                else:
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
            
            # 3.3 logits的范围
            if 'logits' in topk and topk['logits']:
                logits_vals = [l for l in topk['logits'] if l is not None and np.isfinite(l)]
                if len(logits_vals) > 1:
                    logits_range = max(logits_vals) - min(logits_vals)
                    feature_vector.append(logits_range)
                else:
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
        else:
            # 填充默认值
            feature_vector.extend([0.0, 0.0, 0.0])
        
        # 确保特征向量有固定长度
        if len(feature_vector) == 5:
            features.append(feature_vector)
    
    return np.array(features) if features else np.array([])

def compute_cluster_cohesion(vectors):
    """
    计算向量集合的聚类内聚度
    """
    if len(vectors) < 2:
        return {
            'cluster_cohesion': 0.0,
            'optimal_clusters': 1,
            'min_inertia': 0.0,
            'best_silhouette_score': 0.0,
            'n_tokens': len(vectors),
            'feature_variance': 0.0
        }
    
    n_tokens = len(vectors)
    
    # 计算特征方差作为额外指标
    feature_variance = float(np.mean(np.var(vectors, axis=0)))
    
    # 计算质心
    centroid = np.mean(vectors, axis=0)
    
    # 基础度量：到质心的距离
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
    best_silhouette_score = float(np.max(silhouette_scores)) if silhouette_scores else 0.0
    optimal_clusters = int(np.argmax(silhouette_scores) + 1) if silhouette_scores else 1
    min_inertia = float(np.min(inertias)) if inertias else 0.0
    
    # 聚类内聚度 = 1 / (1 + 标准化的最小惯性)
    cluster_cohesion = float(1 / (1 + min_inertia / n_tokens)) if n_tokens > 0 else 0.0
    
    return {
        'cluster_cohesion': cluster_cohesion,
        'optimal_clusters': optimal_clusters,
        'min_inertia': min_inertia,
        'best_silhouette_score': best_silhouette_score,
        'n_tokens': n_tokens,
        'feature_variance': feature_variance,
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
    
    print(f"🔍 Found {len(results)} questions in results")
    
    for qid, question_data in results.items():
        samplings = question_data.get('samplings', {})
        target_answer = question_data.get('target_answer', '')
        
        # 找到正样本和负样本
        positive_samples = []
        negative_samples = []
        
        for sid, sample in samplings.items():
            is_correct = sample.get('is_correct', False)
            
            # 检查是否有token_details
            if 'token_details' not in sample or not sample['token_details']:
                print(f"⚠️ No token_details in {qid}-{sid}")
                continue
            
            if is_correct:
                positive_samples.append((sid, sample))
            else:
                negative_samples.append((sid, sample))
        
        # 如果找到了配对的正负样本
        if positive_samples and negative_samples:
            pairs.append({
                'qid': qid,
                'target_answer': target_answer,
                'original_question': question_data.get('formatted_data', {}).get('original_question', 
                                   question_data.get('original_example', {}).get('input', '')),
                'positive': {
                    'sid': positive_samples[0][0],
                    'sample': positive_samples[0][1]
                },
                'negative': {
                    'sid': negative_samples[0][0],
                    'sample': negative_samples[0][1]
                },
                'total_positive': len(positive_samples),
                'total_negative': len(negative_samples)
            })
    
    print(f"✅ Found {len(pairs)} positive-negative pairs")
    return pairs

def analyze_sample_features(sample, sample_type):
    """分析单个样本的特征分布和聚类性质"""
    
    token_details = sample.get('token_details', [])
    if not token_details:
        return {
            'cluster_cohesion': 0.0,
            'sample_type': sample_type,
            'n_tokens': 0,
            'error': 'No token details'
        }
    
    # 提取特征
    features = extract_token_features(token_details)
    
    if len(features) < 2:
        return {
            'cluster_cohesion': 0.0,
            'sample_type': sample_type,
            'n_tokens': len(features),
            'error': f'Insufficient features: {len(features)}'
        }
    
    # 计算聚类内聚度
    try:
        cohesion_metrics = compute_cluster_cohesion(features)
        cohesion_metrics['sample_type'] = sample_type
        
        # 添加额外的统计信息
        cohesion_metrics.update({
            'feature_mean': float(np.mean(features)),
            'feature_std': float(np.std(features)),
            'prob_variance': float(np.var(features[:, 0])) if features.shape[1] > 0 else 0.0,
            'info_content_mean': float(np.mean(features[:, 1])) if features.shape[1] > 1 else 0.0
        })
        
        return cohesion_metrics
    except Exception as e:
        return {
            'cluster_cohesion': 0.0,
            'sample_type': sample_type,
            'n_tokens': len(features),
            'error': f'Computation error: {str(e)}'
        }

def analyze_all_pairs(pairs):
    """分析所有正负样本对"""
    
    analysis_results = {
        'pair_comparisons': [],
        'summary_statistics': {
            'total_pairs': len(pairs),
            'valid_pairs': 0,
            'positive_stats': {
                'cluster_cohesion': [],
                'optimal_clusters': [],
                'best_silhouette_score': [],
                'n_tokens': [],
                'feature_variance': []
            },
            'negative_stats': {
                'cluster_cohesion': [],
                'optimal_clusters': [],
                'best_silhouette_score': [],
                'n_tokens': [],
                'feature_variance': []
            }
        }
    }
    
    for pair_idx, pair in enumerate(pairs):
        print(f"🔄 Processing pair {pair_idx + 1}/{len(pairs)}: {pair['qid']}")
        
        # 分析正样本和负样本
        positive_analysis = analyze_sample_features(pair['positive']['sample'], 'positive')
        negative_analysis = analyze_sample_features(pair['negative']['sample'], 'negative')
        
        pair_result = {
            'qid': pair['qid'],
            'target_answer': pair['target_answer'],
            'original_question': pair['original_question'],
            'positive_analysis': positive_analysis,
            'negative_analysis': negative_analysis,
            'comparison': {
                'cohesion_difference': positive_analysis.get('cluster_cohesion', 0) - negative_analysis.get('cluster_cohesion', 0),
                'positive_cohesion': positive_analysis.get('cluster_cohesion', 0),
                'negative_cohesion': negative_analysis.get('cluster_cohesion', 0),
                'positive_tokens': positive_analysis.get('n_tokens', 0),
                'negative_tokens': negative_analysis.get('n_tokens', 0)
            }
        }
        
        # 统计有效样本
        if (positive_analysis.get('n_tokens', 0) >= 2 and 
            negative_analysis.get('n_tokens', 0) >= 2 and
            'error' not in positive_analysis and 
            'error' not in negative_analysis):
            
            analysis_results['summary_statistics']['valid_pairs'] += 1
            
            # 收集统计数据
            for stat_key in ['cluster_cohesion', 'optimal_clusters', 'best_silhouette_score', 'n_tokens', 'feature_variance']:
                if stat_key in positive_analysis:
                    analysis_results['summary_statistics']['positive_stats'][stat_key].append(positive_analysis[stat_key])
                if stat_key in negative_analysis:
                    analysis_results['summary_statistics']['negative_stats'][stat_key].append(negative_analysis[stat_key])
        
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
                                np.sqrt((np.var(pos_cohesions) + np.var(neg_cohesions)) / 2)) if (np.var(pos_cohesions) + np.var(neg_cohesions)) > 0 else 0.0
        }
    
    return analysis_results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comprehensive_visualizations(analysis_results):
    """创建综合可视化分析"""
    
    print("📊 Creating comprehensive visualizations...")
    
    # 检查数据可用性
    pos_cohesions = analysis_results['summary_statistics']['positive_stats']['cluster_cohesion']
    neg_cohesions = analysis_results['summary_statistics']['negative_stats']['cluster_cohesion']
    
    if not pos_cohesions or not neg_cohesions:
        print("⚠️ Insufficient data for visualization")
        return
    
    # 1. 主要对比图
    create_main_comparison_plot(analysis_results)
    
    # 2. 特征分析图
    create_feature_analysis_plot(analysis_results)
    
    # 3. 详细统计图
    create_detailed_statistics_plot(analysis_results)

def create_main_comparison_plot(analysis_results):
    """创建主要对比图"""
    
    pos_cohesions = analysis_results['summary_statistics']['positive_stats']['cluster_cohesion']
    neg_cohesions = analysis_results['summary_statistics']['negative_stats']['cluster_cohesion']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: 箱型图对比
    ax1 = axes[0, 0]
    data_to_plot = [pos_cohesions, neg_cohesions]
    labels = ['Positive\n(Correct)', 'Negative\n(Incorrect)']
    colors = ['lightgreen', 'lightcoral']
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Cluster Cohesion Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cluster Cohesion Score')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 分布直方图
    ax2 = axes[0, 1]
    ax2.hist(pos_cohesions, bins=15, alpha=0.7, label='Positive', color='green', density=True)
    ax2.hist(neg_cohesions, bins=15, alpha=0.7, label='Negative', color='red', density=True)
    ax2.set_xlabel('Cluster Cohesion Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 逐对比较
    ax3 = axes[1, 0]
    pairs = analysis_results['pair_comparisons']
    valid_pairs = [(i, p) for i, p in enumerate(pairs) 
                   if p['positive_analysis'].get('n_tokens', 0) >= 2 and 
                      p['negative_analysis'].get('n_tokens', 0) >= 2]
    
    if valid_pairs:
        indices, valid_pair_data = zip(*valid_pairs)
        pos_vals = [p['positive_analysis']['cluster_cohesion'] for p in valid_pair_data]
        neg_vals = [p['negative_analysis']['cluster_cohesion'] for p in valid_pair_data]
        
        x = np.arange(len(valid_pairs))
        width = 0.35
        
        ax3.bar(x - width/2, pos_vals, width, label='Positive', color='green', alpha=0.7)
        ax3.bar(x + width/2, neg_vals, width, label='Negative', color='red', alpha=0.7)
        
        ax3.set_xlabel('Question Pairs')
        ax3.set_ylabel('Cluster Cohesion')
        ax3.set_title('Pairwise Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 设置x轴标签
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Q{i+1}' for i in range(len(valid_pairs))], rotation=45)
    
    # 子图4: 累积分布
    ax4 = axes[1, 1]
    pos_sorted = np.sort(pos_cohesions)
    neg_sorted = np.sort(neg_cohesions)
    
    pos_y = np.arange(1, len(pos_sorted) + 1) / len(pos_sorted)
    neg_y = np.arange(1, len(neg_sorted) + 1) / len(neg_sorted)
    
    ax4.plot(pos_sorted, pos_y, label='Positive', color='green', linewidth=2)
    ax4.plot(neg_sorted, neg_y, label='Negative', color='red', linewidth=2)
    
    ax4.set_xlabel('Cluster Cohesion Score')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/comprehensive_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Comprehensive analysis plot saved")

def create_feature_analysis_plot(analysis_results):
    """创建特征分析图"""
    
    pos_stats = analysis_results['summary_statistics']['positive_stats']
    neg_stats = analysis_results['summary_statistics']['negative_stats']
    
    # 检查所需数据
    if not all(key in pos_stats and pos_stats[key] for key in ['feature_variance', 'n_tokens']):
        print("⚠️ Insufficient data for feature analysis plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 特征方差对比
    ax1 = axes[0, 0]
    if pos_stats['feature_variance'] and neg_stats['feature_variance']:
        ax1.boxplot([pos_stats['feature_variance'], neg_stats['feature_variance']], 
                   labels=['Positive', 'Negative'])
        ax1.set_title('Feature Variance Comparison')
        ax1.set_ylabel('Feature Variance')
        ax1.grid(True, alpha=0.3)
    
    # Token数量对比
    ax2 = axes[0, 1]
    if pos_stats['n_tokens'] and neg_stats['n_tokens']:
        ax2.boxplot([pos_stats['n_tokens'], neg_stats['n_tokens']], 
                   labels=['Positive', 'Negative'])
        ax2.set_title('Number of Tokens Comparison')
        ax2.set_ylabel('Number of Tokens')
        ax2.grid(True, alpha=0.3)
    
    # 最优聚类数对比
    ax3 = axes[1, 0]
    if pos_stats['optimal_clusters'] and neg_stats['optimal_clusters']:
        ax3.boxplot([pos_stats['optimal_clusters'], neg_stats['optimal_clusters']], 
                   labels=['Positive', 'Negative'])
        ax3.set_title('Optimal Clusters Comparison')
        ax3.set_ylabel('Optimal Number of Clusters')
        ax3.grid(True, alpha=0.3)
    
    # 轮廓系数对比
    ax4 = axes[1, 1]
    if pos_stats['best_silhouette_score'] and neg_stats['best_silhouette_score']:
        ax4.boxplot([pos_stats['best_silhouette_score'], neg_stats['best_silhouette_score']], 
                   labels=['Positive', 'Negative'])
        ax4.set_title('Silhouette Score Comparison')
        ax4.set_ylabel('Best Silhouette Score')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/feature_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Feature analysis plot saved")

def create_detailed_statistics_plot(analysis_results):
    """创建详细统计图"""
    
    if 'overall_comparison' not in analysis_results:
        print("⚠️ No overall comparison data available")
        return
    
    stats = analysis_results['overall_comparison']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 均值对比
    categories = ['Cluster Cohesion']
    pos_means = [stats['positive_mean']]
    neg_means = [stats['negative_mean']]
    pos_stds = [stats['positive_std']]
    neg_stds = [stats['negative_std']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pos_means, width, yerr=pos_stds, 
                   label='Positive', color='green', alpha=0.7, capsize=5)
    bars2 = ax1.bar(x + width/2, neg_means, width, yerr=neg_stds, 
                   label='Negative', color='red', alpha=0.7, capsize=5)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Mean Comparison with Error Bars')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 效应量和差异可视化
    ax2.text(0.1, 0.7, f"Mean Difference: {stats['mean_difference']:.4f}", 
             fontsize=14, transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f"Effect Size (Cohen's d): {stats['effect_size']:.4f}", 
             fontsize=14, transform=ax2.transAxes)
    
    # 解释效应量
    effect_size = abs(stats['effect_size'])
    if effect_size < 0.2:
        effect_desc = "Negligible effect"
    elif effect_size < 0.5:
        effect_desc = "Small effect"
    elif effect_size < 0.8:
        effect_desc = "Medium effect"
    else:
        effect_desc = "Large effect"
    
    ax2.text(0.1, 0.5, f"Interpretation: {effect_desc}", 
             fontsize=14, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    direction = "higher" if stats['mean_difference'] > 0 else "lower"
    ax2.text(0.1, 0.3, f"Positive samples show {direction}\ncluster cohesion than negative samples", 
             fontsize=12, transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Statistical Summary')
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/detailed_statistics.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Detailed statistics plot saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("MODIFIED CLUSTER COHESION ANALYSIS")
    print("基于实际数据结构的聚类内聚度分析")
    print("="*80)
    
    # 1. 加载数据
    data = load_data(INPUT_PATH)
    if not data:
        print("❌ Failed to load data. Exiting.")
        return
    
    # 打印数据信息
    if 'dataset_info' in data:
        dataset_info = data['dataset_info']
        print(f"📊 Dataset: {dataset_info.get('name', 'Unknown')}")
        print(f"🤖 Model: {dataset_info.get('model_info', {}).get('model_name', 'Unknown')}")
        print(f"📈 Total examples: {dataset_info.get('total_examples', 'Unknown')}")
        print(f"📈 Batch: {dataset_info.get('batch_info', {}).get('current_batch', 'Unknown')}")
    
    # 2. 提取正负样本对
    pairs = extract_positive_negative_pairs(data)
    if not pairs:
        print("❌ No valid positive-negative pairs found. Exiting.")
        return
    
    # 3. 分析特征和聚类
    print(f"\n🔄 Analyzing {len(pairs)} pairs...")
    analysis_results = analyze_all_pairs(pairs)
    
    # 4. 保存结果
    output_file = f"{OUTPUT_DIR}/cluster_cohesion_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis results saved to: {os.path.abspath(output_file)}")
    
    # 5. 创建可视化
    if analysis_results['summary_statistics']['valid_pairs'] > 0:
        create_comprehensive_visualizations(analysis_results)
    else:
        print("⚠️ No valid pairs found for visualization")
    
    # 6. 打印详细总结
    print_analysis_summary(analysis_results)
    
    print(f"\n📁 All output files saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("✅ Analysis complete!")

def print_analysis_summary(analysis_results):
    """打印分析总结"""
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY / 分析总结")
    print("="*80)
    
    stats = analysis_results['summary_statistics']
    total_pairs = stats['total_pairs']
    valid_pairs = stats['valid_pairs']
    
    print(f"📊 Total pairs found: {total_pairs}")
    print(f"✅ Valid pairs analyzed: {valid_pairs}")
    print(f"📉 Invalid pairs: {total_pairs - valid_pairs}")
    
    if valid_pairs == 0:
        print("❌ No valid pairs found. Cannot perform statistical analysis.")
        print("\n🔍 Possible reasons:")
        print("   - Hidden vectors are empty in the data")
        print("   - Token details are incomplete")
        print("   - Using extracted features instead of hidden vectors")
        return
    
    if 'overall_comparison' in analysis_results:
        comp = analysis_results['overall_comparison']
        print(f"\n📈 POSITIVE samples (Correct answers):")
        print(f"   Mean cluster cohesion: {comp['positive_mean']:.4f} ± {comp['positive_std']:.4f}")
        
        print(f"\n📉 NEGATIVE samples (Incorrect answers):")
        print(f"   Mean cluster cohesion: {comp['negative_mean']:.4f} ± {comp['negative_std']:.4f}")
        
        print(f"\n🔍 COMPARISON:")
        print(f"   Mean difference: {comp['mean_difference']:.4f}")
        print(f"   Effect size (Cohen's d): {comp['effect_size']:.4f}")
        
        # 解释效应量
        effect_size = abs(comp['effect_size'])
        if effect_size < 0.2:
            effect_desc = "negligible (< 0.2)"
            interpretation = "Almost no practical difference"
        elif effect_size < 0.5:
            effect_desc = "small (0.2-0.5)"
            interpretation = "Small practical difference"
        elif effect_size < 0.8:
            effect_desc = "medium (0.5-0.8)"
            interpretation = "Moderate practical difference"
        else:
            effect_desc = "large (> 0.8)"
            interpretation = "Large practical difference"
        
        print(f"   Effect size interpretation: {effect_desc}")
        print(f"   Practical significance: {interpretation}")
        
        direction = "HIGHER" if comp['mean_difference'] > 0 else "LOWER"
        print(f"\n🎯 CONCLUSION:")
        print(f"   Positive samples show {direction} cluster cohesion than negative samples")
        
        if comp['mean_difference'] > 0:
            print("   → Correct answers tend to have more coherent internal representations")
            print("   → This suggests better organized cognitive processing for correct responses")
        else:
            print("   → Incorrect answers tend to have more coherent internal representations")
            print("   → This might indicate overconfident but wrong processing")
    
    # 打印特征统计
    pos_stats = stats['positive_stats']
    neg_stats = stats['negative_stats']
    
    if pos_stats['n_tokens'] and neg_stats['n_tokens']:
        print(f"\n📋 TOKEN STATISTICS:")
        print(f"   Positive samples - avg tokens: {np.mean(pos_stats['n_tokens']):.1f}")
        print(f"   Negative samples - avg tokens: {np.mean(neg_stats['n_tokens']):.1f}")
    
    if pos_stats['feature_variance'] and neg_stats['feature_variance']:
        print(f"\n📋 FEATURE VARIANCE:")
        print(f"   Positive samples - avg variance: {np.mean(pos_stats['feature_variance']):.4f}")
        print(f"   Negative samples - avg variance: {np.mean(neg_stats['feature_variance']):.4f}")
    
    print(f"\n📝 METHODOLOGY NOTE:")
    print(f"   Since hidden_vector fields were empty, analysis used extracted features:")
    print(f"   - Token probability (chosen_prob)")
    print(f"   - Information content")
    print(f"   - Entropy from topk_info")
    print(f"   - Softmax variance")
    print(f"   - Logits range")
    
    print(f"\n📊 VISUALIZATION FILES:")
    print(f"   - {VISUALIZATION_DIR}/comprehensive_analysis.png")
    print(f"   - {VISUALIZATION_DIR}/feature_analysis.png") 
    print(f"   - {VISUALIZATION_DIR}/detailed_statistics.png")

def analyze_data_structure(data):
    """分析数据结构以便调试"""
    
    print("\n🔍 DATA STRUCTURE ANALYSIS:")
    print("="*50)
    
    if 'results' in data:
        results = data['results']
        print(f"📊 Number of questions: {len(results)}")
        
        # 分析第一个问题的结构
        first_qid = list(results.keys())[0]
        first_question = results[first_qid]
        
        print(f"📝 First question ID: {first_qid}")
        print(f"📝 First question keys: {list(first_question.keys())}")
        
        if 'samplings' in first_question:
            samplings = first_question['samplings']
            print(f"📊 Number of samplings: {len(samplings)}")
            
            first_sampling = list(samplings.values())[0]
            print(f"📝 First sampling keys: {list(first_sampling.keys())}")
            
            if 'token_details' in first_sampling:
                token_details = first_sampling['token_details']
                print(f"📊 Number of tokens: {len(token_details)}")
                
                if token_details:
                    first_token = token_details[0]
                    print(f"📝 First token keys: {list(first_token.keys())}")
                    
                    # 检查hidden_vector
                    if 'hidden_vector' in first_token:
                        hv = first_token['hidden_vector']
                        print(f"📝 Hidden vector type: {type(hv)}")
                        print(f"📝 Hidden vector length: {len(hv) if hv else 0}")
                        print(f"📝 Hidden vector sample: {hv[:5] if hv else 'Empty'}")
                    
                    # 检查其他可用特征
                    available_features = []
                    for key in ['chosen_prob', 'information_content', 'topk_info']:
                        if key in first_token and first_token[key] is not None:
                            available_features.append(key)
                    
                    print(f"📝 Available features: {available_features}")

if __name__ == "__main__":
    # 首先分析数据结构
    print("🔍 Starting data structure analysis...")
    temp_data = load_data(INPUT_PATH)
    if temp_data:
        analyze_data_structure(temp_data)
    
    print("\n" + "="*80)
    
    # 运行主分析
    main()