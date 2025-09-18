# scripts/cohesion_final_enhanced.py
# -*- coding: utf-8 -*-
"""
Causal Judgment Cluster Cohesion Analysis (Sample-wise)
修改版：
1. 结果保存到 /results 目录
2. 只用 KMeans 计算 cluster 数量
3. 先正负样本对比，再做所有样本统计（不分正负）
4. bb, bbh, bbeh 三数据集对比
5. 自动处理所有数据集
6. 新增：添加log处理的cohesion指标
7. 修改：使用optimal_k对应的inertia，避免cluster数量偏置
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
# 三个Mistral数据集配置
# NAME = "lies_mistral-nemo-12b_batch_7"
NAME = "lies_ministral-8b_batch_5"
# NAME = "lies_mistral_batch_4"
# NAME = "lies_qwen_batch_3"

# 多数据集对比配置
DATASETS = {
    'bb': {
        'path': f'./results/bb_{NAME}/cluster_analysis_results.json',
        'display_name': 'BB Lies',
        'color': '#1f77b4'
    },
    'bbh': {
        'path': f'./results/bbh_{NAME}/cluster_analysis_results.json',
        'display_name': 'BBH Lies',
        'color': '#ff7f0e'
    },
    'bbeh': {
        'path': f'./results/bbeh_{NAME}/cluster_analysis_results.json',
        'display_name': 'BBEH Lies',
        'color': '#2ca02c'
    }
}

COMPARISON_OUTPUT_DIR = f"./results/comparison_{NAME}_2"
COMPARISON_VIS_DIR = f"{COMPARISON_OUTPUT_DIR}/visualizations"

# 全局变量（用于可视化函数）
VISUALIZATION_DIR = ""  # 将在main函数中动态设置

os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPARISON_VIS_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
def clean_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def collect_vectors(sample):
    return [clean_vector(t["hidden_vector"]) for t in sample.get("token_details", [])
            if isinstance(t.get("hidden_vector"), list) and len(t["hidden_vector"]) > 0]

def find_first_valid_sample(samples):
    for sid, sample in samples:
        vecs = collect_vectors(sample)
        if len(vecs) >= 2:
            return sid, sample, vecs
    return None, None, None

def collect_all_samples(qdata):
    """收集所有有效样本（不分正负）"""
    all_samples = []
    for sid, sample in qdata["samplings"].items():
        vecs = collect_vectors(sample)
        if len(vecs) >= 2:
            all_samples.append((sid, sample, vecs))
    return all_samples

# ---------------- CLUSTER ANALYSIS ----------------
def compute_cluster_metrics(vectors, max_k=5):
    """计算cluster数量和cohesion指标（使用optimal_k的inertia，避免cluster数量偏置）"""
    if len(vectors) < 2:
        return None
    
    vectors = np.array(vectors)
    vectors = StandardScaler().fit_transform(vectors)
    n_tokens = len(vectors)
    max_k = min(max_k, n_tokens)
    
    # 基本统计
    centroid = np.mean(vectors, axis=0)
    centroid_distances = [np.linalg.norm(v - centroid) for v in vectors]
    
    if max_k < 2:
        # 只有一个cluster的情况
        inertia = np.sum([np.linalg.norm(v - centroid) ** 2 for v in vectors])
        optimal_inertia = float(inertia / n_tokens)
        cluster_cohesion = float(1 / (1 + optimal_inertia))
        
        # 新增：对原始cohesion值套log
        log_cohesion = float(np.log(cluster_cohesion))
        
        return {
            "optimal_clusters": 1,
            "n_tokens": n_tokens,
            "silhouette_scores": [0.0],
            "best_silhouette": 0.0,
            "cluster_cohesion": cluster_cohesion,
            "log_cohesion": log_cohesion,
            "optimal_inertia": optimal_inertia,
            "centroid_distance_mean": float(np.mean(centroid_distances)),
            "centroid_distance_std": float(np.std(centroid_distances))
        }
    
    # 多cluster分析
    silhouette_scores = []
    inertias = []
    
    for k in range(1, max_k + 1):
        if k == 1:
            inertia = np.sum([np.linalg.norm(v - centroid) ** 2 for v in vectors])
            inertias.append(inertia)
            silhouette_scores.append(0.0)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            inertias.append(kmeans.inertia_)
            if len(set(labels)) > 1:
                score = silhouette_score(vectors, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0.0)
    
    optimal_k = int(np.argmax(silhouette_scores) + 1)
    
    # 🔧 修改：使用optimal_k对应的inertia，而不是最小inertia
    optimal_inertia = float(inertias[optimal_k - 1] / n_tokens)
    cluster_cohesion = float(1 / (1 + optimal_inertia))
    
    # 新增：对原始cohesion值套log
    log_cohesion = float(np.log(cluster_cohesion))
    
    return {
        "optimal_clusters": optimal_k,
        "n_tokens": n_tokens,
        "silhouette_scores": silhouette_scores,
        "best_silhouette": float(np.max(silhouette_scores)),
        "cluster_cohesion": cluster_cohesion,
        "log_cohesion": log_cohesion,
        "optimal_inertia": optimal_inertia,  # 改名：从min_inertia改为optimal_inertia
        "centroid_distance_mean": float(np.mean(centroid_distances)),
        "centroid_distance_std": float(np.std(centroid_distances)),
        "inertias": [float(i) for i in inertias]
    }

# ---------------- MAIN ANALYSIS ----------------
def analyze_samples(data):
    """分析样本：先正负对比，再所有样本统计"""
    results = {
        "positive_negative_comparison": {
            "pair_comparisons": [],
            "summary": {
                "total_pairs": 0,
                "valid_pairs": 0,
                "positive_stats": {"optimal_clusters": [], "n_tokens": [], "cluster_cohesion": [], "log_cohesion": []},
                "negative_stats": {"optimal_clusters": [], "n_tokens": [], "cluster_cohesion": [], "log_cohesion": []}
            }
        },
        "all_samples_analysis": {
            "question_analyses": [],
            "summary": {
                "total_questions": 0,
                "valid_questions": 0,
                "all_stats": {"optimal_clusters": [], "n_tokens": [], "samples_per_question": [], "cluster_cohesion": [], "log_cohesion": []}
            }
        }
    }

    # 第一部分：正负样本对比
    print("🔍 Step 1: Analyzing positive vs negative samples...")
    for qid, qdata in data.get("results", {}).items():
        pos_samples = [(sid, s) for sid, s in qdata["samplings"].items() if s.get("is_correct", False)]
        neg_samples = [(sid, s) for sid, s in qdata["samplings"].items() if not s.get("is_correct", False)]

        pos_sid, pos_sample, pos_vecs = find_first_valid_sample(pos_samples)
        neg_sid, neg_sample, neg_vecs = find_first_valid_sample(neg_samples)

        results["positive_negative_comparison"]["summary"]["total_pairs"] += 1

        if pos_sample and neg_sample:
            pos_metrics = compute_cluster_metrics(pos_vecs)
            neg_metrics = compute_cluster_metrics(neg_vecs)
            
            if pos_metrics and neg_metrics:
                results["positive_negative_comparison"]["summary"]["valid_pairs"] += 1
                
                # 收集统计数据
                results["positive_negative_comparison"]["summary"]["positive_stats"]["optimal_clusters"].append(pos_metrics["optimal_clusters"])
                results["positive_negative_comparison"]["summary"]["positive_stats"]["n_tokens"].append(pos_metrics["n_tokens"])
                results["positive_negative_comparison"]["summary"]["positive_stats"]["cluster_cohesion"].append(pos_metrics["cluster_cohesion"])
                results["positive_negative_comparison"]["summary"]["positive_stats"]["log_cohesion"].append(pos_metrics["log_cohesion"])
                
                results["positive_negative_comparison"]["summary"]["negative_stats"]["optimal_clusters"].append(neg_metrics["optimal_clusters"])
                results["positive_negative_comparison"]["summary"]["negative_stats"]["n_tokens"].append(neg_metrics["n_tokens"])
                results["positive_negative_comparison"]["summary"]["negative_stats"]["cluster_cohesion"].append(neg_metrics["cluster_cohesion"])
                results["positive_negative_comparison"]["summary"]["negative_stats"]["log_cohesion"].append(neg_metrics["log_cohesion"])
                
                results["positive_negative_comparison"]["pair_comparisons"].append({
                    "qid": qid,
                    "positive_analysis": pos_metrics,
                    "negative_analysis": neg_metrics,
                    "cluster_count_difference": pos_metrics["optimal_clusters"] - neg_metrics["optimal_clusters"],
                    "cohesion_difference": pos_metrics["cluster_cohesion"] - neg_metrics["cluster_cohesion"],
                    "log_cohesion_difference": pos_metrics["log_cohesion"] - neg_metrics["log_cohesion"]
                })

    # 第二部分：所有样本分析（不分正负）
    print("🔍 Step 2: Analyzing all samples together...")
    for qid, qdata in data.get("results", {}).items():
        all_samples = collect_all_samples(qdata)
        results["all_samples_analysis"]["summary"]["total_questions"] += 1
        
        if len(all_samples) > 0:
            results["all_samples_analysis"]["summary"]["valid_questions"] += 1
            
            question_analysis = {
                "qid": qid,
                "n_samples": len(all_samples),
                "sample_analyses": []
            }
            
            for sid, sample, vecs in all_samples:
                metrics = compute_cluster_metrics(vecs)
                if metrics:
                    question_analysis["sample_analyses"].append({
                        "sample_id": sid,
                        "is_correct": sample.get("is_correct", False),
                        "metrics": metrics
                    })
                    
                    # 收集全局统计
                    results["all_samples_analysis"]["summary"]["all_stats"]["optimal_clusters"].append(metrics["optimal_clusters"])
                    results["all_samples_analysis"]["summary"]["all_stats"]["n_tokens"].append(metrics["n_tokens"])
                    results["all_samples_analysis"]["summary"]["all_stats"]["cluster_cohesion"].append(metrics["cluster_cohesion"])
                    results["all_samples_analysis"]["summary"]["all_stats"]["log_cohesion"].append(metrics["log_cohesion"])
            
            if question_analysis["sample_analyses"]:
                results["all_samples_analysis"]["summary"]["all_stats"]["samples_per_question"].append(len(question_analysis["sample_analyses"]))
                results["all_samples_analysis"]["question_analyses"].append(question_analysis)

    # 计算总体统计
    _compute_summary_statistics(results)
    return results

def _compute_summary_statistics(results):
    """计算总体统计信息（包括log_cohesion）"""
    # 正负对比统计
    pos_clusters = results["positive_negative_comparison"]["summary"]["positive_stats"]["optimal_clusters"]
    neg_clusters = results["positive_negative_comparison"]["summary"]["negative_stats"]["optimal_clusters"]
    pos_cohesion = results["positive_negative_comparison"]["summary"]["positive_stats"]["cluster_cohesion"]
    neg_cohesion = results["positive_negative_comparison"]["summary"]["negative_stats"]["cluster_cohesion"]
    pos_log_cohesion = results["positive_negative_comparison"]["summary"]["positive_stats"]["log_cohesion"]
    neg_log_cohesion = results["positive_negative_comparison"]["summary"]["negative_stats"]["log_cohesion"]
    
    if pos_clusters and neg_clusters:
        results["positive_negative_comparison"]["overall_statistics"] = {
            "clusters": {
                "positive_mean_clusters": float(np.mean(pos_clusters)),
                "negative_mean_clusters": float(np.mean(neg_clusters)),
                "cluster_count_difference": float(np.mean(pos_clusters) - np.mean(neg_clusters)),
                "positive_cluster_distribution": {str(k): int(v) for k, v in zip(*np.unique(pos_clusters, return_counts=True))},
                "negative_cluster_distribution": {str(k): int(v) for k, v in zip(*np.unique(neg_clusters, return_counts=True))}
            },
            "cohesion": {
                "positive_mean_cohesion": float(np.mean(pos_cohesion)),
                "negative_mean_cohesion": float(np.mean(neg_cohesion)),
                "cohesion_difference": float(np.mean(pos_cohesion) - np.mean(neg_cohesion)),
                "effect_size": float((np.mean(pos_cohesion) - np.mean(neg_cohesion)) /
                                   np.sqrt((np.var(pos_cohesion) + np.var(neg_cohesion)) / 2)) if len(pos_cohesion) > 1 and len(neg_cohesion) > 1 else 0.0
            },
            "log_cohesion": {
                "positive_mean_log_cohesion": float(np.mean(pos_log_cohesion)),
                "negative_mean_log_cohesion": float(np.mean(neg_log_cohesion)),
                "log_cohesion_difference": float(np.mean(pos_log_cohesion) - np.mean(neg_log_cohesion)),
                "log_effect_size": float((np.mean(pos_log_cohesion) - np.mean(neg_log_cohesion)) /
                                       np.sqrt((np.var(pos_log_cohesion) + np.var(neg_log_cohesion)) / 2)) if len(pos_log_cohesion) > 1 and len(neg_log_cohesion) > 1 else 0.0
            }
        }
    
    # 所有样本统计
    all_clusters = results["all_samples_analysis"]["summary"]["all_stats"]["optimal_clusters"]
    all_tokens = results["all_samples_analysis"]["summary"]["all_stats"]["n_tokens"]
    all_cohesion = results["all_samples_analysis"]["summary"]["all_stats"]["cluster_cohesion"]
    all_log_cohesion = results["all_samples_analysis"]["summary"]["all_stats"]["log_cohesion"]
    samples_per_q = results["all_samples_analysis"]["summary"]["all_stats"]["samples_per_question"]
    
    if all_clusters:
        results["all_samples_analysis"]["overall_statistics"] = {
            "basic_stats": {
                "total_samples_analyzed": len(all_clusters),
                "mean_tokens_per_sample": float(np.mean(all_tokens)),
                "mean_samples_per_question": float(np.mean(samples_per_q)) if samples_per_q else 0
            },
            "clusters": {
                "mean_clusters": float(np.mean(all_clusters)),
                "cluster_distribution": {str(k): int(v) for k, v in zip(*np.unique(all_clusters, return_counts=True))},
                "cluster_stats": {
                    "min": int(np.min(all_clusters)),
                    "max": int(np.max(all_clusters)),
                    "std": float(np.std(all_clusters))
                }
            },
            "cohesion": {
                "mean_cohesion": float(np.mean(all_cohesion)),
                "cohesion_stats": {
                    "min": float(np.min(all_cohesion)),
                    "max": float(np.max(all_cohesion)),
                    "std": float(np.std(all_cohesion))
                }
            },
            "log_cohesion": {
                "mean_log_cohesion": float(np.mean(all_log_cohesion)),
                "log_cohesion_stats": {
                    "min": float(np.min(all_log_cohesion)),
                    "max": float(np.max(all_log_cohesion)),
                    "std": float(np.std(all_log_cohesion))
                }
            }
        }

# ---------------- VISUALIZATION ----------------
def plot_positive_negative_comparison(results):
    """正负样本对比可视化：cluster数量 + cohesion（包括log版本）"""
    pn_data = results["positive_negative_comparison"]
    pos_clusters = pn_data["summary"]["positive_stats"]["optimal_clusters"]
    neg_clusters = pn_data["summary"]["negative_stats"]["optimal_clusters"]
    pos_cohesion = pn_data["summary"]["positive_stats"]["cluster_cohesion"]
    neg_cohesion = pn_data["summary"]["negative_stats"]["cluster_cohesion"]
    pos_log_cohesion = pn_data["summary"]["positive_stats"]["log_cohesion"]
    neg_log_cohesion = pn_data["summary"]["negative_stats"]["log_cohesion"]
    
    if not pos_clusters or not neg_clusters:
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 添加总标题，包含模型名字
    fig.suptitle(f'Positive vs Negative Comparison - {NAME}', fontsize=16, fontweight='bold')
    
    # 图1：cluster数量箱线图对比
    axes[0, 0].boxplot([pos_clusters, neg_clusters], labels=["Positive", "Negative"], patch_artist=True)
    axes[0, 0].set_title(f"Optimal Cluster Count: Positive vs Negative")
    axes[0, 0].set_ylabel("Number of Clusters")
    
    # 图2：cluster数量分布直方图
    all_clusters = list(range(1, max(max(pos_clusters), max(neg_clusters)) + 1))
    pos_counts = [pos_clusters.count(k) for k in all_clusters]
    neg_counts = [neg_clusters.count(k) for k in all_clusters]
    
    x = np.arange(len(all_clusters))
    width = 0.35
    axes[0, 1].bar(x - width/2, pos_counts, width, label='Positive', alpha=0.7, color='lightgreen')
    axes[0, 1].bar(x + width/2, neg_counts, width, label='Negative', alpha=0.7, color='lightcoral')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Cluster Count Distribution')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_clusters)
    axes[0, 1].legend()
    
    # 图3：原始cohesion箱线图对比
    axes[0, 2].boxplot([pos_cohesion, neg_cohesion], labels=["Positive", "Negative"], patch_artist=True)
    axes[0, 2].set_title(f"Cluster Cohesion: Positive vs Negative")
    axes[0, 2].set_ylabel("Cohesion Score")
    
    # 图4：log cohesion箱线图对比
    axes[1, 0].boxplot([pos_log_cohesion, neg_log_cohesion], labels=["Positive", "Negative"], patch_artist=True)
    axes[1, 0].set_title(f"Log Cohesion: Positive vs Negative")
    axes[1, 0].set_ylabel("Log Cohesion Score")
    
    # 图5：原始cohesion散点图
    axes[1, 1].scatter(pos_clusters, pos_cohesion, alpha=0.6, color='lightgreen', label='Positive', s=50)
    axes[1, 1].scatter(neg_clusters, neg_cohesion, alpha=0.6, color='lightcoral', label='Negative', s=50)
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Cohesion Score')
    axes[1, 1].set_title(f'Cluster Count vs Cohesion')
    axes[1, 1].legend()
    
    # 图6：log cohesion散点图
    axes[1, 2].scatter(pos_clusters, pos_log_cohesion, alpha=0.6, color='lightgreen', label='Positive', s=50)
    axes[1, 2].scatter(neg_clusters, neg_log_cohesion, alpha=0.6, color='lightcoral', label='Negative', s=50)
    axes[1, 2].set_xlabel('Number of Clusters')
    axes[1, 2].set_ylabel('Log Cohesion Score')
    axes[1, 2].set_title(f'Cluster Count vs Log Cohesion')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/positive_negative_comparison.png", dpi=150)
    plt.close()

def plot_all_samples_analysis(results):
    """所有样本分析可视化：cluster数量 + cohesion（包括log版本）"""
    all_data = results["all_samples_analysis"]
    all_clusters = all_data["summary"]["all_stats"]["optimal_clusters"]
    all_cohesion = all_data["summary"]["all_stats"]["cluster_cohesion"]
    all_log_cohesion = all_data["summary"]["all_stats"]["log_cohesion"]
    
    if not all_clusters:
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 添加总标题，包含模型名字
    fig.suptitle(f'All Samples Analysis - {NAME}', fontsize=16, fontweight='bold')
    
    # 图1：cluster数量分布直方图
    cluster_counts = {}
    for c in all_clusters:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    
    clusters = list(cluster_counts.keys())
    counts = list(cluster_counts.values())
    
    axes[0, 0].bar(clusters, counts, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Cluster Count Distribution')
    axes[0, 0].set_xticks(clusters)
    
    # 图2：cluster数量箱线图
    axes[0, 1].boxplot([all_clusters], labels=['All Samples'])
    axes[0, 1].set_title(f'Cluster Count Distribution')
    axes[0, 1].set_ylabel('Number of Clusters')
    
    # 图3：原始cohesion分布直方图
    axes[0, 2].hist(all_cohesion, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_xlabel('Cohesion Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Cohesion Distribution')
    
    # 图4：log cohesion分布直方图
    axes[1, 0].hist(all_log_cohesion, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Log Cohesion Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Log Cohesion Distribution')
    
    # 图5：cluster数量 vs 原始cohesion散点图
    axes[1, 1].scatter(all_clusters, all_cohesion, alpha=0.6, color='purple', s=30)
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Cohesion Score')
    axes[1, 1].set_title(f'Cluster Count vs Cohesion')
    
    # 图6：cluster数量 vs log cohesion散点图
    axes[1, 2].scatter(all_clusters, all_log_cohesion, alpha=0.6, color='red', s=30)
    axes[1, 2].set_xlabel('Number of Clusters')
    axes[1, 2].set_ylabel('Log Cohesion Score')
    axes[1, 2].set_title(f'Cluster Count vs Log Cohesion')
    
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/all_samples_analysis.png", dpi=150)
    plt.close()

def plot_pairwise_clusters(results):
    """每个问题的正负样本cluster对比"""
    pairs = results["positive_negative_comparison"]["pair_comparisons"]
    if not pairs:
        return
    pos_vals = [p["positive_analysis"]["optimal_clusters"] for p in pairs]
    neg_vals = [p["negative_analysis"]["optimal_clusters"] for p in pairs]
    labels = [p["qid"] for p in pairs]
    
    fig_width = min(max(len(labels) * 0.5, 8), 30)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # 添加总标题，包含模型名字
    fig.suptitle(f'Pairwise Cluster Count Comparison - {NAME}', fontsize=14, fontweight='bold')
    
    x = np.arange(len(labels))
    ax.bar(x - 0.2, pos_vals, 0.4, label="Positive", color="lightgreen")
    ax.bar(x + 0.2, neg_vals, 0.4, label="Negative", color="lightcoral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_title("Optimal Cluster Count by Question")
    ax.set_ylabel("Number of Clusters")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/pairwise_clusters_comparison.png", dpi=150)
    plt.close()

# ---------------- DATASET COMPARISON ----------------
def load_dataset_results():
    """加载三个数据集的结果"""
    dataset_results = {}
    for key, config in DATASETS.items():
        if os.path.exists(config['path']):
            with open(config['path'], 'r', encoding='utf-8') as f:
                dataset_results[key] = {
                    'data': json.load(f),
                    'config': config
                }
        else:
            print(f"⚠️  Warning: {config['path']} not found, skipping {key}")
    return dataset_results

def plot_dataset_comparison(dataset_results):
    """三数据集对比可视化：cluster数量 + cohesion（包括log版本）"""
    if len(dataset_results) < 2:
        print("⚠️  需要至少2个数据集进行对比")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 添加总标题，包含模型名字
    fig.suptitle(f'Dataset Comparison - {NAME}', fontsize=16, fontweight='bold')
    
    # 数据准备
    datasets = []
    pos_clusters = []
    neg_clusters = []
    pos_cohesion = []
    neg_cohesion = []
    pos_log_cohesion = []
    neg_log_cohesion = []
    all_clusters = []
    all_cohesion = []
    all_log_cohesion = []
    colors = []
    
    for key, result in dataset_results.items():
        data = result['data']
        config = result['config']
        datasets.append(config['display_name'])
        colors.append(config['color'])
        
        # 正负样本数据
        pn_data = data['positive_negative_comparison']['summary']
        pos_clusters.append(pn_data['positive_stats']['optimal_clusters'])
        neg_clusters.append(pn_data['negative_stats']['optimal_clusters'])
        pos_cohesion.append(pn_data['positive_stats']['cluster_cohesion'])
        neg_cohesion.append(pn_data['negative_stats']['cluster_cohesion'])
        pos_log_cohesion.append(pn_data['positive_stats']['log_cohesion'])
        neg_log_cohesion.append(pn_data['negative_stats']['log_cohesion'])
        
        # 所有样本数据
        all_data = data['all_samples_analysis']['summary']
        all_clusters.append(all_data['all_stats']['optimal_clusters'])
        all_cohesion.append(all_data['all_stats']['cluster_cohesion'])
        all_log_cohesion.append(all_data['all_stats']['log_cohesion'])
    
    # 图1: 正负样本cluster数量均值对比
    pos_cluster_means = [np.mean(clusters) for clusters in pos_clusters]
    neg_cluster_means = [np.mean(clusters) for clusters in neg_clusters]
    x = np.arange(len(datasets))
    width = 0.35
    axes[0, 0].bar(x - width/2, pos_cluster_means, width, label='Positive', color='lightgreen')
    axes[0, 0].bar(x + width/2, neg_cluster_means, width, label='Negative', color='lightcoral')
    axes[0, 0].set_title('Positive vs Negative: Mean Cluster Count by Dataset')
    axes[0, 0].set_ylabel('Mean Number of Clusters')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].legend()
    
    # 图2: 正负样本原始cohesion均值对比
    pos_cohesion_means = [np.mean(cohesion) for cohesion in pos_cohesion]
    neg_cohesion_means = [np.mean(cohesion) for cohesion in neg_cohesion]
    axes[0, 1].bar(x - width/2, pos_cohesion_means, width, label='Positive', color='lightgreen')
    axes[0, 1].bar(x + width/2, neg_cohesion_means, width, label='Negative', color='lightcoral')
    axes[0, 1].set_title('Positive vs Negative: Mean Cohesion by Dataset')
    axes[0, 1].set_ylabel('Mean Cohesion Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].legend()
    
    # 图3: 正负样本log cohesion均值对比
    pos_log_cohesion_means = [np.mean(cohesion) for cohesion in pos_log_cohesion]
    neg_log_cohesion_means = [np.mean(cohesion) for cohesion in neg_log_cohesion]
    axes[0, 2].bar(x - width/2, pos_log_cohesion_means, width, label='Positive', color='lightgreen')
    axes[0, 2].bar(x + width/2, neg_log_cohesion_means, width, label='Negative', color='lightcoral')
    axes[0, 2].set_title('Positive vs Negative: Mean Log Cohesion by Dataset')
    axes[0, 2].set_ylabel('Mean Log Cohesion Score')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(datasets)
    axes[0, 2].legend()
    
    # 图4: 所有样本cluster数量对比
    all_cluster_means = [np.mean(clusters) for clusters in all_clusters]
    axes[1, 0].bar(x, all_cluster_means, color='skyblue', alpha=0.7)
    axes[1, 0].set_title('All Samples: Mean Cluster Count by Dataset')
    axes[1, 0].set_ylabel('Mean Number of Clusters')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(datasets)
    
    # 图5: 所有样本原始cohesion对比
    all_cohesion_means = [np.mean(cohesion) for cohesion in all_cohesion]
    axes[1, 1].bar(x, all_cohesion_means, color='lightpink', alpha=0.7)
    axes[1, 1].set_title('All Samples: Mean Cohesion by Dataset')
    axes[1, 1].set_ylabel('Mean Cohesion Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(datasets)
    
    # 图6: 所有样本log cohesion对比
    all_log_cohesion_means = [np.mean(cohesion) for cohesion in all_log_cohesion]
    axes[1, 2].bar(x, all_log_cohesion_means, color='orange', alpha=0.7)
    axes[1, 2].set_title('All Samples: Mean Log Cohesion by Dataset')
    axes[1, 2].set_ylabel('Mean Log Cohesion Score')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(datasets)
    
    plt.tight_layout()
    plt.savefig(f"{COMPARISON_VIS_DIR}/dataset_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_summary(dataset_results):
    """创建对比总结JSON（包括log_cohesion）"""
    comparison_summary = {
        "datasets_compared": len(dataset_results),
        "analysis_type": "kmeans_cluster_count_and_cohesion_with_log_optimal_inertia",
        "comparison_results": {}
    }
    
    for key, result in dataset_results.items():
        data = result['data']
        config = result['config']
        
        comparison_summary["comparison_results"][key] = {
            "display_name": config['display_name'],
            "positive_negative_comparison": data.get('positive_negative_comparison', {}).get('overall_statistics', {}),
            "all_samples_analysis": data.get('all_samples_analysis', {}).get('overall_statistics', {})
        }
    
    output_path = f"{COMPARISON_OUTPUT_DIR}/dataset_comparison_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    print(f"📁 Saved comparison summary to {output_path}")

# ---------------- MAIN ----------------
def main():
    """主函数：自动处理三个数据集 + 对比分析"""
    
    # 1. 自动处理三个数据集
    datasets_to_process = ['bb', 'bbh', 'bbeh']
    processed_datasets = []
    
    global VISUALIZATION_DIR  # 声明全局变量
    
    for dataset in datasets_to_process:
        print(f"\n🔍 Processing dataset: {dataset}_{NAME}")
        input_path = f"./logits/{dataset}_{NAME}.json"
        output_dir = f"./results/{dataset}_{NAME}"
        vis_dir = f"{output_dir}/visualizations"
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        if os.path.exists(input_path):
            try:
                # 读取数据
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 分析数据
                results = analyze_samples(data)

                # 保存结果
                json_path = f"{output_dir}/cluster_analysis_results.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"📁 Saved analysis JSON to {json_path}")

                # 生成可视化（使用临时的全局变量）
                original_vis_dir = VISUALIZATION_DIR
                VISUALIZATION_DIR = vis_dir
                
                plot_positive_negative_comparison(results)
                plot_all_samples_analysis(results)
                plot_pairwise_clusters(results)
                
                VISUALIZATION_DIR = original_vis_dir  # 恢复原值

                # 打印统计信息
                print(f"✅ {dataset} analysis completed")
                pn_stats = results.get('positive_negative_comparison', {}).get('overall_statistics', {})
                all_stats = results.get('all_samples_analysis', {}).get('overall_statistics', {})
                
                if pn_stats:
                    clusters_info = pn_stats.get('clusters', {})
                    cohesion_info = pn_stats.get('cohesion', {})
                    log_cohesion_info = pn_stats.get('log_cohesion', {})
                    print(f"   Pos/Neg Clusters: {clusters_info.get('positive_mean_clusters', 0):.2f} / {clusters_info.get('negative_mean_clusters', 0):.2f}")
                    print(f"   Pos/Neg Cohesion: {cohesion_info.get('positive_mean_cohesion', 0):.4f} / {cohesion_info.get('negative_mean_cohesion', 0):.4f}")
                    print(f"   Pos/Neg Log Cohesion: {log_cohesion_info.get('positive_mean_log_cohesion', 0):.4f} / {log_cohesion_info.get('negative_mean_log_cohesion', 0):.4f}")
                
                if all_stats:
                    basic_info = all_stats.get('basic_stats', {})
                    clusters_info = all_stats.get('clusters', {})
                    cohesion_info = all_stats.get('cohesion', {})
                    log_cohesion_info = all_stats.get('log_cohesion', {})
                    print(f"   All samples: {basic_info.get('total_samples_analyzed', 0)} samples, {clusters_info.get('mean_clusters', 0):.2f} clusters")
                    print(f"   Cohesion: {cohesion_info.get('mean_cohesion', 0):.4f}, Log Cohesion: {log_cohesion_info.get('mean_log_cohesion', 0):.4f}")
                
                processed_datasets.append(dataset)
                
            except Exception as e:
                print(f"❌ Error processing {dataset}: {str(e)}")
        else:
            print(f"⚠️  Input file not found: {input_path}")

    # 2. 多数据集对比分析
    if len(processed_datasets) >= 2:
        print(f"\n📊 Starting dataset comparison analysis...")
        dataset_results = load_dataset_results()
        
        if len(dataset_results) >= 2:
            plot_dataset_comparison(dataset_results)
            create_comparison_summary(dataset_results)
            print(f"📁 Comparison visualizations saved to {COMPARISON_VIS_DIR}/")
            
            # 打印对比摘要
            print(f"\n📋 Dataset Comparison Summary:")
            for key, result in dataset_results.items():
                config = result['config']
                pn_stats = result['data'].get('positive_negative_comparison', {}).get('overall_statistics', {})
                all_stats = result['data'].get('all_samples_analysis', {}).get('overall_statistics', {})
                print(f"   {config['display_name']}:")
                if pn_stats:
                    clusters_info = pn_stats.get('clusters', {})
                    cohesion_info = pn_stats.get('cohesion', {})
                    log_cohesion_info = pn_stats.get('log_cohesion', {})
                    print(f"     Pos/Neg Clusters: {clusters_info.get('positive_mean_clusters', 0):.2f} / {clusters_info.get('negative_mean_clusters', 0):.2f}")
                    print(f"     Pos/Neg Cohesion: {cohesion_info.get('positive_mean_cohesion', 0):.4f} / {cohesion_info.get('negative_mean_cohesion', 0):.4f}")
                    print(f"     Pos/Neg Log Cohesion: {log_cohesion_info.get('positive_mean_log_cohesion', 0):.4f} / {log_cohesion_info.get('negative_mean_log_cohesion', 0):.4f}")
                if all_stats:
                    basic_info = all_stats.get('basic_stats', {})
                    clusters_info = all_stats.get('clusters', {})
                    cohesion_info = all_stats.get('cohesion', {})
                    log_cohesion_info = all_stats.get('log_cohesion', {})
                    print(f"     All samples: {clusters_info.get('mean_clusters', 0):.2f} clusters ({basic_info.get('total_samples_analyzed', 0)} samples)")
                    print(f"     Cohesion: {cohesion_info.get('mean_cohesion', 0):.4f}, Log Cohesion: {log_cohesion_info.get('mean_log_cohesion', 0):.4f}")
        else:
            print("⚠️  需要至少2个数据集结果文件进行对比分析")
    else:
        print("⚠️  处理成功的数据集不足2个，跳过对比分析")
    
    # 3. 总结
    print(f"\n🎉 Analysis completed!")
    print(f"   Processed datasets: {', '.join(processed_datasets)}")
    print(f"   Results saved to: ./results/")
    if len(processed_datasets) >= 2:
        print(f"   Comparison results: ./results/comparison_{NAME}_2/")
    print(f"\n🔧 Key Improvements:")
    print(f"   - Fixed cluster count bias: Using optimal_k inertia instead of min_inertia")
    print(f"   - Log Cohesion: ln(cluster_cohesion) = ln(1/(1+optimal_inertia))")
    print(f"   - Enhanced visualizations with model name labels")
    print(f"   - More accurate cohesion measurements")

if __name__ == "__main__":
    main()