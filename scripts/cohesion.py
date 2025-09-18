# scripts/cohesion_final.py
# -*- coding: utf-8 -*-
"""
Causal Judgment Cluster Cohesion Analysis (Sample-wise)
最终修正版：
1. 有效样本筛选 + 自动找下一个可用正/负样本
2. NaN / Inf → 0，不丢 token
3. 向量归一化（StandardScaler）
4. inertia 除以 token 数防止溢出
5. 完整可视化 + 全量 JSON 输出
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
# NAME = "bbeh_lies_mistral-nemo-12b_batch_7"
# NAME = "bbeh_lies_ministral-8b_batch_5"
NAME = "bbeh_lies_mistral_batch_4"
INPUT_PATH = f"./logits/{NAME}.json"
OUTPUT_DIR = f"./output/{NAME}"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

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

# ---------------- COHESION ----------------
def compute_cluster_cohesion(vectors):
    if len(vectors) < 2:
        return None
    vectors = np.array(vectors)

    # 1) 标准化向量，防止 inertia 爆炸
    vectors = StandardScaler().fit_transform(vectors)

    # 2) 基本统计
    centroid = np.mean(vectors, axis=0)
    centroid_distances = [np.linalg.norm(v - centroid) for v in vectors]
    n_tokens = len(vectors)

    inertias, silhouettes = [], []
    for k in range(1, min(5, n_tokens) + 1):
        if k == 1:
            inertia = np.sum([np.linalg.norm(v - centroid) ** 2 for v in vectors])
            inertias.append(inertia)
            silhouettes.append(0.0)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(vectors, labels) if len(set(labels)) > 1 else 0.0)

    min_inertia = float(np.min(inertias) / n_tokens)  # 3) 除以 token 数
    cluster_cohesion = float(1 / (1 + min_inertia))
    best_silhouette = float(np.max(silhouettes))

    return {
        "cluster_cohesion": cluster_cohesion,
        "optimal_clusters": int(np.argmax(silhouettes) + 1),
        "min_inertia": min_inertia,
        "best_silhouette_score": best_silhouette,
        "n_tokens": n_tokens,
        "centroid_distance_mean": float(np.mean(centroid_distances)),
        "centroid_distance_std": float(np.std(centroid_distances))
    }

# ---------------- MAIN ANALYSIS ----------------
def analyze_pairs(data):
    results = {
        "pair_comparisons": [],
        "summary_statistics": {
            "total_pairs": 0,
            "valid_pairs": 0,
            "positive_stats": {"cluster_cohesion": []},
            "negative_stats": {"cluster_cohesion": []}
        }
    }

    for qid, qdata in data.get("results", {}).items():
        pos_samples = [(sid, s) for sid, s in qdata["samplings"].items() if s.get("is_correct", False)]
        neg_samples = [(sid, s) for sid, s in qdata["samplings"].items() if not s.get("is_correct", False)]

        pos_sid, pos_sample, pos_vecs = find_first_valid_sample(pos_samples)
        neg_sid, neg_sample, neg_vecs = find_first_valid_sample(neg_samples)

        if pos_sample and neg_sample:
            pos_metrics = compute_cluster_cohesion(pos_vecs)
            neg_metrics = compute_cluster_cohesion(neg_vecs)
            if pos_metrics and neg_metrics:
                results["summary_statistics"]["valid_pairs"] += 1
                results["summary_statistics"]["positive_stats"]["cluster_cohesion"].append(pos_metrics["cluster_cohesion"])
                results["summary_statistics"]["negative_stats"]["cluster_cohesion"].append(neg_metrics["cluster_cohesion"])
                results["pair_comparisons"].append({
                    "qid": qid,
                    "positive_analysis": pos_metrics,
                    "negative_analysis": neg_metrics,
                    "cohesion_difference": pos_metrics["cluster_cohesion"] - neg_metrics["cluster_cohesion"]
                })

        results["summary_statistics"]["total_pairs"] += 1

    pos = results["summary_statistics"]["positive_stats"]["cluster_cohesion"]
    neg = results["summary_statistics"]["negative_stats"]["cluster_cohesion"]
    if pos and neg:
        results["overall_comparison"] = {
            "positive_mean": float(np.mean(pos)),
            "negative_mean": float(np.mean(neg)),
            "mean_difference": float(np.mean(pos) - np.mean(neg)),
            "effect_size": float((np.mean(pos) - np.mean(neg)) /
                                 np.sqrt((np.var(pos) + np.var(neg)) / 2))
        }
    return results

# ---------------- VISUALIZATION ----------------
def plot_overall(results):
    pos = results["summary_statistics"]["positive_stats"]["cluster_cohesion"]
    neg = results["summary_statistics"]["negative_stats"]["cluster_cohesion"]
    if not pos or not neg:
        return
    plt.figure(figsize=(8, 6))
    plt.boxplot([pos, neg], labels=["Positive", "Negative"], patch_artist=True)
    plt.title("Cluster Cohesion: Positive vs Negative")
    plt.savefig(f"{VISUALIZATION_DIR}/overall_comparison.png", dpi=150)
    plt.close()

def plot_pairwise(results):
    pairs = results["pair_comparisons"]
    if not pairs:
        return
    pos_vals = [p["positive_analysis"]["cluster_cohesion"] for p in pairs]
    neg_vals = [p["negative_analysis"]["cluster_cohesion"] for p in pairs]
    labels = [p["qid"] for p in pairs]
    fig_width = min(max(len(labels) * 0.5, 8), 30)
    plt.figure(figsize=(fig_width, 6))
    x = np.arange(len(labels))
    plt.bar(x - 0.2, pos_vals, 0.4, label="Positive", color="lightgreen")
    plt.bar(x + 0.2, neg_vals, 0.4, label="Negative", color="lightcoral")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend()
    plt.title("Pairwise Cohesion Comparison")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/pairwise_comparison.png", dpi=150)
    plt.close()

# ---------------- MAIN ----------------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = analyze_pairs(data)

    json_path = f"{OUTPUT_DIR}/cluster_cohesion_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📁 Saved analysis JSON to {json_path}")

    plot_overall(results)
    plot_pairwise(results)

    print(f"✅ Positive mean: {results.get('overall_comparison', {}).get('positive_mean', 0):.4f}")
    print(f"✅ Negative mean: {results.get('overall_comparison', {}).get('negative_mean', 0):.4f}")

if __name__ == "__main__":
    main()
