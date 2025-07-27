import os
import json
import numpy as np
from collections import defaultdict

BASE_DIR = "./output/step_cross"

def find_json_files(folder, prefix):
    """查找指定前缀的json文件"""
    for fname in os.listdir(folder):
        if fname.startswith(prefix) and fname.endswith(".json"):
            return os.path.join(folder, fname)
    return None

def stat_list(arr):
    arr = np.array(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(len(arr))
    }

summary = []
cohesion_stats = {
    "error": [],
    "fix": [],
    "neg_normal": [],
    "pos_normal": []
}

for subdir in os.listdir(BASE_DIR):
    folder = os.path.join(BASE_DIR, subdir)
    if not os.path.isdir(folder):
        continue

    # 1. 读取特征重要性文件
    imp_path = find_json_files(folder, "geometric_feature_importance_")
    geo_path = find_json_files(folder, "geometric_features_")
    if not imp_path or not geo_path:
        continue

    with open(imp_path, "r") as f:
        imp_data = json.load(f)
    with open(geo_path, "r") as f:
        geo_data = json.load(f)

    # 1.1 最有区分度的指标
    if imp_data["feature_importance"]:
        top_metric = imp_data["feature_importance"][0]
        summary.append({
            "folder": subdir,
            "top_metric": top_metric["metric"],
            "effect_size": top_metric["effect_size"],
            "direction": top_metric["direction"]
        })

    # 2. cluster_cohesion 统计
    for region, key in [
        ("error_regions", "error"),
        ("fix_regions", "fix"),
        ("negative_normal", "neg_normal"),
        ("positive_normal", "pos_normal")
    ]:
        for step in geo_data.get(region, []):
            val = step["metrics"].get("cluster_cohesion")
            if val is not None:
                cohesion_stats[key].append(val)

# 1. 汇总最有区分度指标
print("=== 各文件夹最有区分度的指标 ===")
for item in summary:
    print(f"{item['folder']}: {item['top_metric']} (effect_size={item['effect_size']:.3f}, direction={item['direction']:.3f})")

# 1.1 统计 effect_size
effect_sizes = [item["effect_size"] for item in summary]
print("\n=== 最有区分度指标的 effect_size 统计 ===")
print(stat_list(effect_sizes))

# 统计每个指标的 effect_size
metric_effects = defaultdict(list)
for item in summary:
    metric = item['top_metric']
    metric_effects[metric].append(item['effect_size'])

print("\n=== 各指标作为最有区分度指标的统计 ===")
print(f"{'metric':30s} | {'count':>5s} | {'mean':>8s} | {'std':>8s} | {'min':>8s} | {'max':>8s}")
print("-"*75)
for metric, vals in sorted(metric_effects.items(), key=lambda x: -len(x[1])):
    arr = np.array(vals)
    print(f"{metric:30s} | {len(arr):5d} | {np.mean(arr):8.4f} | {np.std(arr):8.4f} | {np.min(arr):8.4f} | {np.max(arr):8.4f}")

# 2. cluster_cohesion 统计
print("\n=== cluster_cohesion 统计 ===")
for region in cohesion_stats:
    print(f"{region}: {stat_list(cohesion_stats[region])}")

# 统计所有指标（不只是top1）的 effect_size 分布
all_metric_effects = defaultdict(list)

for subdir in os.listdir(BASE_DIR):
    folder = os.path.join(BASE_DIR, subdir)
    if not os.path.isdir(folder):
        continue

    imp_path = find_json_files(folder, "geometric_feature_importance_")
    if not imp_path:
        continue

    with open(imp_path, "r") as f:
        imp_data = json.load(f)

    for metric_info in imp_data.get("feature_importance", []):
        metric = metric_info["metric"]
        effect_size = metric_info["effect_size"]
        all_metric_effects[metric].append(effect_size)

print("\n=== 所有指标 effect_size 统计（所有rank） ===")
print(f"{'metric':30s} | {'count':>5s} | {'mean':>8s} | {'std':>8s} | {'min':>8s} | {'max':>8s}")
print("-"*75)
for metric, vals in sorted(all_metric_effects.items(), key=lambda x: -len(x[1])):
    arr = np.array(vals)
    print(f"{metric:30s} | {len(arr):5d} | {np.mean(arr):8.4f} | {np.std(arr):8.4f} | {np.min(arr):8.4f} | {np.max(arr):8.4f}")
