import os
import json
import re
import numpy as np

# 1. 解析 logits 文件，建立 qid -> size 的映射

def parse_size_from_id(id_str):
    m = re.search(r'(\d+)x(\d+)', id_str)
    if m:
        return f"{m.group(1)}x{m.group(2)}"
    return "unknown"

LOGITS_PATHS = [
    "./logits/deepseek-math-7b-zebralogic-0-5.jsonl",
    "./logits/deepseek-math-7b-zebralogic-6-10.jsonl",
    "./logits/deepseek-math-7b-zebralogic-11-30.jsonl",
    "./logits/deepseek-math-7b-zebralogic-0-5.json",
    "./logits/deepseek-math-7b-zebralogic-6-10.json",
    "./logits/deepseek-math-7b-zebralogic-11-30.json"
]
STEP_CROSS_DIR = "./output/step_cross"
SAVE_DIR = "./zebra_stats"
os.makedirs(SAVE_DIR, exist_ok=True)

qid2size = {}
for logits_path in LOGITS_PATHS:
    if not os.path.exists(logits_path):
        continue
    if logits_path.endswith('.jsonl'):
        with open(logits_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    qid = obj.get("qid") or obj.get("id")
                    sample = obj.get("data") if "data" in obj else obj
                    id_str = sample.get("id", "")
                    size = parse_size_from_id(id_str)
                    if qid:
                        qid2size[qid] = size
                except Exception:
                    continue
    else:
        with open(logits_path, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        for qid, sample in data.items():
            id_str = sample.get("id", "")
            size = parse_size_from_id(id_str)
            qid2size[qid] = size

# 2. 遍历 step_cross 目录下 zebralogic 相关子文件夹，统计所有指标
all_metrics = set()
area_stats_by_metric_size = {}

def add_stat(stats_dict, metric, size, region, value):
    if metric not in stats_dict:
        stats_dict[metric] = {}
    if size not in stats_dict[metric]:
        stats_dict[metric][size] = {"error": [], "fix": [], "neg_normal": [], "pos_normal": []}
    stats_dict[metric][size][region].append(value)

for subdir in os.listdir(STEP_CROSS_DIR):
    if "zebralogic" not in subdir:
        continue
    folder = os.path.join(STEP_CROSS_DIR, subdir)
    if not os.path.isdir(folder):
        continue
    geo_path = None
    for fname in os.listdir(folder):
        if fname.startswith("geometric_features_") and fname.endswith(".json"):
            geo_path = os.path.join(folder, fname)
            break
    if not geo_path:
        continue
    with open(geo_path, "r") as f:
        geo_data = json.load(f)
    for region, key in [
        ("error_regions", "error"),
        ("fix_regions", "fix"),
        ("negative_normal", "neg_normal"),
        ("positive_normal", "pos_normal")
    ]:
        for step in geo_data.get(region, []):
            qid = step.get("qid")
            size = qid2size.get(qid, "unknown")
            for metric, val in step.get("metrics", {}).items():
                if val is not None:
                    all_metrics.add(metric)
                    add_stat(area_stats_by_metric_size, metric, size, key, val)

# 3. 保存每个指标/size的统计结果

def stat_list(arr):
    arr = np.array(arr)
    return {
        "mean": float(np.mean(arr)) if len(arr) else None,
        "std": float(np.std(arr)) if len(arr) else None,
        "min": float(np.min(arr)) if len(arr) else None,
        "max": float(np.max(arr)) if len(arr) else None,
        "count": int(len(arr))
    }

for metric in sorted(all_metrics):
    metric_dir = os.path.join(SAVE_DIR, metric)
    os.makedirs(metric_dir, exist_ok=True)
    for size in sorted(area_stats_by_metric_size.get(metric, {})):
        stats = {}
        for region in ["error", "fix", "neg_normal", "pos_normal"]:
            stats[region] = stat_list(area_stats_by_metric_size[metric][size][region])
        save_path = os.path.join(metric_dir, f"{size}.json")
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved: {save_path} | {stats}")