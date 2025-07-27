import os
import json
import numpy as np

BASE_DIR = "./output/step_cross"

def find_json_files(folder, prefix):
    for fname in os.listdir(folder):
        if fname.startswith(prefix) and fname.endswith(".json"):
            return os.path.join(folder, fname)
    return None

# 收集各区域 convex_hull_area
area_stats = {
    "error": [],
    "fix": [],
    "neg_normal": [],
    "pos_normal": []
}

for subdir in os.listdir(BASE_DIR):
    folder = os.path.join(BASE_DIR, subdir)
    if not os.path.isdir(folder):
        continue

    geo_path = find_json_files(folder, "geometric_features_")
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
            val = step["metrics"].get("convex_hull_area")
            if val is not None:
                area_stats[key].append(val)

def stat_list(arr):
    arr = np.array(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(len(arr))
    }

print("=== convex_hull_area 各区域统计 ===")
for region in area_stats:
    print(f"{region:12s}: {stat_list(area_stats[region])}")