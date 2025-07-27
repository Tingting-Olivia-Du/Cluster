import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 分类定义
easy_sizes =  ['2x2', '2x3', '2x4', '2x5', '2x6', '3x2', '3x3', '3x4', '3x5', '4x2']
medium_sizes =  ['3x6', '4x3', '4x4', '5x2', '6x2']
hard_sizes =  ['4x5', '4x6', '5x3', '5x4', '5x5', '5x6', '6x3', '6x4', '6x5', '6x6']
size2difficulty = {}
for s in easy_sizes:
    size2difficulty[s] = 'easy'
for s in medium_sizes:
    size2difficulty[s] = 'medium'
for s in hard_sizes:
    size2difficulty[s] = 'hard'

# 2. 聚合统计
ZEBRA_STATS_DIR = './zebra_stats'
SAVE_DIR = './zebra_stats_difficulty'
os.makedirs(SAVE_DIR, exist_ok=True)

for metric in os.listdir(ZEBRA_STATS_DIR):
    metric_dir = os.path.join(ZEBRA_STATS_DIR, metric)
    if not os.path.isdir(metric_dir):
        continue
    # 聚合数据
    stats_by_difficulty = {'easy': {}, 'medium': {}, 'hard': {}}
    for fname in os.listdir(metric_dir):
        if not fname.endswith('.json'):
            continue
        size = fname.replace('.json', '')
        difficulty = size2difficulty.get(size)
        if not difficulty:
            continue
        with open(os.path.join(metric_dir, fname), 'r') as f:
            data = json.load(f)
        for region, stat in data.items():
            if region not in stats_by_difficulty[difficulty]:
                stats_by_difficulty[difficulty][region] = []
            if stat['mean'] is not None:
                stats_by_difficulty[difficulty][region].append(stat['mean'])
    # 统计
    summary = {}
    for difficulty, region_dict in stats_by_difficulty.items():
        summary[difficulty] = {}
        for region, values in region_dict.items():
            arr = np.array(values)
            summary[difficulty][region] = {
                'mean': float(np.mean(arr)) if len(arr) else None,
                'std': float(np.std(arr)) if len(arr) else None,
                'count': int(len(arr))
            }
    # 保存统计结果
    with open(os.path.join(SAVE_DIR, f'{metric}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    # 画图
    regions = ['error', 'fix', 'neg_normal', 'pos_normal']
    x = np.arange(len(regions))
    width = 0.2
    fig, ax = plt.subplots()
    for i, difficulty in enumerate(['easy', 'medium', 'hard']):
        means = [summary[difficulty][r]['mean'] if r in summary[difficulty] else 0 for r in regions]
        ax.bar(x + i*width, means, width, label=difficulty)
    ax.set_xticks(x + width)
    ax.set_xticklabels(regions)
    ax.set_ylabel('Mean')
    ax.set_title(f'{metric} by difficulty')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{metric}.png'))
    plt.close()