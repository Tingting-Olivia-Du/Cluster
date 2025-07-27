import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 分类定义
categories = {
    'gsm_700_731': ['deepseek7b-gsm-700-731'],
    'gsm_901_950': ['deepseek-math-7b-gsm-901-950'],
    'algebra_level_3': ['deepseek-math-7b-math-0-15-algebra-level_3'],
    'algebra_level_4': ['deepseek-math-7b-math-0-15-algebra-level_4'],
    'algebra_level_5': ['deepseek-math-7b-math-0-10-algebra-level_5']
}

# 2. 设置路径
STEP_CROSS_DIR = './output/step_cross'
SAVE_DIR = './step_cross_stats3'
os.makedirs(SAVE_DIR, exist_ok=True)

# 3. 获取所有指标名称
def get_metric_names(data_dir):
    """从数据中提取所有指标名称"""
    for category_name, folder_names in categories.items():
        for folder_name in folder_names:
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.exists(folder_path):
                # 查找geometric_features文件
                for fname in os.listdir(folder_path):
                    if fname.startswith('geometric_features') and fname.endswith('.json'):
                        file_path = os.path.join(folder_path, fname)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if 'error_regions' in data and len(data['error_regions']) > 0:
                                # 获取第一个样本的指标名称
                                first_sample = data['error_regions'][0]
                                if 'metrics' in first_sample:
                                    return list(first_sample['metrics'].keys())
    return []

# 4. 提取数据
def extract_metrics(data_dir, category_name, folder_name):
    """从指定文件夹提取指标数据"""
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.exists(folder_path):
        return {}
    
    # 查找geometric_features文件
    features_file = None
    for fname in os.listdir(folder_path):
        if fname.startswith('geometric_features') and fname.endswith('.json'):
            features_file = os.path.join(folder_path, fname)
            break
    
    if not features_file:
        return {}
    
    with open(features_file, 'r') as f:
        data = json.load(f)
    
    # 按区域分组数据
    regions_data = {'error_regions': [], 'fix_regions': [], 'neg_normal_regions': [], 'pos_normal_regions': []}
    
    for region_type in regions_data.keys():
        if region_type in data:
            for item in data[region_type]:
                if 'metrics' in item:
                    regions_data[region_type].append(item['metrics'])
    
    return regions_data

# 5. 计算统计信息
def calculate_stats(values):
    """计算统计信息"""
    if not values:
        return {'mean': None, 'std': None, 'count': 0}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'count': int(len(arr))
    }

# 6. 主处理函数
def process_data():
    # 获取指标名称
    metric_names = get_metric_names(STEP_CROSS_DIR)
    print(f"Found {len(metric_names)} metrics: {metric_names}")
    
    # 为每个指标创建统计
    for metric in metric_names:
        print(f"Processing metric: {metric}")
        
        # 收集所有类别的数据
        category_stats = {}
        
        for category_name, folder_names in categories.items():
            category_stats[category_name] = {}
            
            for folder_name in folder_names:
                regions_data = extract_metrics(STEP_CROSS_DIR, category_name, folder_name)
                
                # 为每个区域计算统计
                for region_name, samples in regions_data.items():
                    if region_name not in category_stats[category_name]:
                        category_stats[category_name][region_name] = []
                    
                    # 提取当前指标的值
                    metric_values = []
                    for sample in samples:
                        if metric in sample:
                            metric_values.append(sample[metric])
                    
                    if metric_values:
                        category_stats[category_name][region_name].extend(metric_values)
        
        # 计算最终统计
        final_stats = {}
        for category_name, region_data in category_stats.items():
            final_stats[category_name] = {}
            for region_name, values in region_data.items():
                final_stats[category_name][region_name] = calculate_stats(values)
        
        # 保存统计结果
        with open(os.path.join(SAVE_DIR, f'{metric}.json'), 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # 创建可视化
        create_visualization(metric, final_stats)

def create_visualization(metric, stats):
    """创建可视化图表"""
    regions = ['error_regions', 'fix_regions', 'neg_normal_regions', 'pos_normal_regions']
    categories_list = list(stats.keys())
    
    # 准备数据
    means = []
    stds = []
    labels = []
    
    for category in categories_list:
        for region in regions:
            if region in stats[category] and stats[category][region]['mean'] is not None:
                means.append(stats[category][region]['mean'])
                stds.append(stats[category][region]['std'])
                labels.append(f"{category}\n{region}")
    
    if not means:
        print(f"No data available for metric: {metric}")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 柱状图
    x = np.arange(len(means))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax1.set_xlabel('Category & Region')
    ax1.set_ylabel('Mean Value')
    ax1.set_title(f'{metric} - Mean Values by Category and Region')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 为柱状图添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 热力图
    # 准备热力图数据
    heatmap_data = []
    heatmap_labels = []
    
    for category in categories_list:
        row = []
        for region in regions:
            if region in stats[category] and stats[category][region]['mean'] is not None:
                row.append(stats[category][region]['mean'])
            else:
                row.append(0)
        heatmap_data.append(row)
        heatmap_labels.append(category)
    
    if heatmap_data:
        heatmap_data = np.array(heatmap_data)
        im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(regions)))
        ax2.set_yticks(range(len(categories_list)))
        ax2.set_xticklabels([r.replace('_regions', '') for r in regions], rotation=45)
        ax2.set_yticklabels(heatmap_labels)
        ax2.set_title(f'{metric} - Heatmap')
        
        # 添加数值标签
        for i in range(len(categories_list)):
            for j in range(len(regions)):
                text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="white" if heatmap_data[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    process_data()
    print("Processing completed! Results saved to:", SAVE_DIR)