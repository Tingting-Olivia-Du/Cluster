import os
import json
import numpy as np
from sklearn.metrics import classification_report

# ✅ 路径设置
# BASE_PATH = "/Users/tdu/Documents/GitHub/Cluster"
# range_tag = "0-10"
# DYNAMIC_PATH = f"./output/sequential_dynamics/deepseek-math-7b-math-{range_tag}-algebra-level_5/sequential_dynamics_{range_tag}.json"
# SAVE_PATH = f"./output/sequential_dynamics/deepseek-math-7b-math-{range_tag}-algebra-level_5/detector_results_{range_tag}.json"

range_tag = "901-950"
DYNAMIC_PATH = f"./output/sequential_dynamics/deepseek-math-7b-gsm-{range_tag}/sequential_dynamics_{range_tag}.json"
SAVE_PATH = f"./output/sequential_dynamics/deepseek-math-7b-gsm-{range_tag}/detector_results_{range_tag}.json"

# range_tag = "700-731"
# DYNAMIC_PATH = f"./output/sequential_dynamics/deepseek7b-gsm-{range_tag}/sequential_dynamics_{range_tag}.json"
# SAVE_PATH = f"./output/sequential_dynamics/deepseek7b-gsm-{range_tag}/detector_results_{range_tag}.json"

#zebra
# range_tag = "11-30"
# DYNAMIC_PATH = f"./output/sequential_dynamics/deepseek-math-7b-zebralogic-{range_tag}/sequential_dynamics_{range_tag}.json"
# SAVE_PATH = f"./output/sequential_dynamics/deepseek-math-7b-zebralogic-{range_tag}/detector_results_{range_tag}.json"
# ✅ 加载数据
with open(DYNAMIC_PATH, "r") as f:
    sequential_data = json.load(f)

# ✅ 计算error steps的entropy_change_mean阈值
error_entropy_values = []
for step in sequential_data["error_regions"]:
    if step.get("is_error_step", False):
        entropy_change_mean = step["metrics"].get("entropy_change_mean", 0.0)
        error_entropy_values.append(entropy_change_mean)

# 计算error steps的entropy_change_mean平均值作为阈值
ENTROPY_THRESH = np.mean(error_entropy_values) if error_entropy_values else 0.3
# ENTROPY_THRESH = 0.4
print(f"📊 Error steps entropy_change_mean values: {error_entropy_values}")
print(f"📊 Calculated threshold (mean): {ENTROPY_THRESH:.4f}")

# ✅ Step-based 模式
step_based_preds = {}

for step in sequential_data["error_regions"] + sequential_data["fix_regions"]:
    qid, sid = step["qid"], step["sid"]
    is_error_step = step.get("is_error_step", False)
    metrics = step["metrics"]
    entropy_change_mean = metrics.get("entropy_change_mean", 0.0)

    # 使用entropy_change_mean作为预测指标
    pred_is_error = (entropy_change_mean >= ENTROPY_THRESH)

    key = f"{qid}_{sid}"
    if key not in step_based_preds:
        step_based_preds[key] = {"pred_is_error": [], "true_is_error": False}

    step_based_preds[key]["pred_is_error"].append(pred_is_error)

    if is_error_step:
        step_based_preds[key]["true_is_error"] = True

# ✅ 汇总 per sampling 的预测结果
sampling_predictions = []
for key, info in step_based_preds.items():
    # 任意一个 step 判定为 error，就认为这个 sampling 判定为 error
    pred_is_error = any(info["pred_is_error"])
    true_is_error = info["true_is_error"]
    sampling_predictions.append({
        "key": key,
        "pred_is_error": pred_is_error,
        "true_is_error": true_is_error
    })

# ✅ 分类报告
y_true = [x["true_is_error"] for x in sampling_predictions]
y_pred = [x["pred_is_error"] for x in sampling_predictions]

print(f"\n📊 Step-based Detector Performance (entropy_change_mean threshold: {ENTROPY_THRESH:.4f}):")
print(classification_report(y_true, y_pred, digits=3))

# ✅ 详细统计
total_samples = len(sampling_predictions)
error_samples = sum(1 for x in sampling_predictions if x["true_is_error"])
correct_samples = sum(1 for x in sampling_predictions if x["pred_is_error"] == x["true_is_error"])

print(f"\n📈 Detailed Statistics:")
print(f"Total samples: {total_samples}")
print(f"Error samples: {error_samples}")
print(f"Correct predictions: {correct_samples}")
print(f"Accuracy: {correct_samples/total_samples:.3f}")

# ✅ 保存结果
with open(SAVE_PATH, "w") as f:
    json.dump({
        "threshold": ENTROPY_THRESH,
        "error_entropy_values": error_entropy_values,
        "predictions": sampling_predictions,
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }, f, indent=2)

print(f"\n✅ Results saved to: {SAVE_PATH}")

# ✅ 额外分析：不同阈值的效果
print(f"\n🔍 Threshold Analysis:")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, ENTROPY_THRESH]
for thresh in thresholds:
    # 重新计算预测
    temp_predictions = []
    for key, info in step_based_preds.items():
        pred_is_error = any(entropy >= thresh for entropy in 
                           [step["metrics"].get("entropy_change_mean", 0.0) 
                            for step in sequential_data["error_regions"] + sequential_data["fix_regions"]
                            if step["qid"] + "_" + step["sid"] == key])
        true_is_error = info["true_is_error"]
        temp_predictions.append({
            "pred_is_error": pred_is_error,
            "true_is_error": true_is_error
        })
    
    y_true_temp = [x["true_is_error"] for x in temp_predictions]
    y_pred_temp = [x["pred_is_error"] for x in temp_predictions]
    
    correct = sum(1 for x in temp_predictions if x["pred_is_error"] == x["true_is_error"])
    accuracy = correct / len(temp_predictions)
    
    print(f"Threshold {thresh:.3f}: Accuracy = {accuracy:.3f}")