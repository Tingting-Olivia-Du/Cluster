import json

range_tag = "0-50"
input_json = f"./logits/deepseek-math-7b-math-{range_tag}.json"
output_json = f"./logits/deepseek-math-7b-math-{range_tag}_changed.json"

# 需要替换的 key 映射
key_map = {
    "true_final_answer": "true_final_result",
    "final_answer": "final_result",
    "problem": "question"
}

with open(input_json, "r") as f:
    data = json.load(f)

def replace_keys(obj):
    if isinstance(obj, dict):
        return {key_map.get(k, k): replace_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_keys(item) for item in obj]
    else:
        return obj

new_data = replace_keys(data)

with open(output_json, "w") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)