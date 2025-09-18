import json
import numpy as np

INPUT_PATH = "./logits/bb_lies_mistral-nemo-12b_batch_7.json"

def clean_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

counts = []
for qid, qdata in data["results"].items():
    for sid, sample in qdata["samplings"].items():
        vecs = [clean_vector(t["hidden_vector"]) for t in sample.get("token_details", [])
                if isinstance(t.get("hidden_vector"), list) and len(t["hidden_vector"]) > 0]
        counts.append(len(vecs))

print(f"总样本数: {len(counts)}")
print(f"平均有效向量数: {np.mean(counts):.2f}")
print(f"有效向量 >=2 的样本数: {sum(c >= 2 for c in counts)}")
