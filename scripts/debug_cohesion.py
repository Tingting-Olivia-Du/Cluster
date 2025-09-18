# scripts/debug_cohesion.py
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_PATH = "./logits/bb_lies_mistral-nemo-12b_batch_7.json"

def clean_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def collect_vectors(sample):
    return [clean_vector(t["hidden_vector"]) for t in sample.get("token_details", [])
            if isinstance(t.get("hidden_vector"), list) and len(t["hidden_vector"]) > 0]

def compute_cluster_cohesion(vectors):
    if len(vectors) < 2:
        print("[WARN] Not enough vectors")
        return None
    vectors = np.array(vectors)
    centroid = np.mean(vectors, axis=0)
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

    min_inertia = float(np.min(inertias))
    cluster_cohesion = float(1 / (1 + min_inertia))
    best_silhouette = float(np.max(silhouettes))

    print(f"[DEBUG] n_tokens={n_tokens}, min_inertia={min_inertia:.4f}, cohesion={cluster_cohesion:.6f}, best_silhouette={best_silhouette:.4f}")
    return cluster_cohesion

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    count_pairs = 0
    for qid, qdata in data["results"].items():
        pos_samples = [(sid, s) for sid, s in qdata["samplings"].items() if s.get("is_correct", False)]
        neg_samples = [(sid, s) for sid, s in qdata["samplings"].items() if not s.get("is_correct", False)]

        if not pos_samples or not neg_samples:
            continue

        pos_vecs = collect_vectors(pos_samples[0][1])
        neg_vecs = collect_vectors(neg_samples[0][1])

        print(f"\n[PAIR] QID={qid} Positive sample:")
        compute_cluster_cohesion(pos_vecs)
        print(f"[PAIR] QID={qid} Negative sample:")
        compute_cluster_cohesion(neg_vecs)

        count_pairs += 1
        if count_pairs >= 2:  # 只跑前 2 对
            break

if __name__ == "__main__":
    main()
