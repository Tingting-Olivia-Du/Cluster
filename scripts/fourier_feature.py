# -*- coding: utf-8 -*-
"""
feature_detector.py

端到端脚本：
1) 从 JSON 文件加载正负采样的 token-level 熵序列 和 错误/修正位置信息；
2) 提取时域 & 频域统计特征；
3) 拼接成分类特征向量并打标签（负样本=1，正样本=0）；
4) 训练 LogisticRegression 二分类器并评估性能。
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, correlate
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ---------- 配置项：本地路径 ----------
BASE_PATH      = "/Users/tdu/Documents/GitHub/Cluster"
start_index    = 700
end_index      = 731
range_tag      = f"{start_index}-{end_index}"
LOGITS_PATH    = f"{BASE_PATH}/logits/deepseek7b-gsm-{range_tag}.json"
INDEX_PATH     = f"{BASE_PATH}/error_fix_index/deepseek-7b-{range_tag}_error_fix_index.json"

# ---------- 特征提取函数 ----------
def extract_time_features(entropy):
    feats = {
        'mean': np.mean(entropy),
        'var': np.var(entropy),
        'skew': skew(entropy),
        'kurtosis': kurtosis(entropy),
        'max': np.max(entropy),
        'min': np.min(entropy)
    }
    # 峰值分析
    peaks, props = find_peaks(entropy, prominence=0.01, width=1)
    feats['peak_count']    = len(peaks)
    feats['peak_prom_avg'] = float(np.mean(props['prominences'])) if 'prominences' in props and len(props['prominences'])>0 else 0.0
    feats['peak_width_avg'] = float(np.mean(props['widths'])) if 'widths' in props and len(props['widths'])>0 else 0.0
    # 自相关
    if len(entropy) > 2:
        ac = correlate(entropy - np.mean(entropy), entropy - np.mean(entropy), mode='full')
        ac = ac[ac.size//2:] / ac[ac.size//2]
        feats["autocorr1"] = float(ac[1]) if len(ac)>1 else 0.0
        below = np.where(ac < 0.2)[0]
        feats["autocorr_decay"] = float(below[0]) if len(below)>0 else len(entropy)
    else:
        feats["autocorr1"] = 0.0
        feats["autocorr_decay"] = 0.0
    # 滑动窗口统计
    w = 5
    if len(entropy) >= w:
        local_means = [np.mean(entropy[i:i+w]) for i in range(len(entropy)-w+1)]
        local_vars  = [np.var(entropy[i:i+w]) for i in range(len(entropy)-w+1)]
        local_maxs  = [np.max(entropy[i:i+w]) for i in range(len(entropy)-w+1)]
        feats["win_mean_max"] = np.max(local_means)
        feats["win_mean_min"] = np.min(local_means)
        feats["win_mean_avg"] = np.mean(local_means)
        feats["win_var_max"]  = np.max(local_vars)
        feats["win_var_min"]  = np.min(local_vars)
        feats["win_var_avg"]  = np.mean(local_vars)
        feats["win_max_max"]  = np.max(local_maxs)
        feats["win_max_min"]  = np.min(local_maxs)
        feats["win_max_avg"]  = np.mean(local_maxs)
    else:
        for k in ["win_mean_max","win_mean_min","win_mean_avg","win_var_max","win_var_min","win_var_avg","win_max_max","win_max_min","win_max_avg"]:
            feats[k] = 0.0
    return feats

def extract_freq_features(entropy):
    N  = len(entropy)
    yf = fft(entropy)
    xf = fftfreq(N, d=1)[:N//2]
    amp = np.abs(yf)[:N//2]
    power = amp**2
    total = np.sum(power) + 1e-8

    feats = {}
    feats['spec_centroid']      = float(np.sum(xf * amp) / (np.sum(amp)+1e-8))
    feats['band_mid_high_ratio']= float(np.sum(power[xf>0.1]) / total)
    geo_mean = np.exp(np.mean(np.log(power+1e-8)))
    feats['spec_flatness']      = float(geo_mean / (np.mean(power)+1e-8))
    csum = np.cumsum(power)
    idx  = np.where(csum >= 0.85*total)[0][0]
    feats['spec_rolloff']       = float(xf[idx])
    p_norm = power/total
    feats['spec_entropy']       = float(-np.sum(p_norm * np.log2(p_norm+1e-8)))
    return feats

def extract_features(entropy):
    feats = {}
    feats.update(extract_time_features(entropy))
    feats.update(extract_freq_features(entropy))
    return feats

# ---------- 主流程 ----------
def main():
    # 1. 加载 JSON
    with open(LOGITS_PATH, 'r') as f:
        logits_data = json.load(f)
    with open(INDEX_PATH, 'r') as f:
        index_data  = json.load(f)

    # 2. 构建 sampling 对
    paired = []
    for qid, meta_dict in index_data.items():
        if qid not in logits_data:
            continue
        for neg_sid, meta in meta_dict.items():
            if not neg_sid.startswith("sampling"):
                continue
            pos_sid = meta.get("correct_sampling_id")
            if pos_sid in logits_data[qid]:
                paired.append((qid, neg_sid, pos_sid))
            else:
                print(f"⚠️ 跳过 {qid} | {neg_sid} vs {pos_sid} — 不存在正确采样")

    # 3. 特征提取
    records = []
    for qid, neg_sid, pos_sid in paired:
        try:
            neg_probs = logits_data[qid][neg_sid]["token_probs"]
            pos_probs = logits_data[qid][pos_sid]["token_probs"]
            neg_seq = [tok["topk_info"]["entropy"] for tok in neg_probs]
            pos_seq = [tok["topk_info"]["entropy"] for tok in pos_probs]
        except Exception as e:
            print(f"跳过 {qid} | {neg_sid} vs {pos_sid} — {e}")
            continue

        if len(neg_seq)<4 or len(pos_seq)<4:
            continue

        neg_feats = {f"neg_{k}":v for k,v in extract_features(neg_seq).items()}
        pos_feats = {f"pos_{k}":v for k,v in extract_features(pos_seq).items()}

        # 负样本=1，正样本=0
        records.append({**neg_feats, **pos_feats, "label":1})
        records.append({**pos_feats, **neg_feats, "label":0})

    df = pd.DataFrame(records)
    X  = df.drop(columns="label")
    y  = df["label"]

    # 4. 划分训练/测试
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 5. 标准化
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 6. 训练分类器
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_tr_s, y_tr)

    # 7. 评估
    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:,1]

    print("=== Classification Report ===")
    print(classification_report(y_te, y_pred, digits=4))
    print("=== ROC AUC Score ===")
    print(f"{roc_auc_score(y_te, y_prob):.4f}")

if __name__ == "__main__":
    main()