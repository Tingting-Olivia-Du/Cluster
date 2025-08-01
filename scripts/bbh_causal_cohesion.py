# -*- coding: utf-8 -*-
"""
bbh_causal_cohesion.py

Causal Judgment Cluster Cohesion Analysis (Sample-wise)
Adapted for new BBH causal judgment data structure
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_PATH = "./logits/bbh_causal_judgment_experiment.json"
OUTPUT_DIR = "./output/bbh_causal_judgment_analysis"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

print("🚀 Starting Causal Judgment Cluster Cohesion Analysis (Sample-wise)...")
print(f"📂 Input path: {INPUT_PATH}")
print(f"📁 Output dir: {OUTPUT_DIR}")

# ============================================================================
# HELPERS
# ============================================================================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_cluster_cohesion(vectors):
    if len(vectors) < 2:
        return dict(
            cluster_cohesion=0.0,
            optimal_clusters=1,
            min_inertia=0.0,
            best_silhouette_score=0.0,
            n_tokens=len(vectors),
            centroid_distance_mean=0.0,
            centroid_distance_std=0.0
        )
    X = np.array(vectors)
    centroid = X.mean(axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    inertias, sils = [], []
    max_k = min(5, len(X))
    for k in range(1, max_k+1):
        if k == 1:
            inertia = np.sum(dists**2)
            inertias.append(inertia); sils.append(0.0)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, labels) if len(set(labels))>1 else 0.0)
    best_k = int(np.argmax(sils)+1)
    best_sil = float(np.max(sils))
    min_iner = float(np.min(inertias))
    cohesion = float(1.0/(1.0+min_iner))
    return dict(
        cluster_cohesion=cohesion,
        optimal_clusters=best_k,
        min_inertia=min_iner,
        best_silhouette_score=best_sil,
        n_tokens=len(X),
        centroid_distance_mean=float(dists.mean()),
        centroid_distance_std=float(dists.std())
    )

# ============================================================================
# EXTRACT POS/NEG PAIRS
# ============================================================================
def extract_pairs(data):
    pairs=[]
    for qid, qd in data.get('results', {}).items():
        true_ans = qd.get('target')
        samps = qd.get('samplings', {})
        pos, neg = None, None
        for sid, sm in samps.items():
            if 'is_correct' not in sm or 'tokens' not in sm:
                continue
            if sm['is_correct'] and pos is None:
                pos = sm
            if not sm['is_correct'] and neg is None:
                neg = sm
            if pos and neg: break
        if pos and neg:
            pairs.append(dict(qid=qid, true_ans=true_ans, positive=pos, negative=neg))
    print(f"✅ Found {len(pairs)} positive-negative pairs")
    return pairs

# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_sample(sm, label):
    vecs=[tok['hidden_vector'] for tok in sm.get('tokens', []) if 'hidden_vector' in tok]
    metrics = compute_cluster_cohesion(vecs)
    metrics['label']=label
    return metrics


def analyze_pairs(pairs):
    results = {'pairs':[], 'summary':{'pos':[], 'neg':[]}}
    for p in pairs:
        pos_m = analyze_sample(p['positive'], 'pos')
        neg_m = analyze_sample(p['negative'], 'neg')
        diff = pos_m['cluster_cohesion'] - neg_m['cluster_cohesion']
        results['pairs'].append(dict(
            qid=p['qid'], true_ans=p['true_ans'],
            pos=pos_m, neg=neg_m, diff=diff
        ))
        if pos_m['n_tokens']>=2: results['summary']['pos'].append(pos_m['cluster_cohesion'])
        if neg_m['n_tokens']>=2: results['summary']['neg'].append(neg_m['cluster_cohesion'])
    # overall
    pos_arr=np.array(results['summary']['pos'])
    neg_arr=np.array(results['summary']['neg'])
    if pos_arr.size and neg_arr.size:
        results['overall']=dict(
            pos_mean=pos_arr.mean(), pos_std=pos_arr.std(),
            neg_mean=neg_arr.mean(), neg_std=neg_arr.std(),
            mean_diff=float(pos_arr.mean()-neg_arr.mean()),
            effect_size=float((pos_arr.mean()-neg_arr.mean())/np.sqrt((pos_arr.var()+neg_arr.var())/2))
        )
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_overall_plot(summary):
    pos = summary['pos']
    neg = summary['neg']
    if not pos or not neg:
        return
    plt.figure(figsize=(8,6))
    data = [pos, neg]
    labels = ['Positive','Negative']
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightgreen','lightcoral']
    for patch,color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Cluster Cohesion: Positive vs Negative')
    plt.ylabel('Cohesion')
    plt.grid(alpha=0.3)
    path = os.path.join(VIS_DIR,'overall_cohesion.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall plot: {path}")


def create_pairwise_plot(pairs):
    if not pairs:
        return
    qids=[p['qid'] for p in pairs]
    pos=[p['pos']['cluster_cohesion'] for p in pairs]
    neg=[p['neg']['cluster_cohesion'] for p in pairs]
    x = np.arange(len(qids))
    width=0.35
    plt.figure(figsize=(10,6))
    plt.bar(x-width/2, pos, width, label='Positive', color='lightgreen')
    plt.bar(x+width/2, neg, width, label='Negative', color='lightcoral')
    plt.xticks(x, qids, rotation=45)
    plt.ylabel('Cohesion')
    plt.title('Pairwise Cluster Cohesion')
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(VIS_DIR,'pairwise_cohesion.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved pairwise plot: {path}")


def create_distribution_plot(summary):
    pos = summary['pos']
    neg = summary['neg']
    if not pos or not neg:
        return
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(pos, bins=20, alpha=0.7, label='Pos', density=True, color='lightgreen')
    plt.hist(neg, bins=20, alpha=0.7, label='Neg', density=True, color='lightcoral')
    plt.title('Histogram')
    plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1,2,2)
    for arr,label,color in [(np.sort(pos),'Positive','green'),(np.sort(neg),'Negative','red')]:
        y = np.arange(1,len(arr)+1)/len(arr)
        plt.plot(arr,y,label=label,color=color)
    plt.title('CDF')
    plt.legend(); plt.grid(alpha=0.3)
    path = os.path.join(VIS_DIR,'distribution.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved distribution plot: {path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    data=load_data(INPUT_PATH)
    pairs=extract_pairs(data)
    if not pairs: return
    analysis=analyze_pairs(pairs)
    # save analysis
    out_file=os.path.join(OUTPUT_DIR,'cluster_cohesion_analysis.json')
    with open(out_file,'w',encoding='utf-8') as f:
        json.dump(analysis,f,indent=2,ensure_ascii=False)
    print(f"✅ Results saved: {out_file}")
    # visualizations
    create_overall_plot(analysis['summary'])
    create_pairwise_plot(analysis['pairs'])
    create_distribution_plot(analysis['summary'])

if __name__=='__main__':
    main()