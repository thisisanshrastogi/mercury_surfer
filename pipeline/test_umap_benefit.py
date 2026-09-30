import pandas as pd
import numpy as np
import umap.umap_ as umap
import hdbscan
import argparse
import time

def run_experiment(input_csv, input_npy):
    print(f"\n--- Loading Dataset ---")
    df = pd.read_csv(input_csv)
    embeddings = np.load(input_npy)
    print(f"Total reviews: {len(df)}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    if len(embeddings) < 15:
        print("Not enough data to run this experiment. (Needs at least 15 rows).")
        return

    # ==========================================
    # Approach 1: Direct HDBSCAN (768D)
    # ==========================================
    print("\n[Approach 1] Running Direct HDBSCAN on 768D Embeddings...")
    start_time = time.time()
    
    hdbscan_direct = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    direct_clusters = hdbscan_direct.fit_predict(embeddings)
    
    num_direct = len(set(direct_clusters)) - (1 if -1 in direct_clusters else 0)
    noise_direct = list(direct_clusters).count(-1)
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print(f"Clusters found: {num_direct}")
    print(f"Noise points (outliers): {noise_direct}")
    
    # ==========================================
    # Approach 2: UMAP -> HDBSCAN (5D)
    # ==========================================
    print("\n[Approach 2] Running UMAP (5D) + HDBSCAN...")
    start_time = time.time()
    
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5, 
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    hdbscan_umap = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    umap_clusters = hdbscan_umap.fit_predict(reduced_embeddings)
    
    num_umap = len(set(umap_clusters)) - (1 if -1 in umap_clusters else 0)
    noise_umap = list(umap_clusters).count(-1)
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print(f"Clusters found: {num_umap}")
    print(f"Noise points (outliers): {noise_umap}")

    # ==========================================
    # Qualitative Sample Comparison
    # ==========================================
    print("\n--- Qualitative Comparison ---")
    df['cluster_direct'] = direct_clusters
    df['cluster_umap'] = umap_clusters
    
    print("\n[Approach 1 - Direct HDBSCAN] Sample from each cluster:")
    for c_id in sorted(df['cluster_direct'].unique()):
        if c_id == -1: continue
        sample = df[df['cluster_direct'] == c_id].head(3)
        print(f"\nCluster {c_id} ({len(df[df['cluster_direct'] == c_id])} items):")
        for rev in sample['Review']:
            print(f"  - {str(rev)[:100]}...")
            
    print("\n[Approach 2 - UMAP -> HDBSCAN] Sample from each cluster:")
    for c_id in sorted(df['cluster_umap'].unique()):
        if c_id == -1: continue
        sample = df[df['cluster_umap'] == c_id].head(3)
        print(f"\nCluster {c_id} ({len(df[df['cluster_umap'] == c_id])} items):")
        for rev in sample['Review']:
            print(f"  - {str(rev)[:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--input-npy", required=True)
    args = parser.parse_args()
    
    run_experiment(args.input_csv, args.input_npy)
