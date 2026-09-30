import argparse
import pandas as pd
import numpy as np
import umap.umap_ as umap
import hdbscan

def run_clustering_stage(input_csv, input_npy, output_csv):
    print(f"\n--- Stage 3: UMAP and HDBSCAN Clustering ---")
    print(f"Reading filtered output from {input_csv} and embeddings from {input_npy}")
    
    df = pd.read_csv(input_csv)
    embeddings = np.load(input_npy)
    
    if df.empty or len(embeddings) == 0:
        print("No data to cluster. Exiting Stage 3.")
        return
        
    if len(embeddings) < 15:
        print(f"Not enough data to cluster (found {len(embeddings)} rows). Need at least 15. Skipping clustering for this subset.")
        for i in range(5):
            df[f'umap_dim_{i+1}'] = 0.0
            
        df['hdbscan_cluster_id'] = -1
        df['cluster_probability'] = 0.0
        df.to_csv(output_csv, index=False)
        return
        
    # 2. Dimensionality Reduction (UMAP)
    print("Running UMAP dimensionality reduction (768 -> 5)...")
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5, 
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    # 3. Clustering (HDBSCAN)
    print("Running HDBSCAN clustering...")
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    clusters = hdbscan_model.fit_predict(reduced_embeddings)
    probabilities = hdbscan_model.probabilities_
    
    # 4. Save Structured Results
    for i in range(5):
        df[f'umap_dim_{i+1}'] = reduced_embeddings[:, i]
        
    df['hdbscan_cluster_id'] = clusters
    df['cluster_probability'] = probabilities
    
    df.to_csv(output_csv, index=False)
    
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    outliers = list(clusters).count(-1)
    print(f"Stage 3 complete. Discovered {num_clusters} clusters. Outliers (noise): {outliers}")
    print(f"Saved structured output to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Path to input CSV from Stage 2")
    parser.add_argument("--input-npy", required=True, help="Path to input embeddings from Stage 2")
    parser.add_argument("--output-csv", required=True, help="Path to structured output CSV for Stage 3")
    args = parser.parse_args()
    
    run_clustering_stage(args.input_csv, args.input_npy, args.output_csv)
