import os
import subprocess
import argparse

def run_subprocess(cmd_args, env=None):
    print(f"\n--- Running: {' '.join(cmd_args)} ---")
    try:
        subprocess.run(cmd_args, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing pipeline stage: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end sentiment and clustering pipeline.")
    parser.add_argument("--input", required=True, help="Raw input CSV of app reviews")
    parser.add_argument("--output-dir", default="./pipeline_outputs", help="Directory for pipeline structured outputs")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for testing")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    stage1_output = os.path.join(args.output_dir, "01_jev_sentiment.csv")
    stage2_output_csv = os.path.join(args.output_dir, "02_embeddings.csv")
    stage2_output_npy = os.path.join(args.output_dir, "02_embeddings.npy")
    stage3_output = os.path.join(args.output_dir, "03_hdbscan_clusters.csv")

    # Load variables from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Inherit current environment but ensure it has required keys
    env = os.environ.copy()
    
    # Map JEV_API_KEY to TYPESAFE_API_KEY if present
    if "TYPESAFE_API_KEY" not in env and "JEV_API_KEY" in env:
        env["TYPESAFE_API_KEY"] = env["JEV_API_KEY"]

    if "TYPESAFE_API_KEY" not in env:
        print("Warning: TYPESAFE_API_KEY (or JEV_API_KEY) not found in .env. Stage 1 may fail.")
    if "GEMINI_API_KEY" not in env:
        print("\nERROR: GEMINI_API_KEY not found in .env!")
        print("Please add GEMINI_API_KEY=your_key_here to the .env file.")
        print("Stage 2 requires this key. Exiting.\n")
        return
        
    # Get current python executable
    import sys
    python_exe = sys.executable

    # Run Stage 1
    cmd_stage1 = [
        python_exe, "pipeline/stage1_sentiment.py",
        "--input", args.input,
        "--output", stage1_output
    ]
    if args.sample:
        cmd_stage1.extend(["--sample", str(args.sample)])
        
    run_subprocess(cmd_stage1, env=env)
    
    sentiments_to_analyze = ["negative", "neutral", "positive"]
    final_outputs = []
    
    for sentiment in sentiments_to_analyze:
        print(f"\n===========================================")
        print(f" Processing Sentiment: {sentiment.upper()}")
        print(f"===========================================")
        
        stage2_output_csv = os.path.join(args.output_dir, f"02_embeddings_{sentiment}.csv")
        stage2_output_npy = os.path.join(args.output_dir, f"02_embeddings_{sentiment}.npy")
        stage3_output = os.path.join(args.output_dir, f"03_hdbscan_clusters_{sentiment}.csv")
        
        # Run Stage 2
        cmd_stage2 = [
            python_exe, "pipeline/stage2_embed.py",
            "--input", stage1_output,
            "--output-csv", stage2_output_csv,
            "--output-npy", stage2_output_npy,
            "--filter-sentiment", sentiment
        ]
        run_subprocess(cmd_stage2, env=env)
        
        # Run Stage 3
        cmd_stage3 = [
            python_exe, "pipeline/stage3_cluster.py",
            "--input-csv", stage2_output_csv,
            "--input-npy", stage2_output_npy,
            "--output-csv", stage3_output
        ]
        run_subprocess(cmd_stage3, env=env)
        
        final_outputs.append(stage3_output)
        
    print("\n===========================================")
    print(" Generating final JSON result...")
    final_json_path = os.path.join(args.output_dir, "final_analysis.json")
    
    import json
    import pandas as pd
    
    final_json = {}
    
    for sentiment, out_file in zip(sentiments_to_analyze, final_outputs):
        if os.path.exists(out_file):
            df = pd.read_csv(out_file)
            if df.empty:
                final_json[sentiment] = {"clusters": []}
                continue
                
            sentiment_data = {"clusters": []}
            
            # Group by cluster ID
            # cluster -1 is noise
            for cluster_id, group in df.groupby("hdbscan_cluster_id"):
                cluster_id = int(cluster_id)
                cluster_strength = float(group["cluster_probability"].mean()) if cluster_id != -1 else 0.0
                
                # Convert group rows to list of dicts
                reviews_list = group.to_dict(orient="records")
                
                sentiment_data["clusters"].append({
                    "cluster_id": cluster_id,
                    "is_noise": cluster_id == -1,
                    "cluster_size": len(group),
                    "cluster_strength": round(cluster_strength, 4),
                    "reviews": reviews_list
                })
                
            final_json[sentiment] = sentiment_data
            
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    print("\n===========================================")
    print(f"Pipeline completed successfully for all sentiments!")
    print(f"Final structured JSON available at: {final_json_path}")
    print(f"Final CSV outputs available at:")
    for out_file in final_outputs:
        print(f" - {out_file}")
    print("===========================================")

if __name__ == "__main__":
    main()
