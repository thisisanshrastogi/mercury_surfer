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
    parser.add_argument("--input", default=None, help="Raw input CSV of app reviews (required unless --rebuild-json-only is used)")
    parser.add_argument("--output-dir", default="./pipeline_outputs", help="Directory for pipeline structured outputs")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for testing")
    parser.add_argument("--skip-stage4", action="store_true", help="Skip Stage 4 topic modeling")
    parser.add_argument("--gemini-model", default="gemini-3.8-flash", help="Gemini model for Stage 4 topic synthesis")
    parser.add_argument("--skip-llm-topics", action="store_true", help="Skip LLM synthesis and only use c-TF-IDF")
    parser.add_argument("--stage4-batch-size", type=int, default=10, help="Number of clusters per Gemini call in Stage 4 (0 for all at once)")
    parser.add_argument("--stage4-sample-size", type=int, default=5, help="Number of exemplar reviews per cluster in Stage 4")
    parser.add_argument("--stage4-only", action="store_true", help="Only run Stage 4 topic modeling and generate final_analysis.json from existing Stage 3 outputs")
    parser.add_argument("--rebuild-json-only", action="store_true", help="Only rebuild final_analysis.json from existing files in output-dir")
    args = parser.parse_args()

    if not args.rebuild_json_only and not args.stage4_only and not args.input:
        parser.error("--input is required unless --rebuild-json-only or --stage4-only is specified.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    stage1_output = os.path.join(args.output_dir, "01_jev_sentiment.csv")

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

    sentiments_to_analyze = ["negative", "neutral", "positive"]
    final_outputs = []
    stage4_summaries = {}

    import sys
    python_exe = sys.executable

    if args.stage4_only:
        print("\n===========================================")
        print(" Running Stage 4 only on existing Stage 3 outputs...")
        print("===========================================")
        for sentiment in sentiments_to_analyze:
            stage3_output = os.path.join(args.output_dir, f"03_hdbscan_clusters_{sentiment}.csv")
            if not os.path.exists(stage3_output):
                print(f"Warning: {stage3_output} not found. Skipping {sentiment}.")
                continue
                
            stage4_output_csv = os.path.join(args.output_dir, f"04_topics_clusters_{sentiment}.csv")
            stage4_output_json = os.path.join(args.output_dir, f"04_topics_summary_{sentiment}.json")
            
            cmd_stage4 = [
                python_exe, "pipeline/stage4_topics.py",
                "--input-csv", stage3_output,
                "--output-csv", stage4_output_csv,
                "--output-json", stage4_output_json,
                "--sentiment", sentiment,
                "--model", args.gemini_model,
                "--batch-size", str(args.stage4_batch_size),
                "--sample-size", str(args.stage4_sample_size)
            ]
            if args.skip_llm_topics:
                cmd_stage4.append("--skip-llm")
                
            run_subprocess(cmd_stage4, env=env)
            final_outputs.append(stage4_output_csv)
            stage4_summaries[sentiment] = stage4_output_json

    elif not args.rebuild_json_only:
        if "TYPESAFE_API_KEY" not in env:
            print("Warning: TYPESAFE_API_KEY (or JEV_API_KEY) not found in .env. Stage 1 may fail.")
        if "GEMINI_API_KEY" not in env:
            print("\nERROR: GEMINI_API_KEY not found in .env!")
            print("Please add GEMINI_API_KEY=your_key_here to the .env file.")
            print("Stage 2 and Stage 4 require this key. Exiting.\n")
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
            
            # Run Stage 4 (Topic Modeling: Option A c-TF-IDF + Option C LLM)
            if not args.skip_stage4:
                stage4_output_csv = os.path.join(args.output_dir, f"04_topics_clusters_{sentiment}.csv")
                stage4_output_json = os.path.join(args.output_dir, f"04_topics_summary_{sentiment}.json")
                
                cmd_stage4 = [
                    python_exe, "pipeline/stage4_topics.py",
                    "--input-csv", stage3_output,
                    "--output-csv", stage4_output_csv,
                    "--output-json", stage4_output_json,
                    "--sentiment", sentiment,
                    "--model", args.gemini_model,
                    "--batch-size", str(args.stage4_batch_size),
                    "--sample-size", str(args.stage4_sample_size)
                ]
                if args.skip_llm_topics:
                    cmd_stage4.append("--skip-llm")
                    
                run_subprocess(cmd_stage4, env=env)
                final_outputs.append(stage4_output_csv)
                stage4_summaries[sentiment] = stage4_output_json
            else:
                final_outputs.append(stage3_output)
        
    print("\n===========================================")
    print(" Generating final JSON result...")
    final_json_path = os.path.join(args.output_dir, "final_analysis.json")
    
    import json
    import math
    import pandas as pd

    def sanitize_for_json(obj):
        """
        Recursively converts NaN, Infinity, and numpy types to strictly compliant JSON types.
        NaN -> None (rendered as null in JSON)
        """
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif hasattr(obj, "item"):
            val = obj.item()
            return sanitize_for_json(val)
        return obj
    
    final_json = {}
    
    for sentiment in sentiments_to_analyze:
        # Check for Stage 4 enriched CSV first, then Stage 3 cluster CSV
        csv_file = os.path.join(args.output_dir, f"04_topics_clusters_{sentiment}.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(args.output_dir, f"03_hdbscan_clusters_{sentiment}.csv")

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if df.empty:
                final_json[sentiment] = {"total_reviews": 0, "total_clusters": 0, "clusters": []}
                continue

            # Ensure text columns never contain NaN
            if "topic_keywords" in df.columns:
                df["topic_keywords"] = df["topic_keywords"].fillna("")
            if "topic_title" in df.columns:
                df["topic_title"] = df["topic_title"].fillna("")
            if "Review" in df.columns:
                df["Review"] = df["Review"].fillna("")
                
            # Load Stage 4 topic summaries from memory or disk
            summary_file = stage4_summaries.get(
                sentiment, 
                os.path.join(args.output_dir, f"04_topics_summary_{sentiment}.json")
            )
            topic_lookup = {}
            topics_overview = []
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        topics_overview = json.load(f)
                        for t in topics_overview:
                            topic_lookup[t["cluster_id"]] = t
                except Exception as e:
                    print(f"Warning: Failed to parse {summary_file}: {e}")

            clusters_list = []
            
            # Group by cluster ID (-1 is noise)
            for cluster_id, group in df.groupby("hdbscan_cluster_id"):
                cluster_id = int(cluster_id)
                cluster_strength = float(group["cluster_probability"].mean()) if cluster_id != -1 else 0.0
                
                # Sanitize reviews rows so no float NaN enters dictionaries
                raw_reviews = group.to_dict(orient="records")
                reviews_list = []
                for r in raw_reviews:
                    clean_r = {}
                    for k, v in r.items():
                        if pd.isna(v):
                            clean_r[k] = "" if isinstance(v, str) or k in ["topic_keywords", "topic_title", "Review", "Date"] else None
                        else:
                            clean_r[k] = v
                    reviews_list.append(clean_r)
                
                meta = topic_lookup.get(cluster_id, {})
                
                # Resolve topic title
                if meta.get("topic_title"):
                    topic_title = meta["topic_title"]
                elif "topic_title" in group.columns and pd.notna(group["topic_title"].iloc[0]) and str(group["topic_title"].iloc[0]).strip():
                    topic_title = str(group["topic_title"].iloc[0])
                elif cluster_id == -1:
                    topic_title = "Miscellaneous / Outliers"
                else:
                    topic_title = f"Topic {cluster_id}"

                # Resolve topic keywords
                if meta.get("keywords"):
                    topic_keywords = meta["keywords"]
                elif "topic_keywords" in group.columns and pd.notna(group["topic_keywords"].iloc[0]) and str(group["topic_keywords"].iloc[0]).strip():
                    topic_keywords = [k.strip() for k in str(group["topic_keywords"].iloc[0]).split(",") if k.strip()]
                else:
                    topic_keywords = []

                summary = meta.get(
                    "summary", 
                    "Individual outlier reviews." if cluster_id == -1 else "Discovered review cluster."
                )
                actionable_takeaway = meta.get(
                    "actionable_takeaway", 
                    "Monitor for emerging issues." if cluster_id == -1 else "Review customer feedback."
                )
                representative_reviews = meta.get(
                    "representative_reviews",
                    group.sort_values(by="cluster_probability", ascending=False)["Review"].dropna().head(args.stage4_sample_size).tolist() if "cluster_probability" in group.columns else []
                )

                cluster_info = {
                    "cluster_id": cluster_id,
                    "is_noise": cluster_id == -1,
                    "topic_title": topic_title,
                    "topic_keywords": topic_keywords,
                    "summary": summary,
                    "actionable_takeaway": actionable_takeaway,
                    "cluster_size": len(group),
                    "cluster_strength": round(cluster_strength, 4),
                    "representative_reviews": representative_reviews,
                    "reviews": reviews_list
                }
                clusters_list.append(cluster_info)
                
            non_noise_count = len([cid for cid in df["hdbscan_cluster_id"].unique() if cid != -1])
            noise_count = int((df["hdbscan_cluster_id"] == -1).sum())

            final_json[sentiment] = {
                "total_reviews": len(df),
                "total_clusters": non_noise_count,
                "noise_reviews": noise_count,
                "topics_overview": topics_overview,
                "clusters": clusters_list
            }
            
    # Strict JSON sanitization pass: convert any lingering NaN/Infinity to None
    clean_final_json = sanitize_for_json(final_json)

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_final_json, f, indent=2, ensure_ascii=False)
    
    print("\n===========================================")
    print(f"Pipeline completed successfully!")
    print(f"Final structured JSON available at: {final_json_path}")
    print("===========================================")

if __name__ == "__main__":
    main()
