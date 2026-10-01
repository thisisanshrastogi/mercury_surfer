import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def compute_c_tfidf(cluster_docs, ngram_range=(1, 2), top_k=8):
    """
    Computes class-based TF-IDF (c-TF-IDF) as used in BERTopic to extract
    top keywords for each cluster.
    
    cluster_docs: dict mapping cluster_id -> list of review strings
    """
    cluster_ids = list(cluster_docs.keys())
    # Concatenate all docs in each cluster into a single mega-document
    aggregated_texts = [" ".join(cluster_docs[cid]) for cid in cluster_ids]

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=1,
        max_features=10000
    )
    
    try:
        X = vectorizer.fit_transform(aggregated_texts)
    except ValueError:
        # If vocabulary is empty (e.g. all empty strings or stop words)
        return {cid: [] for cid in cluster_ids}

    feature_names = np.array(vectorizer.get_feature_names_out())
    tf = X.toarray()
    
    # Total frequency of each term across all classes
    f_t = tf.sum(axis=0)
    # Average frequency of words across classes
    A = f_t.sum() / max(tf.shape[0], 1)
    
    # Smoothed c-TF-IDF formula: tf * log(1 + A / f_t)
    idf = np.log(1 + (A / (f_t + 1e-9)))
    c_tfidf = tf * idf

    keywords_by_cluster = {}
    for idx, cid in enumerate(cluster_ids):
        top_indices = np.argsort(c_tfidf[idx])[::-1][:top_k]
        # Filter out any zero-score words
        top_words = [feature_names[i] for i in top_indices if c_tfidf[idx][i] > 0]
        keywords_by_cluster[cid] = top_words

    return keywords_by_cluster

from pydantic import BaseModel, Field

class ClusterTopicAnalysis(BaseModel):
    cluster_id: int = Field(description="The integer ID of the cluster analyzed")
    topic_title: str = Field(description="A concise, descriptive 3-5 word title summarizing this topic")
    summary: str = Field(description="1-2 sentence explanation describing the user issue, theme, or praise")
    actionable_takeaway: str = Field(description="1 concrete, actionable recommendation for product or engineering teams")

class ClusterBatchResponse(BaseModel):
    clusters: list[ClusterTopicAnalysis] = Field(description="List of topic analyses for each cluster in the batch")

def generate_llm_topic_summaries(cluster_data, sentiment, api_key, model_name="gemini-3.8-flash", batch_size=10, sample_size=5):
    """
    Uses Gemini with strict JSON Schema enforcement (via Pydantic) to synthesize
    c-TF-IDF keywords and representative reviews into structured topic summaries.
    Processes clusters in batches (default: 10 clusters per API call) to maximize speed.
    """
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    results = {}
    
    # Prepare batch chunks
    clusters_to_process = list(cluster_data.keys())
    effective_batch_size = len(clusters_to_process) if batch_size <= 0 else batch_size
    total_batches = (len(clusters_to_process) + effective_batch_size - 1) // effective_batch_size

    # Strict Schema Configuration
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ClusterBatchResponse,
        temperature=0.2
    )

    for b in tqdm(range(total_batches), desc=f"LLM topic synthesis ({sentiment})"):
        batch_ids = clusters_to_process[b * effective_batch_size : (b + 1) * effective_batch_size]
        
        prompt_payload = []
        for cid in batch_ids:
            item = cluster_data[cid]
            prompt_payload.append({
                "cluster_id": int(cid),
                "keywords": [str(k) for k in item["keywords"]],
                "sample_reviews": [str(r) for r in item["sample_reviews"][:sample_size]]
            })

        prompt = f"""
You are an expert product and user-feedback analyst.
Analyze the following clusters of app reviews with sentiment '{sentiment}'.
Each cluster contains top extracted c-TF-IDF keywords and {sample_size} exemplar user reviews.

Clusters:
{json.dumps(prompt_payload, indent=2, default=str)}

Analyze each cluster and return a list of cluster topic analyses matching the required schema.
"""
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            parsed_data = ClusterBatchResponse.model_validate_json(response.text.strip())

            for entry in parsed_data.clusters:
                results[entry.cluster_id] = {
                    "topic_title": entry.topic_title,
                    "summary": entry.summary,
                    "actionable_takeaway": entry.actionable_takeaway
                }
        except Exception as e:
            print(f"Warning: Batch {b+1}/{total_batches} LLM synthesis failed ({e}). Using fallback titles.")
            for cid in batch_ids:
                item = cluster_data[cid]
                kw_title = ", ".join(item["keywords"][:3]).title() if item["keywords"] else f"Cluster {cid}"
                results[cid] = {
                    "topic_title": kw_title,
                    "summary": f"User reviews centered on: {', '.join(item['keywords'])}",
                    "actionable_takeaway": "Review individual feedback for this cluster."
                }

    # Fill any missing clusters with fallback
    for cid in clusters_to_process:
        if cid not in results:
            item = cluster_data[cid]
            kw_title = ", ".join(item["keywords"][:3]).title() if item["keywords"] else f"Cluster {cid}"
            results[cid] = {
                "topic_title": kw_title,
                "summary": f"User reviews centered on: {', '.join(item['keywords'])}",
                "actionable_takeaway": "Review individual feedback for this cluster."
            }

    return results

def run_stage4(input_csv, output_csv, output_json, sentiment="all", 
               model_name="gemini-3.8-flash", top_k=8, sample_size=5, batch_size=10, skip_llm=False):
    print(f"\n--- Stage 4: Topic Modeling (c-TF-IDF + LLM Synthesis) ---")
    print(f"Reading clusters from: {input_csv}")
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} does not exist.")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    if df.empty or "hdbscan_cluster_id" not in df.columns:
        print("No cluster data available in input. Skipping Stage 4.")
        df.to_csv(output_csv, index=False)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    # Separate valid clusters from noise (-1)
    valid_clusters_df = df[df["hdbscan_cluster_id"] != -1].copy()
    unique_clusters = [int(cid) for cid in sorted(valid_clusters_df["hdbscan_cluster_id"].unique())]

    print(f"Discovered {len(unique_clusters)} distinct clusters (excluding noise -1).")

    cluster_docs = {}
    cluster_exemplars = {}

    for cid in unique_clusters:
        sub = valid_clusters_df[valid_clusters_df["hdbscan_cluster_id"] == cid]
        reviews = sub["Review"].dropna().astype(str).tolist()
        cluster_docs[cid] = reviews
        
        # Sort by cluster_probability descending to get best representative reviews
        if "cluster_probability" in sub.columns:
            top_sub = sub.sort_values(by="cluster_probability", ascending=False)
        else:
            top_sub = sub
        exemplar_reviews = top_sub["Review"].dropna().astype(str).head(sample_size).tolist()
        cluster_exemplars[cid] = exemplar_reviews

    # 1. Option A: Compute c-TF-IDF keywords
    print(f"Computing c-TF-IDF keywords for {len(unique_clusters)} clusters...")
    keywords_by_cluster = compute_c_tfidf(cluster_docs, top_k=top_k)

    # 2. Option C: LLM synthesis
    topic_metadata = {}
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not skip_llm and gemini_api_key and len(unique_clusters) > 0:
        cluster_payload = {}
        for cid in unique_clusters:
            cluster_payload[cid] = {
                "keywords": keywords_by_cluster.get(cid, []),
                "sample_reviews": cluster_exemplars.get(cid, [])
            }
        print(f"Synthesizing topic labels and insights with {model_name} (batch_size={batch_size}, sample_size={sample_size})...")
        topic_metadata = generate_llm_topic_summaries(
            cluster_payload, 
            sentiment=sentiment, 
            api_key=gemini_api_key, 
            model_name=model_name,
            batch_size=batch_size,
            sample_size=sample_size
        )
    else:
        if skip_llm:
            print("Skipping LLM synthesis as requested (--skip-llm). Using c-TF-IDF keywords for titles.")
        elif not gemini_api_key:
            print("Warning: GEMINI_API_KEY not found. Skipping LLM synthesis and using c-TF-IDF fallback.")
        
        for cid in unique_clusters:
            kws = keywords_by_cluster.get(cid, [])
            topic_metadata[cid] = {
                "topic_title": ", ".join(kws[:3]).title() if kws else f"Topic {cid}",
                "summary": f"Reviews primarily characterized by: {', '.join(kws)}",
                "actionable_takeaway": "Inspect cluster feedback for detailed product insight."
            }

    # Add metadata for noise cluster (-1)
    noise_count = (df["hdbscan_cluster_id"] == -1).sum()
    topic_metadata[-1] = {
        "topic_title": "Miscellaneous / Outliers",
        "summary": "Individual unclustered reviews with mixed, non-cohesive feedback.",
        "actionable_takeaway": "Monitor for new emerging themes as dataset grows."
    }
    keywords_by_cluster[-1] = []

    # 3. Enrich DataFrame
    df["topic_title"] = df["hdbscan_cluster_id"].apply(
        lambda cid: topic_metadata.get(int(cid), {}).get("topic_title", "Unknown")
    ).fillna("")
    df["topic_keywords"] = df["hdbscan_cluster_id"].apply(
        lambda cid: ", ".join(keywords_by_cluster.get(int(cid), []))
    ).fillna("")

    df.to_csv(output_csv, index=False)
    print(f"Saved enriched clusters CSV with topic titles to: {output_csv}")

    # 4. Generate structured summary JSON
    summary_list = []
    for cid in unique_clusters:
        sub = valid_clusters_df[valid_clusters_df["hdbscan_cluster_id"] == cid]
        avg_strength = float(sub["cluster_probability"].mean()) if "cluster_probability" in sub.columns else 0.0
        meta = topic_metadata.get(cid, {})
        
        summary_list.append({
            "cluster_id": int(cid),
            "topic_title": meta.get("topic_title", f"Topic {cid}"),
            "keywords": keywords_by_cluster.get(cid, []),
            "summary": meta.get("summary", ""),
            "actionable_takeaway": meta.get("actionable_takeaway", ""),
            "cluster_size": len(sub),
            "avg_cluster_strength": round(avg_strength, 4),
            "representative_reviews": cluster_exemplars.get(cid, [])
        })

    # Append noise overview
    if noise_count > 0:
        summary_list.append({
            "cluster_id": -1,
            "topic_title": "Miscellaneous / Outliers",
            "keywords": [],
            "summary": "Outlier reviews not belonging to any dense semantic cluster.",
            "actionable_takeaway": "Periodically re-cluster as new data is collected.",
            "cluster_size": int(noise_count),
            "avg_cluster_strength": 0.0,
            "representative_reviews": df[df["hdbscan_cluster_id"] == -1]["Review"].dropna().head(sample_size).tolist()
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False, default=str)
        
    print(f"Saved structured topics summary JSON to: {output_json}")
    print(f"Stage 4 complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: c-TF-IDF and LLM Topic Modeling")
    parser.add_argument("--input-csv", required=True, help="Path to 03_hdbscan_clusters_{sentiment}.csv")
    parser.add_argument("--output-csv", required=True, help="Path to save enriched CSV")
    parser.add_argument("--output-json", required=True, help="Path to save topic summary JSON")
    parser.add_argument("--sentiment", default="negative", help="Sentiment partition being processed")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Gemini model for topic synthesis")
    parser.add_argument("--top-k", type=int, default=8, help="Number of c-TF-IDF keywords per cluster")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of exemplar reviews per cluster")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of clusters per Gemini API call (0 for all at once)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM synthesis and only use c-TF-IDF")
    args = parser.parse_args()

    run_stage4(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        sentiment=args.sentiment,
        model_name=args.model,
        top_k=args.top_k,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        skip_llm=args.skip_llm
    )
