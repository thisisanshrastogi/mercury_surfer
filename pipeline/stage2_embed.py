import argparse
import os
import pandas as pd
import numpy as np
from google import genai
from tqdm import tqdm

def get_gemini_embeddings(texts, api_key):
    print("Extracting embeddings using Gemini API (gemini-embedding-2)...")
    client = genai.Client(api_key=api_key)
    
    all_embeddings = []
    
    for text in tqdm(texts, desc="Embedding reviews"):
        response = client.models.embed_content(
            model="gemini-embedding-2", 
            contents=text
        )
        # Assuming response.embeddings is a list of embeddings for the input
        all_embeddings.append(response.embeddings[0].values)
        
    return np.array(all_embeddings)

def run_embedding_stage(input_path, output_csv, output_npy, filter_sentiment=None):
    print(f"\n--- Stage 2: Embeddings ---")
    print(f"Reading structured output from {input_path}")
    
    df = pd.read_csv(input_path)
    
    if filter_sentiment:
        print(f"Filtering dataset to only include '{filter_sentiment}' sentiment...")
        df = df[df['jev_sentiment'] == filter_sentiment].copy()
        
    if df.empty:
        print(f"No rows left for sentiment '{filter_sentiment}'. Creating empty output files.")
        df.to_csv(output_csv, index=False)
        np.save(output_npy, np.array([]))
        return
        
    reviews = df['Review'].tolist()
    
    # 1. Embeddings
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
    embeddings = get_gemini_embeddings(reviews, gemini_api_key)
    
    df.to_csv(output_csv, index=False)
    np.save(output_npy, embeddings)
    
    print(f"Stage 2 complete. Saved filtered data to {output_csv} and embeddings to {output_npy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV from Stage 1")
    parser.add_argument("--output-csv", required=True, help="Path to structured output CSV for Stage 2")
    parser.add_argument("--output-npy", required=True, help="Path to save embeddings numpy array")
    parser.add_argument("--filter-sentiment", type=str, default="negative", help="Optional: filter by sentiment (e.g., negative) before clustering")
    args = parser.parse_args()
    
    run_embedding_stage(args.input, args.output_csv, args.output_npy, args.filter_sentiment)
