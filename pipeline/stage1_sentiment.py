import argparse
import os
import pandas as pd
from typesafe_sdk import TypeSafeClient, Choice
from tqdm import tqdm

def run_sentiment_stage(input_path, output_path, sample_size=None):
    print(f"--- Stage 1: Running TypeSafe Jev Sentiment Classification ---")
    print(f"Reading from {input_path}")
    
    df = pd.read_csv(input_path)
    if 'Review' not in df.columns:
        raise ValueError("Input CSV must contain a 'Review' column.")

    df = df[df['Review'].notnull()]
    
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} rows for pipeline speed...")
        df = df.sample(sample_size, random_state=42)

    # Initialize Jev client
    # Expects TYPESAFE_API_KEY in environment
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("Warning: TYPESAFE_API_KEY environment variable is not set.")
        
    results = []
    
    # Use the context manager as recommended by TypeSafe documentation
    with TypeSafeClient(api_key=api_key) as client:
        # Process each review
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying sentiment with Jev"):
            review_text = str(row['Review']).strip()
            if not review_text:
                continue
                
            try:
                # Jev System One prompt
                response = client.system_one(
                    state=review_text,
                    questions={
                        "sentiment": Choice(
                            instructions=(
                                "Classify the overall sentiment expressed "
                                "by the user in this app review."
                            ),
                            criteria={
                                "positive": (
                                    "The user expresses satisfaction, praise, "
                                    "enjoyment, or recommendation. Overall sentiment "
                                    "is favorable."
                                ),
                                "neutral": (
                                    "The review is factual, informational, or "
                                    "genuinely mixed without a clearly dominant "
                                    "positive or negative sentiment."
                                ),
                                "negative": (
                                    "The user expresses dissatisfaction, frustration, "
                                    "criticism, complaints, or problems with the app. "
                                    "Overall sentiment is unfavorable."
                                )
                            }
                        )
                    }
                )
                
                sentiment_ans = response.answers["sentiment"]
                
                results.append({
                    **row.to_dict(),
                    "jev_sentiment": sentiment_ans.choice,
                    "jev_confidence": sentiment_ans.confidence,
                    "positive_probability": sentiment_ans.probabilities.get("positive", 0.0),
                    "neutral_probability": sentiment_ans.probabilities.get("neutral", 0.0),
                    "negative_probability": sentiment_ans.probabilities.get("negative", 0.0)
                })
                
            except Exception as e:
                print(f"Error processing review: {e}")
                results.append({
                    **row.to_dict(),
                    "jev_sentiment": "error",
                    "jev_confidence": 0.0,
                    "positive_probability": 0.0,
                    "neutral_probability": 0.0,
                    "negative_probability": 0.0
                })

    output_df = pd.DataFrame(results)
    
    # Save structured results
    output_df.to_csv(output_path, index=False)
    print(f"Stage 1 complete. Saved structured output to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to structured output CSV")
    parser.add_argument("--sample", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()
    
    run_sentiment_stage(args.input, args.output, args.sample)
