from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import pandas as pd

# this file is for testing

df = pd.read_csv("./preprocessed.csv")


cleaned_docs = df['Cleaned Reviews']

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=100,
    max_df= 0.8,
    min_df=5,
    ngram_range=(1,3),
    stop_words='english'
)

vectors = vectorizer.fit_transform(cleaned_docs)

print(vectors[0])
