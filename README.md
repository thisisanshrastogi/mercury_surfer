# mercury_surfer

This repository contains scripts for scraping and analyzing reviews of Android banking apps in the U.S.

## Overview

We built an automated browser-based review scraper (less of a traditional scraper) that collects reviews for a list of 20 Android banking app package names specified in `./app_list.txt`. Using this tool, we gathered over **72,000 reviews** from the Google Play Store.

> **Note:** I do not consider myself an expert on these topics, so I welcome any feedback or suggestions for improving the approaches described here.

## Data Preprocessing

After collecting the reviews, we performed basic preprocessing steps:

- Removing stop words
- Stemming and lemmatization

We initially applied sentiment analysis using the VADER tool. However, we later relied on the star ratings from the reviews instead of the sentiment scores.

## Topic Modeling Approaches

We explored three methodologies for topic modeling:

1. **LDA (Latent Dirichlet Allocation)**
2. **BERTopic**
3. **KMeans Clustering**

### 1. LDA (Latent Dirichlet Allocation)

- Converted reviews into bigrams, inspired by [this paper](https://www.researchgate.net/publication/389917478_Banking_on_Feedback_Text_Analysis_of_Mobile_Banking_iOS_and_Google_App_Reviews/fulltext/67d8ec027d56ad0a0f05aaa2/Banking-on-Feedback-Text-Analysis-of-Mobile-Banking-iOS-and-Google-App-Reviews.pdf?origin=scientificContributions).
- Used the `gensim` library to create a dictionary and corpus.
- Trained an LDA model (`LdaModel` from gensim) to extract topics.
- Determined the optimal number of topics (12) using coherence scores.
- Saved results to `./output/LDA_topics_by_model.csv`.
- Passed the topics to multiple LLMs for generating problem statements, followed by manual curation by Tamajit and Lizann.

### 2. BERTopic

- Used the same corpus and dictionary as LDA.
- Removed 5-star reviews to avoid skewing topics toward positive sentiment.
- Generated embeddings using the `all-mpnet-base-v2` model from Sentence Transformers.
- Applied the `bertopic` library to extract topics and top words.
- Saved results to `./output/bertopic_topics_expressive.csv`.
- Topics were further processed by LLMs and manually curated.

### 3. KMeans Clustering

- Used the same corpus and dictionary as previous methods.
- Clustered reviews using KMeans instead of topic modeling.
- Chose 12 clusters to match the number of LDA topics.
- Attempted to visualize clusters, but found them indistinct.
- Clustered reviews and extracted top words for each cluster.
- Passed clusters to LLMs for problem statement generation.
- Did not complete this approach as the team was satisfied with previous results.
- Results saved in `./output/kmean_clustering_output.md`.

---

## Output Files

- `./output/LDA_topics_by_model.csv`: LDA topics and top words
- `./output/bertopic_topics_expressive.csv`: BERTopic topics and top words
- `./output/kmean_clustering_output.md`: KMeans clustering results

---

## Contributors

- Tamajit
- Lizann
- Aadi
- Ansh
- And others if any.

---

## References

- [Banking on Feedback: Text Analysis of Mobile Banking iOS and Google App Reviews](https://www.researchgate.net/publication/389917478_Banking_on_Feedback_Text_Analysis_of_Mobile_Banking_iOS_and_Google_App_Reviews/fulltext/67d8ec027d56ad0a0f05aaa2/Banking-on-Feedback-Text-Analysis-of-Mobile-Banking-iOS-and-Google-App-Reviews.pdf?origin=scientificContributions)
