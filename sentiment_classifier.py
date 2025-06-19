# import nltk
# nltk.download("vader_lexicon")
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
tqdm.pandas()

vader  = SentimentIntensityAnalyzer()

df = pd.read_csv("./preprocessed_2.csv")

REVIEW_COLUMN = 'Review'

df = df[df[REVIEW_COLUMN].notnull()]
df = df[df[REVIEW_COLUMN].str.strip() != ""]

def map_stars_to_sentiment(star):
    if star in [1,2]:
        return 0
    elif star in [3,4]:
        return 1
    else:
        return 2

def get_vader_sentiment(text):
    scores = vader.polarity_scores(text)
    compound = scores['compound']
    if compound >=0.05:
        label = 'positive'
    elif compound <=-0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return pd.Series([scores['neg'],scores['neu'],scores['pos'],compound,label])    

df[['neg','neu','pos','compound','vader_label']] = df[REVIEW_COLUMN].progress_apply(get_vader_sentiment)

df['star_sentiments'] = df['Stars'].progress_apply(map_stars_to_sentiment)

df.to_csv('sentiment_classified_2.csv',index=False)

print("Sentiment analysis completed and saved as sentiment_classified_2.csv")

