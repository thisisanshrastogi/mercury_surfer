import pandas as pd
import re
import string
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
import os 
import glob
tqdm.pandas()


csv_files = glob.glob(os.path.join("./", "*.csv"))

df_list = [pd.read_csv(file) for file in csv_files]

df_list = [df for file, df in zip(csv_files, df_list) if os.path.basename(file).startswith("reviews_com")]

# df = pd.read_csv("./reviews/reviews_com.tdbank.csv")

df = pd.concat(df_list, ignore_index=True)

reviews = df['Review'].dropna().astype(str)

# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


nlp = spacy.load('en_core_web_sm',disable=["parser","ner"])

def clean_text(text):
    text = text.lower()
    #Can add more regexes for filtering
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # text = re.sub(r'[^a-zA-Z\s]','',text)
    text = re.sub(r'\s+',' ',text.strip())
    return text

def tokeize_lemmatize(text):
    doc = nlp(text)
    return ' '.join(
        [
            token.lemma_ for token in doc
            if token.lemma_ not in stop_words and token.is_alpha and len(token) > 2
        ]
    )

print("Cleaning.....")
cleaned_reviews = reviews.progress_apply(clean_text)
print("Tokenizing and Lemmatizing.....")
cleaned_reviews = reviews.progress_apply(tokeize_lemmatize)

df['Cleaned Reviews'] = cleaned_reviews


df.to_csv("preprocessed_2.csv",index=False)



