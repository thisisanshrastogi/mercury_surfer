import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
import gensim
import pyLDAvis.gensim_models
import pyLDAvis
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser

tqdm.pandas()

nltk.download("stopwords")
nltk.download("word")

df = pd.read_csv("./sentiment_classified.csv")

def clean_more(text):
    text = text.lower()
    return text.split()

df['Cleaned Reviews'] = df['Cleaned Reviews'].fillna('')
df['tokens'] = df["Cleaned Reviews"].progress_apply(clean_more)

bigram_phrases = Phrases(df['tokens'],min_count=5,threshold=5)

bigram_mod = Phraser(bigram_phrases)

df['bigram_tokens'] = df['tokens'].progress_apply(lambda doc: bigram_mod[doc])

print(df['bigram_tokens'].sample(5).tolist())


dictionary = corpora.Dictionary(df['bigram_tokens'])

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=5000)

corpus = [dictionary.doc2bow(doc) for doc in df['bigram_tokens']]

num_topics = 10

# lda = LdaMulticore(
#     corpus=corpus,
#     id2word=dictionary,
#     num_topics=num_topics,
#     passes=15,       # how many times the model will sweep over the corpus
#     workers=6,       # parallel processes
#     random_state=42
# )

# print("\nDiscovered topics:")
# for idx, topic in lda.print_topics(num_topics=num_topics, num_words=8):
#     print(f"Topic {idx+1}: {topic}")


