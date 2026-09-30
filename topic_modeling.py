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
from gensim.models.coherencemodel import CoherenceModel


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


def compute_coherence_values(dictionary, corpus, texts, start=5, limit=20, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit + 1, step):
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=15,
            workers=6,
            random_state=42
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherencemodel.get_coherence()
        coherence_values.append(coherence_score)
        print(f'Number of topics: {num_topics}, Coherence Score: {coherence_score:.4f}')
    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary, corpus, df['bigram_tokens'], start=5, limit=15, step=1)


best_index = coherence_values.index(max(coherence_values))
best_model = model_list[best_index]
best_num_topics = 5 + best_index  # because start=5 and step=1

print(f"\nBest number of topics: {best_num_topics} with coherence score {coherence_values[best_index]:.4f}")

# Print topics of the best model
print("\nDiscovered topics:")
for idx, topic in best_model.print_topics(num_topics=best_num_topics, num_words=8):
    print(f"Topic {idx+1}: {topic}")