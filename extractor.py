import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# import nltk
# nltk.download('punkt_tab')
complain_counter = Counter()

all_tokens = []


df = pd.read_csv('./preprocessed.csv')


for review in df['Cleaned Reviews'].dropna():
    all_tokens.extend(word_tokenize(review))

unigram = Counter(all_tokens)
print("Top 20 Unigrams : ",unigram.most_common(20))

bigram = Counter(ngrams(all_tokens,2))
print("Top 20 bigrams : ",bigram.most_common(20))

trigram = Counter(ngrams(all_tokens,3))
print("Top 30 trigrams : ",trigram.most_common(20))


wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(' '.join(all_tokens))

plt.figure(figsize=(14, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in Reviews')
# plt.show()
plt.savefig("word_freq.png") 