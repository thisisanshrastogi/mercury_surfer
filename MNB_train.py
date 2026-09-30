from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
from sklearn.utils import resample




tqdm.pandas()

df = pd.read_csv('./sentiment_classified.csv')

df['Cleaned Reviews'] = df['Cleaned Reviews'].fillna('')

# df_neg = df[df['label_column'] == 0]
# df_neu = df[df['label_column'] == 1]
# df_pos = df[df['label_column'] == 2]

df_0 = df[df['star_sentiments'] == 0]
df_1 = df[df['star_sentiments'] == 1]
df_2 = df[df['star_sentiments'] == 2]

min_size = min(len(df_0), len(df_1), len(df_2))
df_0_downsampled = resample(df_0, replace=False, n_samples=min_size, random_state=42)
df_1_downsampled = resample(df_1, replace=False, n_samples=min_size, random_state=42)
df_2_downsampled = resample(df_2, replace=False, n_samples=min_size, random_state=42)
df_balanced = pd.concat([df_0_downsampled, df_1_downsampled, df_2_downsampled])



vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')

X = vectorizer.fit_transform(df_balanced['Review'].fillna(''))

# vader_features = df[['neg', 'neu', 'pos']].fillna(0).values
# X_vader = csr_matrix(vader_features)

# X = hstack([X_text, X_vader])


y = df_balanced['star_sentiments']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)



clf = MultinomialNB()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

