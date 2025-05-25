import re
import spacy
import emoji
from langdetect import detect
from nltk.corpus import stopwords

# Load English model for spaCy
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    try:
        # Language filter
        if detect(text) != 'en':
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')

        # Remove HTML tags and URLs
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Lemmatization and stopword removal
        doc = nlp(text)
        cleaned = " ".join([token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct])
        
        return cleaned.strip()
    
    except Exception:
        return ""
print(clean_text("Love this app! 😍 But bill pay failed 3 times last week!!"))