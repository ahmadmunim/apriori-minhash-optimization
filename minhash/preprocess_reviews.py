import pandas as pd
import re
import ast
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# Drop rows without review text
df = df.dropna(subset=['Review Text'])

df = df[['Clothing ID', 'Review Text', 'Rating', 'Recommended IND']]

# Clean and tokenize
stop_words = set(stopwords.words('english'))

def preprocess_text(row):
    text = row['Review Text'].lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    tokens = word_tokenize(text, language='english', preserve_line=True)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens

def get_weight(row):
    rating = row['Rating'] / 5.0  # scale from 1–5 to 0.2–1.0
    recommended = 1.5 if row['Recommended IND'] == 1 else 1.0
    return rating * recommended

df['Tokens'] = df.apply(preprocess_text, axis=1)
df['Weight'] = df.apply(get_weight, axis=1)

# Drop empty token rows
df = df[df['Tokens'].map(len) > 0]

df[['Clothing ID', 'Tokens', 'Weight']].to_csv("preprocessed_reviews.csv", index=False)
print("Preprocessing complete. Output saved to preprocessed_reviews.csv")
