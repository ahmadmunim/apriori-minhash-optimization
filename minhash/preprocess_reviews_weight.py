import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# Drop missing review text
df = df.dropna(subset=['Review Text'])

# Select relevant columns
df = df[['Clothing ID', 'Review Text', 'Rating', 'Recommended IND']]

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text, language='english', preserve_line=True)
    return [t for t in tokens if t not in stop_words and len(t) > 1]

def get_token_weights(row):
    tokens = preprocess_text(row['Review Text'])
    if not tokens:
        return {}

    # Weight = normalized rating × recommendation factor
    rating = row['Rating'] / 5.0
    rec_factor = 1.5 if row['Recommended IND'] == 1 else 1.0
    weight = rating * rec_factor

    # Assign the same weight to all tokens in this review
    return {token: round(weight, 3) for token in tokens}

df['TokenWeights'] = df.apply(get_token_weights, axis=1)

# Remove rows with empty token sets
df = df[df['TokenWeights'].map(len) > 0]

df[['Clothing ID', 'TokenWeights']].to_csv("preprocessed_reviews_weighted.csv", index=False)
print("Preprocessing complete. Output saved to preprocessed_reviews_weighted.csv")
