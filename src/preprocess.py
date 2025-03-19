import re
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import pandas as pd
from tqdm import tqdm
from spellchecker import SpellChecker

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

def preprocess_transcriptions(df: pd.DataFrame):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    english_words = set(words.words())
    spell = SpellChecker()

    normalized_text = []

    texts = df["transcript"].to_list()
    for text in tqdm(texts):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words_list = text.split()

        processed_words = []
        known_words = spell.known(words_list)
        for word in words_list:
            if (word in english_words or word in known_words) and word not in stop_words:
                lemmatized_word = lemmatizer.lemmatize(word)
                processed_words.append(lemmatized_word)
        
        processed_text = ' '.join(processed_words)
        normalized_text.append(processed_text)
    
    return normalized_text

df = pd.read_csv('data/merged_categorized.csv', header=0)
normalized_text = preprocess_transcriptions(df)

print(len(df))
print(len(normalized_text))

normalized_df = df.copy()
normalized_df["transcript"] = normalized_text

# remove rows with empty transcript
normalized_df = normalized_df[normalized_df["transcript"].str.len() > 0]
print(len(normalized_df))

# add document length
normalized_df["doc_length"] = normalized_df["transcript"].apply(lambda x: len(x.split()))
# add lexical diversity
normalized_df["lexical_diversity"] = normalized_df["transcript"].apply(lambda x: len(set(x.split())) / len(x.split()))

normalized_df.to_csv("normalized_merged.csv")
