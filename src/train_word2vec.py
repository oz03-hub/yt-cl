import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download("punkt")
nltk.download("punkt_tab")

# Load the text file
df = pd.read_csv("data/normalized.csv", header=0)

def preprocess_text(text):
    tokens = word_tokenize(text)
    return tokens

corpus = [preprocess_text(text) for text in df["transcript"]]
model = Word2Vec(corpus, vector_size=200, window=5, min_count=1, workers=4, sg=0, epochs=10)
model.save("word2vec_youtube.model")
model.wv.save("word2vec_youtube_vectors.kv")

print("Word2Vec model trained and saved.")
