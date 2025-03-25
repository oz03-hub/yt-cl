import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import tyro

nltk.download("punkt")
nltk.download("punkt_tab")

def preprocess_text(text):
    tokens = word_tokenize(text)
    return tokens

def main(input_file: str = "data/normalized_merged_categorized_good_len_filtered.csv", output_file: str = "word2vec_youtube"):
    df = pd.read_csv(input_file, header=0)

    corpus = [preprocess_text(text) for text in df["transcript"]]
    model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4, sg=0, epochs=10)
    model.save(output_file + ".model")
    model.wv.save(output_file + "_vectors.kv")


if __name__ == "__main__":
    tyro.cli(main)
