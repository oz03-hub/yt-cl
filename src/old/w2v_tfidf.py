import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from util import find_topk_words
import datetime
import tyro
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")

def main(
    data_path: str = "data/transcripts_normalized.csv",
    output_dir: str = "data/w2v_tfidf",
    n_clusters: int = 18,
    n_words: int = 25,
    num_closest_to_center: int = 5,
    random_state: int | None = 42,
    save_output: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    wv = gensim.models.KeyedVectors.load("word2vec_youtube_vectors.kv")

    df = pd.read_csv(data_path, header=0)
    df.dropna(subset=["transcript"], inplace=True)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: word_tokenize(x))
    X_tfidf = vectorizer.fit_transform(df["transcript"])
    X_tfidf = X_tfidf.toarray()
    X = np.matmul(X_tfidf, wv.vectors)
    dl = [len(word_tokenize(x)) for x in df["transcript"]]
    dl = np.array(dl)
    X_avg = X / dl[:, np.newaxis]


    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_avg)

    text_buffer = "w2v tfidf, general info\n"
    text_buffer += f"n_clusters: {n_clusters}\n"
    text_buffer += f"n_words to save: {n_words}\n"
    text_buffer += f"num_closest_to_center: {num_closest_to_center}\n"
    text_buffer += f"random_state: {random_state}\n"
    text_buffer += "\n"
    text_buffer += f"inertia: {kmeans.inertia_}\n"
    text_buffer += f"n_iter: {kmeans.n_iter_}\n"
    text_buffer += f"silhouette_score: {silhouette_score(X_avg, kmeans.labels_)}\n"
    text_buffer += (
        f"davies_bouldin_score: {davies_bouldin_score(X_avg, kmeans.labels_)}\n"
    )
    text_buffer += "\n"

    text_buffer += "TOP WORDS PER CLUSTER:\n"
    cluster_words = find_topk_words(df, kmeans, n_words)
    for i, words in enumerate(cluster_words):
        text_buffer += f"CLUSTER {i}:\n"
        for word, freq in words:
            text_buffer += f"{word}: {freq}\n"
        text_buffer += "\n"

    df["cluster"] = kmeans.labels_
    # number of docs per cluster
    text_buffer += "NUMBER OF DOCS PER CLUSTER:\n"
    for i in range(n_clusters):
        text_buffer += f"CLUSTER {i}: {len(df[df['cluster'] == i])}\n"
    text_buffer += "\n"

    # text_buffer += "CLUSTER CENTERS:\n"
    # cluster_centers = kmeans.cluster_centers_

    # df["w2v_tfidf_vector"] = [row for row in X_avg]
    
    # for i in range(n_clusters):
    #     text_buffer += f"CLUSTER {i}:\n"
    #     cluster_mask = df["cluster"] == i
    #     cluster_docs = df[cluster_mask].copy()
    #     cluster_center = cluster_centers[i]
    #     distances = np.linalg.norm(cluster_docs["w2v_tfidf_vector"] - cluster_center, axis=1)
    #     cluster_docs["distance"] = distances
    #     closest_docs = cluster_docs.sort_values(by="distance", ascending=True).head(num_closest_to_center)
    #     for video_id in closest_docs["video_id"]:
    #         text_buffer += f"{video_id}\n"
    #     text_buffer += "\n"

    if save_output:
        with open(os.path.join(output_dir, f"w2v_tfidf_{datetime.datetime.now().strftime('%m%d_%H%M')}_info.txt"), "w") as f:
            f.write(text_buffer)


if __name__ == "__main__":
    tyro.cli(main)
