import tyro
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from util import find_topk_words
import numpy as np
import datetime

def get_x(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["transcript"])
    return X

def cluster_from_x(X, n_clusters, random_state=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans


def main(
    data_path: str = "data/transcripts_normalized.csv",
    output_dir: str = "data/tfidf",
    n_clusters: int = 18,
    n_words: int = 25,
    num_closest_to_center: int = 5,
    random_state: int | None = 42,
    save_output: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path, header=0)
    df.dropna(subset=["transcript"], inplace=True)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["transcript"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)

    text_buffer = "tfidf kmeans, general info\n"
    text_buffer += f"n_clusters: {n_clusters}\n"
    text_buffer += f"n_words to save: {n_words}\n"
    text_buffer += f"num_closest_to_center: {num_closest_to_center}\n"
    text_buffer += f"random_state: {random_state}\n"
    text_buffer += "\n"
    text_buffer += f"inertia: {kmeans.inertia_}\n"
    text_buffer += f"n_iter: {kmeans.n_iter_}\n"
    text_buffer += f"silhouette_score: {silhouette_score(X, kmeans.labels_)}\n"
    text_buffer += (
        f"davies_bouldin_score: {davies_bouldin_score(X.toarray(), kmeans.labels_)}\n"
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

    text_buffer += "CLUSTER CENTERS:\n"
    cluster_centers = kmeans.cluster_centers_

    df["tfidf_vector"] = [row for row in X.todense()]
    df["tfidf_vector"] = df["tfidf_vector"].apply(lambda x: np.array(x))

    for i in range(n_clusters):
        text_buffer += f"CLUSTER {i}:\n"  # find closest docs to center
        cluster_mask = df["cluster"] == i
        cluster_docs = df[cluster_mask].copy()
        cluster_center = cluster_centers[i]
        tfidf_matrix = np.vstack(cluster_docs["tfidf_vector"].values)
        distances = np.linalg.norm(tfidf_matrix - cluster_center, axis=1)
        cluster_docs["distance"] = distances
        closest_docs = cluster_docs.sort_values(by="distance", ascending=True).head(
            num_closest_to_center
        )
        for video_id in closest_docs["video_id"]:
            text_buffer += f"{video_id}\n"
        text_buffer += "\n"

    if save_output:
        with open(
            os.path.join(
                output_dir,
                f"tfidf_{datetime.datetime.now().strftime('%m%d_%H%M')}_info.txt",
            ),
            "w",
        ) as f:
            f.write(text_buffer)

    return kmeans.labels_


if __name__ == "__main__":
    tyro.cli(main)
