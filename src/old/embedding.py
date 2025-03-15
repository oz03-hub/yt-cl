import tyro
import os
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from util import find_topk_words
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer


def get_x(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(df["transcript"].to_list(), convert_to_numpy=True)
    return X


def cluster_from_x(X, n_clusters, random_state=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans


def main(
    data_path: str = "data/transcripts_normalized.csv",
    output_dir: str = "data/embedding",
    n_clusters: int = 18,
    n_words: int = 25,
    num_closest_to_center: int = 5,
    random_state: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path, header=0)
    df.dropna(subset=["transcript"], inplace=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(df["transcript"].to_list(), convert_to_numpy=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)

    text_buffer = "embedding kmeans, general info\n"
    text_buffer += f"n_clusters: {n_clusters}\n"
    text_buffer += f"n_words to save: {n_words}\n"
    text_buffer += f"num_closest_to_center: {num_closest_to_center}\n"
    text_buffer += f"random_state: {random_state}\n"
    text_buffer += "\n"
    text_buffer += f"inertia: {kmeans.inertia_}\n"
    text_buffer += f"n_iter: {kmeans.n_iter_}\n"
    text_buffer += f"silhouette_score: {silhouette_score(X, kmeans.labels_)}\n"
    text_buffer += f"davies_bouldin_score: {davies_bouldin_score(X, kmeans.labels_)}\n"
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

    df["embedding"] = [row for row in X]
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))

    for i in range(n_clusters):
        text_buffer += f"CLUSTER {i}:\n"  # find closest docs to center
        cluster_mask = df["cluster"] == i
        cluster_docs = df[cluster_mask].copy()
        cluster_center = cluster_centers[i]
        embedding_matrix = np.vstack(cluster_docs["embedding"].to_list())
        distances = np.linalg.norm(embedding_matrix - cluster_center, axis=1)
        cluster_docs["distance"] = distances
        closest_docs = cluster_docs.sort_values(by="distance", ascending=True).head(
            num_closest_to_center
        )
        for video_id in closest_docs["video_id"]:
            text_buffer += f"{video_id}\n"
        text_buffer += "\n"

    with open(
        os.path.join(
            output_dir,
            f"embedding_{datetime.datetime.now().strftime('%m%d_%H%M')}_info.txt",
        ),
        "w",
    ) as f:
        f.write(text_buffer)


if __name__ == "__main__":
    tyro.cli(main)
