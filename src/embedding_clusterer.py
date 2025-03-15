import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict

from base import BaseClusterer


class EmbeddingClusterer(BaseClusterer):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        super().__init__(**kwargs)
        self.method_name = "embedding"
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        X = self.embedding_model.encode(df["transcript"].to_list(), convert_to_numpy=True)
        return X
    
    def fit(self, X: np.ndarray) -> None:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(X)
        self.model = kmeans
        self.labels_ = kmeans.labels_
    
    def calculate_metrics(self) -> Dict[str, float]:
        self.metrics = {
            "inertia": self.model.inertia_,
            "n_iter": self.model.n_iter_,
            "silhouette_score": silhouette_score(self.X, self.labels_),
            "davies_bouldin_score": davies_bouldin_score(self.X, self.labels_)
        }
        return self.metrics
    
    def get_cluster_centers(self, df: pd.DataFrame) -> str:
        text_buffer = "CLUSTER CENTERS:\n"
        cluster_centers = self.model.cluster_centers_
        
        df_with_vectors = df.copy()
        df_with_vectors["embedding"] = [row for row in self.X]
        
        for i in range(self.n_clusters):
            text_buffer += f"CLUSTER {i}:\n"
            cluster_mask = df_with_vectors["cluster"] == i
            cluster_docs = df_with_vectors[cluster_mask].copy()
            cluster_center = cluster_centers[i]
            embedding_matrix = np.vstack(cluster_docs["embedding"].to_list())
            distances = np.linalg.norm(embedding_matrix - cluster_center, axis=1)
            cluster_docs["distance"] = distances
            closest_docs = cluster_docs.sort_values(by="distance", ascending=True).head(
                self.num_closest_to_center
            )
            for video_id in closest_docs["video_id"]:
                text_buffer += f"{video_id}\n"
            text_buffer += "\n"
            
        return text_buffer