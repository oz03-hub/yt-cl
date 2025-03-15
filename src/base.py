import os
import pandas as pd
import numpy as np
import datetime
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score
from util import find_topk_words


class BaseClusterer(ABC):
    def __init__(
        self,
        n_clusters: int = 18,
        n_words: int = 25,
        num_closest_to_center: int = 5,
        random_state: Optional[int] = 42,
    ):
        self.n_clusters = n_clusters
        self.n_words = n_words
        self.num_closest_to_center = num_closest_to_center
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.X = None
        self.labels_ = None
        self.method_name = "base"
        self.metrics = {}

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Transform the dataframe into feature vectors"""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the clustering model to the data"""
        pass

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full clustering pipeline"""
        # Preprocess data
        self.X = self.preprocess(df)
        
        # Fit model
        self.fit(self.X)
        
        # Add cluster labels to dataframe
        df_result = df.copy()
        df_result["cluster"] = self.labels_
        
        # Calculate metrics
        self.calculate_metrics()
        
        return df_result
    
    @abstractmethod
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        pass
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a text report of clustering results"""
        text_buffer = f"{self.method_name}, general info\n"
        text_buffer += f"n_clusters: {self.n_clusters}\n"
        text_buffer += f"n_words: {self.n_words}\n"
        text_buffer += f"num_closest_to_center: {self.num_closest_to_center}\n"
        text_buffer += f"random_state: {self.random_state}\n"
        text_buffer += "\n"
        
        # Add metrics
        for metric_name, metric_value in self.metrics.items():
            text_buffer += f"{metric_name}: {metric_value}\n"
        text_buffer += "\n"
        
        # Top words per cluster
        if self.method_name != "lda":
            text_buffer += "TOP WORDS PER CLUSTER:\n"
            cluster_words = find_topk_words(df, self, self.n_words)
            for i, words in enumerate(cluster_words):
                text_buffer += f"CLUSTER {i}:\n"
                for word, freq in words:
                    text_buffer += f"{word}: {freq}\n"
                text_buffer += "\n"
            
            # Number of docs per cluster
            text_buffer += "NUMBER OF DOCS PER CLUSTER:\n"
            for i in range(self.n_clusters):
                text_buffer += f"CLUSTER {i}: {len(df[df['cluster'] == i])}\n"
            text_buffer += "\n"
        
        # Add cluster centers if implemented
        centers_text = self.get_cluster_centers(df)
        if centers_text:
            text_buffer += centers_text
        
        return text_buffer
    
    def get_cluster_centers(self, df: pd.DataFrame) -> str:
        """Get the closest documents to each cluster center"""
        return ""  # Default implementation returns empty string
    
    def save_report(self, report: str, output_dir: str) -> None:
        """Save the report to a file"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
        with open(
            os.path.join(output_dir, f"{self.method_name}_{timestamp}_info.txt"),
            "w"
        ) as f:
            f.write(report)