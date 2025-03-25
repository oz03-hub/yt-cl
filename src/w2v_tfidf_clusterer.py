import numpy as np
import pandas as pd
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Optional
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

from base import BaseClusterer


class W2vTfidfClusterer(BaseClusterer):
    def __init__(
        self, 
        word2vec_path: str = "word2vec_youtube_vectors.kv", 
        cache_embeddings: bool = False,
        cache_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.method_name = "w2v_tfidf"
        self.word2vec_path = word2vec_path
        self.cache_embeddings = cache_embeddings
        self.cache_path = cache_path or f"data/cache/{self.method_name}_embeddings.npz"
        self.wv = None
        self.vectorizer = None
        
    def create_document_embeddings(self, documents):
        """
        Create document embeddings as weighted averages of word vectors.
        
        Args:
            documents: List of document texts
        
        Returns:
            Document embeddings matrix
        """
        # Tokenize documents
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize, lowercase=True)
        self.vectorizer.fit(documents)
        
        # Get vocabulary mapping
        vocab = self.vectorizer.vocabulary_
        
        # Initialize document embeddings matrix
        vector_size = self.wv.vector_size
        doc_vectors = np.zeros((len(documents), vector_size))
        
        # For each document
        for i, doc in enumerate(tokenized_docs):
            if not doc:
                continue
                
            # Get TF-IDF for this document
            doc_tfidf = self.vectorizer.transform([documents[i]]).toarray()[0]
            
            # Initialize weighted sum and total weight
            weighted_sum = np.zeros(vector_size)
            total_weight = 0
            
            # For each word in document
            for word in doc:
                # Skip if word not in vocabulary
                if word not in vocab:
                    continue
                    
                # Skip if word not in word2vec vocabulary
                if word not in self.wv:
                    continue
                    
                # Get word index in TF-IDF vocabulary
                word_idx = vocab[word]
                
                # Get word weight
                weight = doc_tfidf[word_idx]
                
                # Skip if weight is zero
                if weight == 0:
                    continue
                    
                # Add weighted word vector
                weighted_sum += weight * self.wv[word]
                total_weight += weight
            
            # Normalize by total weight if non-zero
            if total_weight > 0:
                doc_vectors[i] = weighted_sum / total_weight
        
        return doc_vectors
        
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Transform the dataframe into feature vectors"""
        # Check for cached embeddings
        if self.cache_embeddings and os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}")
            cached = np.load(self.cache_path)
            return cached['embeddings']
        
        # Load word vectors if not already loaded
        if self.wv is None:
            print(f"Loading word vectors from {self.word2vec_path}")
            self.wv = gensim.models.KeyedVectors.load(self.word2vec_path)
        
        # Create document embeddings
        print("Creating document embeddings...")
        X = self.create_document_embeddings(df["transcript"].tolist())
        
        # Cache embeddings if requested
        if self.cache_embeddings:
            print(f"Caching embeddings to {self.cache_path}")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez_compressed(self.cache_path, embeddings=X)
            
        return X
    
    def fit(self, X: np.ndarray) -> None:
        """Fit the clustering model to the data"""
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
        )
        kmeans.fit(X)
        self.model = kmeans
        self.labels_ = kmeans.labels_
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        self.metrics = {
            "inertia": self.model.inertia_,
            "n_iter": self.model.n_iter_,
            "silhouette_score": silhouette_score(self.X, self.labels_),
            "davies_bouldin_score": davies_bouldin_score(self.X, self.labels_)
        }
        return self.metrics
    
    def get_cluster_centers(self, df: pd.DataFrame) -> str:
        """Get the closest documents to each cluster center"""
        text_buffer = "CLUSTER CENTERS:\n"
        cluster_centers = self.model.cluster_centers_
        
        # Store vectors for distance calculation
        df_with_vectors = df.copy()
        df_with_vectors["w2v_tfidf_vector"] = list(self.X)
        
        for i in range(self.n_clusters):
            text_buffer += f"CLUSTER {i}:\n"
            cluster_mask = df_with_vectors["cluster"] == i
            cluster_docs = df_with_vectors[cluster_mask].copy()
            
            if len(cluster_docs) == 0:
                text_buffer += "No documents in this cluster\n\n"
                continue
                
            cluster_center = cluster_centers[i]
            
            # Calculate distances to center
            distances = np.linalg.norm(
                np.stack(cluster_docs["w2v_tfidf_vector"].values) - cluster_center, 
                axis=1
            )
            
            cluster_docs["distance"] = distances
            closest_docs = cluster_docs.sort_values(by="distance", ascending=True).head(
                self.num_closest_to_center
            )
            
            for video_id in closest_docs["video_id"]:
                text_buffer += f"{video_id}\n"
            text_buffer += "\n"
                
        return text_buffer