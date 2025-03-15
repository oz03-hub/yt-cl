import numpy as np
import pandas as pd
import gensim
from typing import Dict, Tuple, List

from base import BaseClusterer


class LdaClusterer(BaseClusterer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_name = "lda"
        self.id2word = None
        self.corpus = None
        
    def preprocess(self, df: pd.DataFrame) -> Tuple[gensim.corpora.Dictionary, List]:
        self.id2word = gensim.corpora.Dictionary(df["transcript"].apply(lambda x: x.split()))
        self.corpus = [self.id2word.doc2bow(doc.split()) for doc in df["transcript"]]
        return self.corpus  # Return corpus as X
    
    def fit(self, X: List) -> None:
        lda = gensim.models.ldamodel.LdaModel(
            corpus=X,
            id2word=self.id2word,
            num_topics=self.n_clusters,
            random_state=self.random_state,
            chunksize=200,
            alpha="asymmetric",
            eta="auto",
        )
        self.model = lda
        
        # Find highest topic for each document
        labels = []
        for doc in X:
            topic_probs = lda.get_document_topics(doc, minimum_probability=0)
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            most_likely_topic = topic_probs[0][0]
            labels.append(most_likely_topic)
        
        self.labels_ = labels
        self.X = X  # Store corpus as X
    
    def calculate_metrics(self) -> Dict[str, float]:
        # For LDA, we use coherence as the main metric
        coherence = gensim.models.coherencemodel.CoherenceModel(
            model=self.model, 
            texts=[doc.split() for doc in pd.read_csv("data/normalized.csv")["transcript"]],
            dictionary=self.id2word, 
            coherence='c_v',
            processes=1
        ).get_coherence()
        
        self.metrics = {
            "coherence": coherence
        }
        return self.metrics
    
    def get_cluster_centers(self, df: pd.DataFrame) -> str:
        text_buffer = "CLUSTER CENTERS:\n"
        
        # Get topic-document probabilities
        topic_doc_probs = []
        for i, doc in enumerate(self.corpus):
            topic_probs = self.model.get_document_topics(doc, minimum_probability=0)
            for topic_num, prob in topic_probs:
                topic_doc_probs.append((topic_num, prob, df.iloc[i]["video_id"]))
        
        topic_df = pd.DataFrame(
            topic_doc_probs, columns=["topic", "probability", "video_id"]
        )
        top_videos_per_topic = (
            topic_df.sort_values(["topic", "probability"], ascending=[True, False])
            .groupby("topic")
            .head(self.num_closest_to_center)
        )
        
        for topic in range(self.n_clusters):
            video_ids = top_videos_per_topic[top_videos_per_topic["topic"] == topic][
                "video_id"
            ].tolist()
            text_buffer += f"CLUSTER {topic}:\n"
            for video_id in video_ids:
                text_buffer += f"{video_id}\n"
            text_buffer += "\n"
            
        return text_buffer
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Override to add topic words"""
        text_buffer = super().generate_report(df)
        
        # Add topic words from LDA model
        topics_text = "TOP WORDS PER TOPIC FROM LDA MODEL:\n"
        topics = self.model.show_topics(num_topics=self.n_clusters, num_words=self.n_words)
        for i, topic in enumerate(topics):
            topics_text += f"TOPIC {i}:\n"
            topics_text += f"{topic}\n\n"
        
        text_buffer += topics_text

        # add number of docs per cluster
        text_buffer += f"NUMBER OF DOCS PER CLUSTER:\n"
        for i in range(self.n_clusters):
            text_buffer += f"CLUSTER {i}: {len(df[df['cluster'] == i])}\n"

        return text_buffer
    