import tyro
import os
import pandas as pd
from util import find_topk_words
import numpy as np
import datetime
import gensim

def get_x(df):
    id2word = gensim.corpora.Dictionary(df["transcript"].apply(lambda x: x.split()))
    corpus = [id2word.doc2bow(doc.split()) for doc in df["transcript"]]
    return id2word, corpus

def get_cluster_from_df(id2word, corpus, n_clusters, random_state=None):
    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=n_clusters,
        random_state=random_state,
        chunksize=200,
        alpha="asymmetric",
        eta="auto",
    )

    # find highest topic for each document
    labels = []
    for doc in corpus:
        topic_probs = lda.get_document_topics(doc, minimum_probability=0)
        topic_probs.sort(key=lambda x: x[1], reverse=True)
        most_likely_topic = topic_probs[0][0]
        labels.append(most_likely_topic)
    return labels

def main(
    data_path: str = "data/transcripts_normalized.csv",
    output_dir: str = "data/lda",
    n_clusters: int = 18,
    n_words: int = 25,
    num_closest_to_center: int = 5,
    random_state: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path, header=0)
    df.dropna(subset=["transcript"], inplace=True)
    id2word = gensim.corpora.Dictionary(df["transcript"].apply(lambda x: x.split()))
    corpus = [id2word.doc2bow(doc.split()) for doc in df["transcript"]]
    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=n_clusters,
        random_state=random_state,
        chunksize=200,
    )

    text_buffer = "lda, general info\n"
    text_buffer += f"n_clusters: {n_clusters}\n"
    text_buffer += f"n_words: {n_words}\n"
    text_buffer += f"num_closest_to_center: {num_closest_to_center}\n"
    text_buffer += f"random_state: {random_state}\n"
    text_buffer += "\n"

    text_buffer += "TOP WORDS PER CLUSTER:\n"
    topics = lda.show_topics(num_topics=n_clusters, num_words=n_words)
    for i, topic in enumerate(topics):
        text_buffer += f"CLUSTER {i}:\n"
        text_buffer += f"{topic}\n"
        # for word, prob in topic:
        #     text_buffer += f"{word}: {prob}\n"
        text_buffer += "\n"
    text_buffer += "\n"

    text_buffer += f"coherence: {gensim.models.coherencemodel.CoherenceModel(model=lda, texts=df['transcript'].apply(lambda x: x.split()).tolist(), dictionary=id2word, coherence='c_v').get_coherence()}\n"
    text_buffer += "\n"

    text_buffer += "NUMBER OF DOCS PER CLUSTER:\n"
    labels = get_cluster_from_df(id2word, corpus, n_clusters, random_state=random_state)
    df["cluster"] = labels
    for i in range(n_clusters):
        text_buffer += f"CLUSTER {i}: {len(df[df['cluster'] == i])}\n"
    text_buffer += "\n"

    topic_doc_probs = []
    for i, doc in enumerate(corpus):
        topic_probs = lda.get_document_topics(doc, minimum_probability=0)
        for topic_num, prob in topic_probs:
            topic_doc_probs.append((topic_num, prob, df.iloc[i]["video_id"]))

    topic_df = pd.DataFrame(
        topic_doc_probs, columns=["topic", "probability", "video_id"]
    )
    top_videos_per_topic = (
        topic_df.sort_values(["topic", "probability"], ascending=[True, False])
        .groupby("topic")
        .head(num_closest_to_center)
    )

    text_buffer += "CLUSTER CENTERS:\n"
    for topic in range(n_clusters):
        video_ids = top_videos_per_topic[top_videos_per_topic["topic"] == topic][
            "video_id"
        ].tolist()
        text_buffer += f"CLUSTER {topic}:\n"
        for video_id in video_ids:
            text_buffer += f"{video_id}\n"
        text_buffer += "\n"

    with open(
        os.path.join(
            output_dir,
            f"lda_{datetime.datetime.now().strftime('%m%d_%H%M')}_info.txt",
        ),
        "w",
    ) as f:
        f.write(text_buffer)


if __name__ == "__main__":
    tyro.cli(main)
