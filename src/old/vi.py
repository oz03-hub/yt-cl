import tyro
from sklearn.metrics import mutual_info_score
import numpy as np
import tfidf
import embedding
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lda


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))


def variation_of_information(labels1, labels2):
    h1 = entropy(labels1)
    h2 = entropy(labels2)
    mi = mutual_info_score(labels1, labels2)
    return h1 + h2 - 2 * mi  # eq 19 from paper


def run_n_clustering(n_clusters, X, model, random_state=None):
    if model == "random":
        np.random.seed(random_state)
        cluster_runs = np.random.randint(0, n_clusters, size=len(X))
        return cluster_runs

    if model == "tfidf":
        labels = tfidf.cluster_from_x(X, n_clusters, random_state=random_state).labels_
    elif model == "embedding":
        labels = embedding.cluster_from_x(
            X, n_clusters, random_state=random_state
        ).labels_
    elif model == "lda":
        # X[0] is id2word, X[1] is corpus
        labels = lda.get_cluster_from_df(
            X[0], X[1], n_clusters, random_state=random_state
        )
    return labels


def main(
    n_clusters: int = 18,
    runs: int = 10,
):
    models = ["tfidf", "embedding", "lda", "random"]

    df = pd.read_csv("data/transcripts_normalized.csv", header=0)
    df.dropna(subset=["transcript"], inplace=True)

    X_tfidf = tfidf.get_x(df)
    X_embedding = embedding.get_x(df)
    X_lda = lda.get_x(df)  # This returns (id2word, corpus)
    X_random = np.random.randint(0, n_clusters, size=len(df))
    X_runs = [X_tfidf, X_embedding, X_lda, X_random]

    run_matrices = []

    # create matrix of vi scores
    for run in tqdm(range(runs), desc="Running clustering comparisons"):
        # random_state = 42 + run  # Use different random state for each run
        random_state = None
        vi_scores = np.zeros((len(models), len(models)))
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                model1_labels = run_n_clustering(
                    n_clusters, X_runs[i], model1, random_state=random_state
                )
                model2_labels = run_n_clustering(
                    n_clusters, X_runs[j], model2, random_state=random_state
                )
                vi_scores[i, j] = variation_of_information(model1_labels, model2_labels)
        run_matrices.append(vi_scores)

    # calculate mean vi scores
    mean_vi_scores = np.mean(run_matrices, axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mean_vi_scores,
        annot=True,
        cmap="YlGnBu",
        xticklabels=models,
        yticklabels=models,
        fmt=".3f",
    )
    plt.title(f"Mean Variation of Information ({n_clusters} clusters, {runs} runs)")
    plt.tight_layout()
    plt.savefig(f"vi_scores_{n_clusters}.png")
    plt.close()

    # create matrix of std vi scores
    std_vi_scores = np.std(run_matrices, axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        std_vi_scores,
        annot=True,
        cmap="YlGnBu",
        xticklabels=models,
        yticklabels=models,
        fmt=".3f",
    )
    plt.title(
        f"Standard Deviation of Variation of Information ({n_clusters} clusters, {runs} runs)"
    )
    plt.tight_layout()
    plt.savefig(f"vi_scores_std_{n_clusters}.png")
    plt.close()


if __name__ == "__main__":
    tyro.cli(main)

# import tyro
# from sklearn.metrics import mutual_info_score
# import numpy as np
# import tfidf
# import embedding
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import lda

# def entropy(labels):
#     _, counts = np.unique(labels, return_counts=True)
#     probs = counts / counts.sum()
#     return -np.sum(probs * np.log(probs))


# def variation_of_information(labels1, labels2):
#     h1 = entropy(labels1)
#     h2 = entropy(labels2)
#     mi = mutual_info_score(labels1, labels2)
#     return h1 + h2 - 2 * mi # eq 19 from paper


# def run_n_clustering(n_clusters, X, model):
#     if model == "random":
#         cluster_runs = np.random.randint(0, n_clusters, size=len(X))
#         return cluster_runs

#     if model == "tfidf":
#         labels = tfidf.cluster_from_x(X, n_clusters).labels_
#     elif model == "embedding":
#         labels = embedding.cluster_from_x(X, n_clusters).labels_
#     elif model == "lda":
#         labels = lda.get_cluster_from_df(X[0], X[1], n_clusters)
#     return labels

# def main(
#     n_clusters: int = 18,
#     runs: int = 10,
# ):
#     models = ["tfidf", "embedding", "lda", "random"]

#     df = pd.read_csv("data/transcripts_normalized.csv", header=0)
#     df.dropna(subset=["transcript"], inplace=True)

#     X_tfidf = tfidf.get_x(df)
#     X_embedding = embedding.get_x(df)
#     X_random = np.random.randint(0, n_clusters, size=len(df))
#     X_lda = lda.get_x(df)
#     X_runs = [X_tfidf, X_embedding, X_lda, X_random]

#     run_matrices = []

#     # create matrix of vi scores
#     for run in range(runs):
#         vi_scores = np.zeros((len(models), len(models)))
#         for i, model1 in tqdm(enumerate(models)):
#             for j, model2 in enumerate(models):
#                 model1_labels = run_n_clustering(n_clusters, X_runs[i], model1)
#                 model2_labels = run_n_clustering(n_clusters, X_runs[j], model2)
#                 vi_scores[i, j] = variation_of_information(model1_labels, model2_labels)
#         run_matrices.append(vi_scores)

#     # calculate mean vi scores
#     mean_vi_scores = np.mean(run_matrices, axis=0)
#     sns.heatmap(mean_vi_scores, annot=True, cmap="YlGnBu", xticklabels=models, yticklabels=models, fmt=".3f")
#     plt.title(f"VOI {n_clusters} clusters")
#     plt.savefig(f"vi_scores_{n_clusters}.png")
#     plt.close()

#     # create matrix of std vi scores
#     std_vi_scores = np.std(run_matrices, axis=0)
#     sns.heatmap(std_vi_scores, annot=True, cmap="YlGnBu", xticklabels=models, yticklabels=models, fmt=".3f")
#     plt.title(f"VOI std {n_clusters} clusters")
#     plt.savefig(f"vi_scores_std_{n_clusters}.png")
#     plt.close()

# if __name__ == "__main__":
#     tyro.cli(main)
