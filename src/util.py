from collections import Counter


def find_topk_words(df, km, k, remove_custom_stops: bool = True):
    custom_stops = []
    if remove_custom_stops:
        with open("data/custom_stopwords.txt", "r") as f:
            custom_stops = f.read().splitlines()

    cluster_words = []
    for i in range(km.n_clusters):
        cluster_mask = km.labels_ == i
        cluster_docs = df[cluster_mask]["transcript"].tolist()
        words = " ".join(cluster_docs).split()
        wc = Counter(words)
        if remove_custom_stops:
            for stop in custom_stops:
                del wc[stop]

        top_words = wc.most_common(k)
        cluster_words.append(top_words)

    return cluster_words


def save_topk_words(file_prefix, df, kms, k, rg, remove_custom_stops: bool = True):
    custom_stops = []
    if remove_custom_stops:
        with open("../data/custom_stopwords.txt", "r") as f:
            custom_stops = f.read().splitlines()

    for nc in range(rg[0], rg[1], rg[2]):
        text_buffer = ""
        cluster_words = []
        idx = nc - rg[0]
        for i in range(nc):
            cluster_mask = kms[idx].labels_ == i
            cluster_docs = df[cluster_mask]["transcript"].tolist()

            words = " ".join(cluster_docs).split()
            wc = Counter(words)
            if remove_custom_stops:
                for stop in custom_stops:
                    del wc[stop]

            top_words = wc.most_common(k)
            cluster_words.append(top_words)

            text_buffer += f"Cluster {i} top {k} words:\n"
            for w, c in top_words:
                text_buffer += f"{w}: {c}\n"
            text_buffer += "\n"

        with open(f"{file_prefix}_{nc}_{k}.txt", "+w") as f:
            f.write(text_buffer)
