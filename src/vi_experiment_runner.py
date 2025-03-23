import os
import pandas as pd
import numpy as np
import tyro
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from functools import partial
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

# Import clusterers
from tfidf_clusterer import TfidfClusterer
from embedding_clusterer import EmbeddingClusterer
from lda_clusterer import LdaClusterer
from w2v_tfidf_clusterer import W2vTfidfClusterer

def entropy(labels):
    """Calculate entropy of cluster labels"""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))


def variation_of_information(labels1, labels2):
    """Calculate variation of information between two clusterings"""
    h1 = entropy(labels1)
    h2 = entropy(labels2)
    mi = mutual_info_score(labels1, labels2)
    return h1 + h2 - 2 * mi  # eq 19 from paper


def run_clustering(
    clusterer_class,
    data_path: str,
    config: Dict[str, Any],
    df: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """Run a single clustering model and return labels"""
    # Create clusterer with config
    clusterer = clusterer_class(**config)
    
    # Load data if not provided
    if df is None:
        df = pd.read_csv(data_path)
        df.dropna(subset=["transcript"], inplace=True)
    
    # Run clustering
    df_result = clusterer.run(df)
    
    return clusterer.labels_


def run_random_clustering(n_clusters: int, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
    """Generate random cluster labels"""
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.randint(0, n_clusters, size=n_samples)


def run_vi_experiment(
    data_path: str,
    output_dir: str,
    methods: List[str],
    method_map: Dict[str, Any],
    configs: Dict[str, Dict[str, Any]],
    run_id: int,
    random_state_base: int
) -> np.ndarray:
    """Run a single VI experiment with all methods"""
    # Load data once
    df = pd.read_csv(data_path)
    df.dropna(subset=["transcript"], inplace=True)
    
    # Get number of samples for random clustering
    n_samples = len(df)
    
    # Run all clustering methods
    labels_dict = {}
    for method in methods:
        if method == "random":
            # For random clustering, use the run_id to modify the random state
            random_state = random_state_base + run_id if random_state_base is not None else None
            n_clusters = configs.get("random", {}).get("n_clusters", 18)
            labels = run_random_clustering(n_clusters, n_samples, random_state)
        else:
            # For other methods, update the random state in the config
            config = configs.get(method, {}).copy()
            if random_state_base is not None:
                config["random_state"] = random_state_base + run_id
            
            # Run the clustering
            labels = run_clustering(method_map[method], data_path, config, df)
        
        labels_dict[method] = labels
    
    # Calculate VI scores
    vi_scores = np.zeros((len(methods), len(methods)))
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            vi_scores[i, j] = variation_of_information(
                labels_dict[method1], 
                labels_dict[method2]
            )
    
    return vi_scores


def main(
    data_path: str = "data/normalized.csv",
    output_dir: str = "data/vi_experiments",
    experiment_name: str = "exp",
    config_path: Optional[str] = None,
    methods: List[str] = ["tfidf", "embedding", "lda", "w2v_tfidf", "random"],
    n_clusters: int = 18,
    n_words: int = 25,
    num_closest_to_center: int = 5,
    random_state: Optional[int] = 42,
    runs: int = 5,
    parallel: bool = True,
    n_jobs: int = -1,
    figsize: tuple = (12, 10),
):
    """Run VI experiments across multiple clustering methods"""
    output_dir = Path(output_dir) / experiment_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Define method to class mapping
    method_map = {
        "tfidf": TfidfClusterer,
        "embedding": EmbeddingClusterer,
        "lda": LdaClusterer,
        "w2v_tfidf": W2vTfidfClusterer,
        # "random" is handled separately
    }
    
    # Filter methods
    selected_methods = [m for m in methods if m == "random" or m in method_map]
    
    # Load config if provided, otherwise use CLI args
    if config_path:
        with open(config_path, 'r') as f:
            configs = json.load(f)
    else:
        # Create default config for each method
        configs = {
            method: {
                "n_clusters": n_clusters,
                "n_words": n_words,
                "num_closest_to_center": num_closest_to_center,
                "random_state": random_state
            }
            for method in selected_methods
        }
    
    # Prepare experiment tasks
    tasks = []
    for run_id in range(runs):
        tasks.append((
            data_path,
            output_dir,
            selected_methods,
            method_map,
            configs,
            run_id,
            random_state
        ))
    
    # Run experiments
    print(f"Running {runs} VI experiments with methods: {', '.join(selected_methods)}")
    
    vi_matrices = []
    if parallel and runs > 1:
        # Run in parallel
        n_processes = mp.cpu_count() if n_jobs == -1 else n_jobs
        n_processes = min(n_processes, runs)
        
        print(f"Running in parallel with {n_processes} processes")
        
        with mp.Pool(processes=n_processes) as pool:
            run_func = partial(run_vi_experiment)
            vi_matrices = list(tqdm(pool.starmap(run_func, tasks), total=runs))
    else:
        # Run sequentially
        print(f"Running sequentially")
        for task in tqdm(tasks):
            vi_matrices.append(run_vi_experiment(*task))
    
    # Calculate mean and std of VI scores
    mean_vi = np.mean(vi_matrices, axis=0)
    std_vi = np.std(vi_matrices, axis=0)
    
    # Save raw results
    np.save(os.path.join(output_dir, "vi_matrices.npy"), np.array(vi_matrices))
    np.save(os.path.join(output_dir, "mean_vi.npy"), mean_vi)
    np.save(os.path.join(output_dir, "std_vi.npy"), std_vi)
    
    # Create heatmaps
    plt.figure(figsize=figsize)
    sns.heatmap(
        mean_vi,
        annot=True,
        cmap="YlGnBu",
        xticklabels=selected_methods,
        yticklabels=selected_methods,
        fmt=".3f",
    )
    plt.title(f"Mean Variation of Information ({n_clusters} clusters, {runs} runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"vi_mean_{n_clusters}.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        std_vi,
        annot=True,
        cmap="YlGnBu",
        xticklabels=selected_methods,
        yticklabels=selected_methods,
        fmt=".3f",
    )
    plt.title(f"Standard Deviation of Variation of Information ({n_clusters} clusters, {runs} runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"vi_std_{n_clusters}.png"), dpi=300)
    plt.close()
    
    # Create normalized VI heatmap (divide by max possible VI)
    # Max possible VI is log(n_clusters) * 2
    max_vi = np.log(n_clusters) * 2
    norm_mean_vi = mean_vi / max_vi
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        norm_mean_vi,
        annot=True,
        cmap="YlGnBu",
        xticklabels=selected_methods,
        yticklabels=selected_methods,
        fmt=".3f",
        vmin=0,
        vmax=1,
    )
    plt.title(f"Normalized Mean Variation of Information ({n_clusters} clusters, {runs} runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"vi_norm_mean_{n_clusters}.png"), dpi=300)
    plt.close()
    
    # Save summary as text
    with open(os.path.join(output_dir, "vi_summary.txt"), 'w') as f:
        f.write(f"VI Experiment Summary\n")
        f.write(f"====================\n\n")
        f.write(f"Methods: {', '.join(selected_methods)}\n")
        f.write(f"Number of clusters: {n_clusters}\n")
        f.write(f"Number of runs: {runs}\n")
        f.write(f"Random state base: {random_state}\n\n")
        
        f.write(f"Mean VI Scores:\n")
        for i, method1 in enumerate(selected_methods):
            for j, method2 in enumerate(selected_methods):
                f.write(f"{method1} vs {method2}: {mean_vi[i, j]:.4f} ± {std_vi[i, j]:.4f}\n")
            f.write("\n")
    
    print(f"VI experiment results saved to {output_dir}")
    
    # Return the mean VI matrix
    return mean_vi


if __name__ == "__main__":
    tyro.cli(main)
    