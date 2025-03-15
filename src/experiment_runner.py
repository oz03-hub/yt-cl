import os
import pandas as pd
import tyro
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from functools import partial
import time
import json
import numpy as np
from pathlib import Path

from tfidf_clusterer import TfidfClusterer
from embedding_clusterer import EmbeddingClusterer
from lda_clusterer import LdaClusterer
from w2v_tfidf_clusterer import W2vTfidfClusterer

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def run_experiment(
    clusterer_class,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single clustering experiment"""
    start_time = time.time()
    
    # Create clusterer with config
    clusterer = clusterer_class(**config)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Run clustering
    df_result = clusterer.run(df)
    
    # Generate and save report
    report = clusterer.generate_report(df_result)
    method_output_dir = os.path.join(output_dir, clusterer.method_name)
    clusterer.save_report(report, method_output_dir)
    
    # Save results
    results = {
        "method": clusterer.method_name,
        "config": config,
        "metrics": clusterer.metrics,
        "runtime": time.time() - start_time
    }

    results = convert_to_serializable(results)
    
    return results


def main(
    data_path: str = "data/normalized.csv",
    output_dir: str = "data/experiments",
    experiment_name: str = "exp",
    config_path: Optional[str] = None,
    methods: List[str] = ["tfidf", "embedding", "lda", "w2v_tfidf"],
    n_clusters: int = 18,
    n_words: int = 25,
    num_closest_to_center: int = 5,
    random_state: int = 42,
    parallel: bool = True,
    n_jobs: int = -1,
):
    """Run clustering experiments"""
    output_dir = Path(output_dir) / experiment_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Define method to class mapping
    method_map = {
        "tfidf": TfidfClusterer,
        "embedding": EmbeddingClusterer,
        "lda": LdaClusterer,
        "w2v_tfidf": W2vTfidfClusterer
    }
    
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
            for method in methods
        }
    
    # Filter methods
    selected_methods = [m for m in methods if m in method_map]
    
    # Prepare experiment tasks
    tasks = []
    for method in selected_methods:
        if method in configs:
            config = configs[method]
            tasks.append((method_map[method], data_path, output_dir, config))
    
    # Run experiments
    results = []
    if parallel and len(tasks) > 1:
        # Run in parallel
        n_processes = mp.cpu_count() if n_jobs == -1 else n_jobs
        n_processes = min(n_processes, len(tasks))
        
        print(f"Running {len(tasks)} experiments in parallel with {n_processes} processes")
        
        with mp.Pool(processes=n_processes) as pool:
            run_func = partial(run_experiment)
            results = pool.starmap(run_func, tasks)
    else:
        # Run sequentially
        print(f"Running {len(tasks)} experiments sequentially")
        for task in tasks:
            results.append(run_experiment(*task))
    
    # Save overall results
    with open(os.path.join(output_dir, "experiment_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nExperiment Results Summary:")
    for result in results:
        print(f"\n{result['method']} ({result['runtime']:.2f}s):")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    tyro.cli(main)
