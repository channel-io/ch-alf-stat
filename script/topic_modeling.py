import argparse
import dotenv
import asyncio
import logging
import nest_asyncio
import torch
import numpy as np
import os

nest_asyncio.apply()
dotenv.load_dotenv()

from typing import Optional, List
from src.log_handler import LogHandler
from src.clustering import *
from src.embedder import EmbeddingExtractor
from src.clustering.utils import reduce_dimensions
from src.topic_model import ALFTopicModel

def main(
    logs_dir: str,
    start_date: str = None,
    end_date: str = None,
    subdirs: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
):
    # Prepare dataset
    log_handler = LogHandler(logs_dir, subdirs, start_date, end_date)
    log_handler.detect_language()
    logs = [t for t in log_handler.logs if t.with_knowledge and t.sent and t.language == "ko"]
    summaries = [t.summary for t in logs]
    
    # Handle empty summaries case
    if not summaries:
        logging.warning("No summaries found. Check your logs directory and subdirectories.")
        return

    # Embed dataset
    if cache_dir and os.path.exists(cache_dir):
        X = torch.load(cache_dir, weights_only=False)
        X = np.array(X)
    else:
        loop = asyncio.get_event_loop()
        try:
            X = loop.run_until_complete(EmbeddingExtractor.async_get_embeddings(summaries))
            X = reduce_dimensions(X, method="umap", n_components=64)
        finally:
            loop.close()    
            if cache_dir:
                torch.save(X, cache_dir)

    # Cluster dataset
    clustering = HdbscanClustering(
        X=X,
        logs=logs
    )
    clustering.fit()
    clustering.sort()
    clusters = clustering.clusters

    print(f"Found {len(clusters)} clusters")
    print(f"Found {len([cluster for cluster in clusters if cluster.count > 10])} clusters with more than 10 logs")

    # Evaluate clusters
    metrics = Clustering.evaluate_clusters(clusters)
    print(metrics)
    
    topic_model = ALFTopicModel(clusters, model="gpt-4o")
    keywords = topic_model.extract_keywords(clusters[0], n_keywords=10)
    print(keywords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--subdirs", nargs="+", default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO) 
    main(**vars(args))
