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


def main(
    logs_dir: str,
    start_date: str = None,
    end_date: str = None,
    subdirs: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
):
    # Prepare dataset
    log_handler = LogHandler(logs_dir, subdirs, start_date, end_date)
    turns_kb, _ = log_handler.split_data_by_response_type()
    summaries = [t.summary for t in turns_kb]
    
    # Handle empty summaries case
    if not summaries:
        logging.warning("No summaries found. Check your logs directory and subdirectories.")
        return

    # Embed dataset
    if os.path.exists(cache_dir):
        X = torch.load(cache_dir, weights_only=False)
        X = np.array(X)
    else:
        loop = asyncio.get_event_loop()
        try:
            X = loop.run_until_complete(EmbeddingExtractor.async_get_embeddings(summaries))
            X = reduce_dimensions(X, method="umap", n_components=64)
            torch.save(X, cache_dir)
        finally:
            loop.close()    

    # Cluster dataset
    clustering = HdbscanClustering(
        X=X,
        texts=summaries
    )
    clusters = clustering.fit()


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
