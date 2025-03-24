import argparse
import dotenv
import asyncio
import logging
import nest_asyncio

nest_asyncio.apply()
dotenv.load_dotenv()

from typing import Optional, List
from src.log_handler import LogHandler
from src.clustering.hybrid_cluster import HybridCluster
from src.embedder import EmbeddingExtractor
from src.clustering.utils import reduce_dimensions


def main(
    logs_dir: str,
    start_date: str = None,
    end_date: str = None,
    subdirs: Optional[List[str]] = None,
):
    # Prepare dataset
    log_handler = LogHandler(logs_dir, subdirs, start_date, end_date)
    summaries = [t["summary"] for t in log_handler.turns_kb]
    
    # Handle empty summaries case
    if not summaries:
        logging.warning("No summaries found. Check your logs directory and subdirectories.")
        return

    # Embed dataset
    loop = asyncio.get_event_loop()
    try:
        X = loop.run_until_complete(EmbeddingExtractor.async_get_embeddings(summaries))
        X = reduce_dimensions(X, method="umap", n_components=64)

        # Cluster dataset
        hybrid_cluster = HybridCluster(
            X=X,
            texts=summaries
        )
        hybrid_cluster.fit()
    finally:
        loop.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--subdirs", nargs="+", default=None)  # Remove type=list[str]
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    main(**vars(args))
