import argparse
import numpy as np
from bertopic import BERTopic
import json
from src.log_handler import LogHandler
from src.embedder import EmbeddingExtractor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer, ClassTfidfTransformer
from river import stream
from river import cluster

class River:
    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--subdirs", nargs="+", default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    args = parser.parse_args()
    
     # Prepare dataset
    log_handler = LogHandler(
        args.logs_dir, 
        args.subdirs, 
        args.start_date, 
        args.end_date
    )
    log_handler.detect_language()
    logs = [t for t in log_handler.logs if t.with_knowledge and t.sent and t.language == "ko"]

    # Prepare sub-models that support online learning
    umap_model = IncrementalPCA(n_components=32)
    cluster_model = River(cluster.DBSTREAM(clustering_threshold=0.5))
    vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=.01)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

    topic_model = BERTopic(
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model
    )

    # Group logs by week and incrementally fit the topic model
    from datetime import datetime, timedelta
    
    # Sort logs by time
    logs_with_time = [(log, datetime.fromisoformat(log.time_out.split('.')[0])) for log in logs]
    logs_with_time.sort(key=lambda x: x[1])
    
    # Group by week
    weekly_batches = {}
    for log, log_time in logs_with_time:
        # Get the start of the week (Monday)
        week_start = log_time - timedelta(days=log_time.weekday())
        week_key = week_start.strftime('%Y-%m-%d')
        
        if week_key not in weekly_batches:
            weekly_batches[week_key] = []
        weekly_batches[week_key].append(log.summary)
    
    # Process each week
    weekly_topics = []
    for i, (week_key, batch) in enumerate(weekly_batches.items()):
        print(f"Processing week {i+1}/{len(weekly_batches)}: {week_key} with {len(batch)} summaries")
        embeddings = EmbeddingExtractor.get_embeddings(batch).astype(np.float64)
        topic_model.partial_fit(batch, embeddings)
        
        # Get topics and their representations
        weekly_topics.append(topic_model.get_topic_info().to_dict())

    # Save topics
    with open("script/weekly_topics.json", "w", encoding="utf-8") as f:
        json.dump(weekly_topics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()