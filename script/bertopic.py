import argparse
import numpy as np
from bertopic import BERTopic

from src.log_handler import LogHandler
from src.embedder import EmbeddingExtractor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

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
    logs = [t for t in log_handler.logs if t.with_knowledge and t.sent]
    summaries = [t.summary for t in logs]

    # Prepare sub-models that support online learning
    umap_model = IncrementalPCA(n_components=32)
    cluster_model = River(cluster.DBSTREAM())
    vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=.01)

    topic_model = BERTopic(
        language="multilingual",
        umap_model=umap_model,
        hdbscan_model=MiniBatchKMeans(n_clusters=100, random_state=0),
        vectorizer_model=vectorizer_model
    )

    # Incrementally fit the topic model by training on 1000 summaries at a time
    batch_size = 1000
    for index in range(0, len(summaries), batch_size):
        batch = summaries[index:index + batch_size]
        embeddings = EmbeddingExtractor.get_embeddings(batch).astype(np.float64)
        print(f"Processing batch {index // batch_size + 1}/{(len(summaries) + batch_size - 1) // batch_size}: {len(batch)} summaries")
        topic_model.partial_fit(batch, embeddings)
        # Get topics and their representations
        topic_model.get_topic(0)
        
    
    # Print topic information
    topic_info = topic_model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info)
    
    # Print top topics
    print("\nTop Topics:")
    for topic_id in topic_info.head(10)["Topic"]:
        if topic_id != -1:  # Skip outlier topic
            print(f"Topic {topic_id}: {topic_model.get_topic(topic_id)}")

if __name__ == "__main__":
    main()