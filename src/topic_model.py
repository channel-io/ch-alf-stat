import logging
import numpy as np

from typing import Optional, List

from src.clustering.clustering import Cluster
from src.llm import llm_factory
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class ALFTopicModel:
    def __init__(
        self,
        clusters: list[Cluster],
        model: str = "gpt-4o",
    ):
        self.clusters = clusters
        self.llm = llm_factory(model)

    def remove_noise(self):
        pass

    def scatterplot(self):
        pass
    
    def explain(self):
        pass

    def extract_keywords(self, cluster: Cluster, n_keywords: int = 10) -> list[str]:
        texts = cluster.texts
        if not texts:
            logger.warning(f"Cluster {cluster.id} has no texts")
            return []
        
        vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=2,
            stop_words='english',
            lowercase=True,
            use_idf=True,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = avg_tfidf_scores.argsort()[-n_keywords:][::-1]
        return [feature_names[i] for i in top_indices]

    def extract_topics(self, clusters: list[Cluster] = None) -> list[str]:
        pass
