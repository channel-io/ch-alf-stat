import numpy as np
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.model import ALFLog

@dataclass
class Cluster:
    id: int
    indices: list[int]
    data: np.ndarray
    texts: list[str]
    logs: list[ALFLog]
    count: int
    
    def sort(self):
        # Sort data based on the distance to the centroid
        centroid = np.mean(self.data, axis=0)
        distances = np.linalg.norm(self.data - centroid, axis=1)
        sorted_indices = np.argsort(distances)
        self.data = self.data[sorted_indices]
        self.texts = [self.texts[i] for i in sorted_indices]
        self.indices = [self.indices[i] for i in sorted_indices]
        
    def __repr__(self):
        samples = "\n".join(self.texts[:10])
        return f"Cluster(id={self.id}, count={self.count})\nTop 10 samples:\n{samples}\n"


class Clustering(ABC):
    def __init__(
        self,
        X: np.ndarray,
        logs: list[ALFLog],
        random_state: int = 42,
    ):
        assert X.shape[0] == len(logs), "X and logs must have the same number of rows"
        self.X = X
        self.N = X.shape[0]
        self.logs = logs
        self.texts = [log.summary for log in logs]
        self.random_state = random_state

    @abstractmethod
    def fit(self) -> list[Cluster]:
        """
        Fit the clustering model to the data.
        
        This method trains the clustering model on the input data X and assigns
        cluster labels to each data point. The implementation details depend on
        the specific clustering algorithm used by the subclass.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster labels for new data.
        
        This method assigns cluster labels to new data points based on the
        trained clustering model. The implementation details depend on the
        specific clustering algorithm used by the subclass.

        Args:
            X: New data points to predict cluster labels for.

        Returns:
            np.ndarray: Predicted cluster labels for the new data points.
        """
        pass

    @staticmethod
    def save_clusters(clusters: list[Cluster], path: str):
        assert path.endswith(".json"), "Path must end with .json"
        cluster_dict = []
        for cluster in clusters:
            cluster_dict.append({
                "id": cluster.id,
                "count": cluster.count,
                "texts": cluster.texts,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cluster_dict, f, ensure_ascii=False, indent=4)

    @staticmethod
    def evaluate_clusters(clusters: list[Cluster]) -> dict:
        """
        Calculate clustering evaluation metrics for the provided clusters.
        
        This method calculates the Silhouette Score, Davies-Bouldin Index, and
        Calinski-Harabasz Index for the clustering result. These metrics help evaluate
        the quality of the clustering.
        
        Args:
            clusters: List of Cluster objects containing the clustering results
            
        Returns:
            dict: Dictionary containing the calculated metrics
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        # Extract data points and cluster labels
        all_data = []
        labels = []
        
        for cluster in clusters:
            all_data.append(cluster.data)
            cluster_labels = [cluster.id] * len(cluster.data)
            labels.extend(cluster_labels)
        
        # Combine all data points
        X = np.vstack(all_data)
        labels = np.array(labels)
        
        # Calculate metrics
        metrics = {}
        
        # Only calculate if there are more than one cluster
        if len(clusters) > 1:
            # Silhouette Score (higher is better, range: -1 to 1)
            metrics["silhouette_score"] = float(silhouette_score(X, labels))
            # Davies-Bouldin Index (lower is better)
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, labels))
            # Calinski-Harabasz Index (higher is better)
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
        else:
            metrics["silhouette_score"] = None
            metrics["davies_bouldin_score"] = None
            metrics["calinski_harabasz_score"] = None
            
        return metrics