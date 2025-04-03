import numpy as np
import hdbscan

from .clustering import Clustering, Cluster
from typing import Optional
from src.model import ALFLog

class HdbscanClustering(Clustering):
    def __init__(
        self,
        X: np.ndarray,
        logs: list[ALFLog],
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        alpha: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the HDBSCAN clustering algorithm.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            texts: List of text corresponding to each data point
            min_cluster_size: The minimum size of clusters to form
            min_samples: The number of samples in a neighborhood for a point to be considered a core point
            metric: The metric to use for distance computation
            cluster_selection_method: The method to select flat clusters from the hierarchy
            alpha: A distance scaling parameter that adjusts how HDBSCAN forms clusters
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments to pass to the HDBSCAN constructor
        """
        super().__init__(X, logs, random_state)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.alpha = alpha
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
        self.clusters = None

    def fit(self) -> list[Cluster]:
        """
        Fit the HDBSCAN clustering model to the data.
        
        Returns:
            list[Cluster]: A list of Cluster objects, each representing a cluster.
        """
        # Initialize and fit the HDBSCAN model
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            alpha=self.alpha,
            **self.kwargs
        )
        self.labels_ = self.model.fit_predict(self.X)
        
        # Process the clusters
        unique_labels = np.unique(self.labels_)
        self.clusters = []
        
        # Create cluster objects, excluding noise points (label -1)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            mask = self.labels_ == label
            cluster_data = self.X[mask]
            cluster_texts = [self.texts[i] for i, is_member in enumerate(mask) if is_member]
            cluster_logs = [self.logs[i] for i, is_member in enumerate(mask) if is_member]

            cluster = Cluster(
                id=int(label),
                data=cluster_data,
                texts=cluster_texts,
                logs=cluster_logs,
                count=len(cluster_data),
                indices=np.where(mask)[0]
            )
            cluster.sort()  # Sort the cluster data by distance to centroid
            self.clusters.append(cluster)
            
        # Reassign IDs based on sorted order
        for i, cluster in enumerate(self.clusters):
            cluster.id = i
            
        return self.clusters

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster labels for new data.
        
        Args:
            X: New data points to predict cluster labels for.
            
        Returns:
            np.ndarray: Predicted cluster labels for the new data points.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
            
        # HDBSCAN doesn't have a native predict method, so we need to use approximate_predict
        labels, strengths = hdbscan.approximate_predict(self.model, X)
        return labels

    def sort(self):
        # Sort clusters by size (descending)
        self.clusters.sort(key=lambda x: x.count, reverse=True)
        for cluster in self.clusters:
            cluster.sort()