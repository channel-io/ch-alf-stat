import numpy as np
import logging

from src.clustering.clustering import Clustering
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Union, override

logger = logging.getLogger(__name__)


class HybridClustering(Clustering):
    """
    A hybrid clustering approach that combines hierarchical clustering with distance-based merging.
    
    This method initially uses hierarchical clustering to form clusters, then refines them
    by merging clusters based on Mahalanobis distance. This approach helps overcome the
    limitations of traditional clustering methods when dealing with clusters of different
    shapes, sizes, and covariances.
    
    The algorithm works in two phases:
    1. Initial clustering using hierarchical clustering with Ward's method
    2. Cluster refinement by merging clusters based on Mahalanobis distance
    """
    def __init__(
        self, 
        X: np.ndarray,
        texts: list[str],
        random_state: int = 42,
        noise_threshold: int = 10,
    ):
        super().__init__(X, texts, random_state)
        self.noise_threshold = noise_threshold
        self.Z = linkage(self.X, method='ward')
        logger.info(f"Hybrid clustering initialized with {self.N} samples")


    def _initial_clustering(self, t_fine: Union[float, int] = 0.5):
        logger.info("Initializing clusters")

        # Initial cutoff of hierarchical tree
        # If t_fine is integer (t_fine > 1), then it is the number of clusters
        # If t_fine is float (0 < t_fine < 1), then it is the distance threshold
        if t_fine > 1:
            initial_labels = fcluster(self.Z, t_fine, criterion='maxclust')
        else:
            initial_labels = fcluster(self.Z, t_fine, criterion='distance')

        # Build a mapping from cluster label to the list of sample indices in that cluster.
        cluster_indices = {}
        for idx, label in enumerate(initial_labels):
            if label not in cluster_indices:
                cluster_indices[label] = [idx]
            else:
                cluster_indices[label].append(idx)

        noise_clusters = [c for c in cluster_indices.keys() if len(cluster_indices[c]) > self.noise_threshold]
        logger.info(f"Found {len(noise_clusters)} noise clusters (less than {self.noise_threshold} samples)")

        self.labels = initial_labels
        self.clusters = cluster_indices
        self.parent_map = self._get_parent_map()

    def _get_parent_map(self):
        logger.info("Mapping parent labels")
        parent_map = {}
        for i, (child1, child2, _, _) in enumerate(self.Z):
            new_cluster = self.N + i  # new cluster index
            parent_map[int(child1)] = new_cluster
            parent_map[int(child2)] = new_cluster
        
        # Filter parent_map to only include child-parent mapping for labels in self.labels
        # Filter parent_map to include labels in self.labels or new clusters (label > self.N)
        filtered_parent_map = {}
        for label in parent_map:
            if label in np.unique(self.labels) or label >= self.N:
                filtered_parent_map[label] = parent_map[label]
        
        parent_map = filtered_parent_map
        return parent_map
        
    def _refine_clusters(self, t_distance: float = 2.0):
        pass

    @override
    def fit(self):
        self._initial_clustering(t_fine=0.5)
        self._refine_clusters()


    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Predict method not implemented for HybridCluster")
