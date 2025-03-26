import numpy as np
import logging

from src.clustering.clustering import Clustering, Cluster
from scipy.cluster.hierarchy import linkage, to_tree, ClusterNode
from typing import Union, override, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HybridNode(ClusterNode):
    def __init__(
        self,
        node: ClusterNode,
        labels: Optional[np.ndarray] = None,
    ):
        super().__init__(node.id, node.left, node.right, node.dist, node.count)
        self.labels = labels
        
        # No longer storing the actual data, just references
        self.cov = None
        self.mean = None
    
    def compute_stats(self, X: np.ndarray):
        """Compute covariance and mean using the data indexed by labels"""
        if self.labels is not None and len(self.labels) > 0:
            data = X[self.labels]
            self.cov = np.cov(data, rowvar=False)
            self.mean = np.mean(data, axis=0)


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


    @override
    def fit(self, t_dist: float = 0.5, t_merge: float = 0.5) -> list[Cluster]:
        root, _ = to_tree(self.Z, rd=True)

        hybrid_nodes = {}
        self.pruned = {"dist": 0, "merge": 0}
        
        def convert_node(node):
            if node is None:
                return None, None
            
            if node.id in hybrid_nodes:
                hybrid_node = hybrid_nodes[node.id]
                return hybrid_node, hybrid_node.labels if hybrid_node else None

            if node.is_leaf():
                labels = np.array([node.id])
                hybrid_node = HybridNode(node, labels)
                hybrid_node.compute_stats(self.X)
                hybrid_nodes[node.id] = hybrid_node
                return hybrid_node, labels
            
            left_hybrid, left_labels = convert_node(node.left)
            right_hybrid, right_labels = convert_node(node.right)
                        
            if left_labels is not None and right_labels is not None:
                labels = np.concatenate([left_labels, right_labels])
            elif left_labels is not None:
                labels = left_labels
            elif right_labels is not None:
                labels = right_labels
            else:
                labels = None

            # Check if both child nodes exist and should be merged based on Mahalanobis distance
            if left_hybrid is not None and right_hybrid is not None:
                if left_hybrid.labels is not None and right_hybrid.labels is not None:
                    left_data = self.X[left_hybrid.labels]
                    right_data = self.X[right_hybrid.labels]
                    
                    maha_dist = self.general_mahalanobis(
                        left_data, 
                        right_data, 
                        strategy='pooled'
                    )
                    
                    # If the distance is below threshold, merge them by making them None
                    if maha_dist < t_merge:
                        hybrid_nodes[left_hybrid.id] = None
                        hybrid_nodes[right_hybrid.id] = None
                        left_hybrid = None
                        right_hybrid = None
                        self.pruned["dist"] += 1

            hybrid_node = HybridNode(node, labels)
            hybrid_node.compute_stats(self.X)
            
            # Check if this node should be pruned based on distance threshold
            if node.dist < t_dist:
                hybrid_nodes[node.id] = None
                self.pruned["merge"] += 1
                return None, labels
            
            hybrid_nodes[node.id] = hybrid_node
            hybrid_node.left = left_hybrid
            hybrid_node.right = right_hybrid
            
            return hybrid_node, labels
        
        hybrid_root, _ = convert_node(root)
        
        self.root = hybrid_root
        self.nodes = hybrid_nodes

        self.clusters = []

        def count_leaf_nodes(node):
            if node is None:
                return 0
            
            if node.left is None and node.right is None:
                data = self.X[node.labels]
                texts = [self.texts[idx] for idx in node.labels]
                cluster = Cluster(id=node.id, data=data, texts=texts, count=len(node.labels), indices=node.labels)
                self.clusters.append(cluster)
                return 1
            
            return count_leaf_nodes(node.left) + count_leaf_nodes(node.right)
        
        self.num_leaves = count_leaf_nodes(self.root)
        logger.info(f"Hybrid clustering created with {self.num_leaves} leaf nodes")

        return self.clusters


    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Predict method not implemented for HybridCluster")
    
    def sort(self):
        self.clusters.sort(key=lambda x: x.count, reverse=True)  # Sort by count, largest first
        for cluster in self.clusters:
            cluster.sort()


    @staticmethod
    def compute_shrinkage_lambda(cov, n_samples, shrinkage_strength=0.5):
        d = cov.shape[0]
        sample_penalty = np.exp(- (n_samples / (d + 1)))
        cond_penalty = np.tanh(np.log10(np.linalg.cond(cov)))
        lam = shrinkage_strength * (sample_penalty + cond_penalty) / 2.0
        lam = np.clip(lam, 0.0, 1.0)
        return lam


    @staticmethod
    def interpolated_mahalanobis(
        cov: np.ndarray,
        diff: np.ndarray,
        n_samples: int,
        shrinkage_strength: float = 0.5
    ) -> float:
        lam = HybridClustering.compute_shrinkage_lambda(cov, n_samples, shrinkage_strength)
        blended_cov = lam * cov + (1 - lam) * np.eye(cov.shape[0])
        return np.sqrt(np.dot(diff, np.dot(np.linalg.inv(blended_cov), diff)))


    @staticmethod
    def general_mahalanobis(
        cluster1: np.ndarray,
        cluster2: np.ndarray,
        strategy: str = 'pooled',
        shrinkage_strength: float = 0.5
    ) -> float:
        """
        Compute the Mahalanobis distance between two clusters
        """
        centroid1 = np.mean(cluster1, axis=0)
        centroid2 = np.mean(cluster2, axis=0)

        if strategy == 'pooled':
            pooled_cov = np.cov(np.concatenate([cluster1, cluster2], axis=0), rowvar=False)
            diff = centroid1 - centroid2
            return HybridClustering.interpolated_mahalanobis(pooled_cov, diff, len(cluster1) + len(cluster2), shrinkage_strength)
        
        elif strategy == 'bidirectional':
            diff = centroid1 - centroid2
            cov1 = np.cov(cluster1, rowvar=False)
            cov2 = np.cov(cluster2, rowvar=False)
            dist1 = HybridClustering.interpolated_mahalanobis(cov1, diff, len(cluster1), shrinkage_strength)
            dist2 = HybridClustering.interpolated_mahalanobis(cov2, -diff, len(cluster2), shrinkage_strength)
            return (dist1 + dist2) / 2
        
        else:
            raise ValueError(f"Invalid strategy: {strategy}")