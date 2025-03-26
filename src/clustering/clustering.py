import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Cluster:
    id: int
    data: np.ndarray
    texts: list[str]
    count: int
    
    def sort(self):
        # Sort data based on the distance to the centroid
        centroid = np.mean(self.data, axis=0)
        distances = np.linalg.norm(self.data - centroid, axis=1)
        sorted_indices = np.argsort(distances)
        self.data = self.data[sorted_indices]
        self.texts = [self.texts[i] for i in sorted_indices]

    def __repr__(self):
        samples = "\n".join(self.texts[:10])
        return f"Cluster(id={self.id}, count={self.count})\nTop 10 samples:\n{samples}\n"


class Clustering(ABC):
    def __init__(
        self,
        X: np.ndarray,
        texts: list[str],
        random_state: int = 42,
    ):
        assert X.shape[0] == len(texts), "X and texts must have the same number of rows"
        self.X = X
        self.N = X.shape[0]
        self.texts = texts
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
