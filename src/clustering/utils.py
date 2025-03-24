import numpy as np
import warnings
import logging

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

def reduce_dimensions(
    embeddings: np.ndarray, 
    method: str = "umap",
    n_components: int = 64, 
    random_state: int = 42
):
    """
    Reduce the dimensions of the embeddings.
    Available methods: "umap", "pca", "tsne"
    """
    logger.info(f"Reducing dimensions of {embeddings.shape[0]} embeddings using {method} method")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method.lower() == "umap":
            reducer = UMAP(n_components=n_components, metric="cosine", random_state=random_state)
        elif method.lower() == "pca":
            reducer = PCA(n_components=n_components)
        elif method.lower() == "tsne":
            reducer = TSNE(n_components=n_components, random_state=random_state)
        else:
            raise ValueError(f"Invalid reduction method: {method}")
        reduced = reducer.fit_transform(embeddings)
    return np.array(reduced)


def compute_shrinkage_lambda(cov, n_samples, shrinkage_strength=0.5):
    d = cov.shape[0]
    sample_penalty = np.exp(- (n_samples / (d + 1)))
    cond_penalty = np.tanh(np.log10(np.linalg.cond(cov)))
    lam = shrinkage_strength * (sample_penalty + cond_penalty) / 2.0
    lam = np.clip(lam, 0.0, 1.0)
    return lam


def mahalanobis(
    cov: np.ndarray,
    diff: np.ndarray,
    n_samples: int,
    shrinkage_strength: float = 0.5
) -> float:
    lam = compute_shrinkage_lambda(cov, n_samples, shrinkage_strength)
    blended_cov = lam * cov + (1 - lam) * np.eye(cov.shape[0])
    return np.sqrt(np.dot(diff, np.dot(np.linalg.inv(blended_cov), diff)))


def interpolated_mahalanobis(
    cluster1: np.ndarray,
    cluster2: np.ndarray,
    strategy: str = 'pooled'
) -> float:
    """
    Compute the Mahalanobis distance between two clusters
    cluster1: numpy array of shape (n_samples, n_features) 
    cluster2: numpy array of shape (n_samples, n_features)
    strategy: 'pooled' or 'bidirectional'
    """
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)

    if strategy == 'pooled':
        pooled_cov = np.cov(np.concatenate([cluster1, cluster2], axis=0), rowvar=False)
        diff = centroid1 - centroid2
        return mahalanobis(pooled_cov, diff, len(cluster1) + len(cluster2))
    
    elif strategy == 'bidirectional':
        diff = centroid1 - centroid2
        dist1 = mahalanobis(np.cov(cluster1, rowvar=False), diff, len(cluster1))
        dist2 = mahalanobis(np.cov(cluster2, rowvar=False), -diff, len(cluster2))
        return (dist1 + dist2) / 2
    
    else:
        raise ValueError(f"Invalid strategy: {strategy}")