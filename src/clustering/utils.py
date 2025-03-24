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
