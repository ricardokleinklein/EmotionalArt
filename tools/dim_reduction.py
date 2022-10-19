"""
Tools aimed at carrying out reduction of dimensionality over embeddings.
"""
import numpy
import umap

from sklearn.decomposition import PCA

class Pca:
    seed = 1234
    def __init__(self, dims: int) -> None:
        """
        Wrapper for sklearn's PCA.
        Args:
            dims (int): Dimension of the projected space.
        """
        self.pca = PCA(dims, random_state=self.seed)
    def __call__(self, X: numpy.ndarray) -> numpy.ndarray:
        return self.pca.fit_transform(X)
    def transform(self, X: numpy.ndarray) -> numpy.ndarray:
        return self.pca.transform(X)



class Umap:
    seed = 1234

    def __init__(self, dims: int, nneighs: int, metric: str ='cosine') -> None:
        """
        Wrapper for UMAP from umap-learn.

        Args:
            dims (int): Dimension of the projected space.
            nneighs (int): Proximity parameter.
            metric (str): Distance measure.
        """
        self.umap = umap.UMAP(n_neighbors=nneighs, n_components=dims,
                              metric=metric, random_state=self.seed)

    def __call__(self, X: numpy.ndarray) -> numpy.ndarray:
        return self.umap.fit_transform(X)