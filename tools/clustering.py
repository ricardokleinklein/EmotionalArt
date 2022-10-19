"""
Clustering methods
"""
import hdbscan
import numpy

from typing import Any, List, Dict, Union


class Hdbscan:
    def __init__(self, min_cluster_size: int, metric: str = 'euclidean') -> \
            None:
        """
        Perform HDBSCAN clustering.

        Args:
         min_cluster_size (min): Minimum amount of samples to set a cluster.
         metric (str): Distance measure.
        """
        self.method = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method='eom')

    def fit(self, X: numpy.ndarray) -> None:
        self.method.fit(X)

    def get_labels(self) -> numpy.ndarray:
        return self.method.labels_

    def get_core_samples(self) -> list:
        """
        Select only the most representative samples per cluster.

        Returns:
            list:
        """
        core_samples = []
        for cluster, core in enumerate(self.method.exemplars_):
            core_samples.append((cluster, len(core)))
        return core_samples

    def measure_cluster_size(self) -> List[tuple]:
        """
        Compute the number of labels for the elements
        unique.

        Returns:
            List:tuple
        """
        labels = self.get_labels()
        unique = sorted(list(set(labels)))
        clus_size = []
        for i, cluster in enumerate(unique[1:]):
            clus_size.append((cluster, len(numpy.where(labels == cluster)[0])))
        return clus_size

    def build_as_documents(self, text: Union[List[str], numpy.ndarray],
                           allow_noise: bool = False) -> Dict[int, str]:
        """
        Given a set of predicted labels, aggregate input text according to
        those labels to build document-like objects.

        Args:
            text (List[str]): Sample-wise text to aggregate.
            allow_noise (bool): Whether to remove noisy label (-1).

        Returns:
            Dict[int, str]
        """
        if isinstance(text, list):
            text = numpy.array(text)
        documents = dict()
        labels = self.get_labels()
        unique_labels = sorted(list(set(labels)))
        for i, cluster in enumerate(unique_labels):
            cluster_samples = numpy.where(labels == cluster)[0]
            cluster_text = text[cluster_samples]
            documents[cluster] = cluster_text
        if not allow_noise:
            del documents[-1]
        return documents

    def avg_score_by_label(self, scores: Union[list, numpy.ndarray],
                           allow_noise: bool = False) -> Dict[int, float]:
        """
        Using the outcome of the clustering, aggregate input scores by the
        predicted labels.

        Args:
            scores (Union[list, numpy.ndarray]): Scores to aggregate.

        Returns:
            Dict[int, float]

        NOTE: Scores must match the ordering of the data the method is
        trained on.
        """
        if isinstance(scores, list):
            scores = numpy.array(scores)
        labels = self.get_labels()
        unique_labels = sorted(list(set(labels)))
        score_per_label = dict()
        for i, cluster in enumerate(unique_labels):
            cluster_samples = numpy.where(labels == cluster)[0]
            cluster_scores = scores[cluster_samples]
            score_per_label[cluster] = cluster_scores
        if not allow_noise:
            del score_per_label[-1]
        return score_per_label
